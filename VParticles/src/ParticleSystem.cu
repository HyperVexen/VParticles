#include "VParticles/ParticleSystem.h"

#include "VParticles/CudaCheck.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vparticles {
namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kTargetBlocksPerMultiprocessor = 8;
constexpr size_t kTelemetryRingSize = 8;

template <typename T>
void cudaAlloc(T*& pointer, size_t count)
{
    VP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
}

uint32_t blockCount(uint32_t itemCount)
{
    return static_cast<uint32_t>(
        (static_cast<uint64_t>(itemCount) + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

void allocatePool(ParticlePool& pool, uint32_t capacity)
{
    cudaAlloc(pool.pos, capacity);
    cudaAlloc(pool.vel, capacity);
    cudaAlloc(pool.color, capacity);
    cudaAlloc(pool.age, capacity);
    cudaAlloc(pool.lifetime, capacity);
    cudaAlloc(pool.systemId, capacity);
    pool.activeIndices = nullptr;
    pool.capacity = capacity;
    pool.aliveCount = 0;
}

void releasePool(ParticlePool& pool)
{
    cudaFree(pool.pos);
    cudaFree(pool.vel);
    cudaFree(pool.color);
    cudaFree(pool.age);
    cudaFree(pool.lifetime);
    cudaFree(pool.systemId);
    pool = {};
}

__device__ uint32_t mixBits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

__device__ float random01(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    const uint32_t mixed = mixBits(seed ^ (particle * 0x9E3779B9u) ^ (frame * 0x85EBCA6Bu) ^ lane);
    return static_cast<float>(mixed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ float randomSigned(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    return random01(seed, particle, frame, lane) * 2.0f - 1.0f;
}

// This state is intentionally device-resident. Kernels read the current active
// count and active-list pointer directly, so CPU submission never needs an
// exact lifecycle readback to choose the next frame's work.
struct DeviceFrameState {
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t activeCount = 0;
    uint32_t freeCount = 0;
    uint32_t deadCount = 0;
    uint32_t compactedCount = 0;
    uint32_t spawnStart = 0;
    uint32_t spawnCount = 0;
    uint32_t freeStart = 0;
    uint32_t requestedSpawn = 0;
    uint32_t frameIndex = 0;
    uint32_t activeBufferIndex = 0;
};

struct DeviceTelemetry {
    uint32_t aliveCount = 0;
    uint32_t requestedSpawn = 0;
    uint32_t spawned = 0;
    uint32_t dropped = 0;
    uint32_t deadCount = 0;
    uint32_t frameIndex = 0;
    uint32_t activeBufferIndex = 0;
};

__global__ void initializeFreeListKernel(uint32_t* freeIndices, uint32_t capacity)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < capacity) {
        freeIndices[index] = index;
    }
}

__global__ void beginFrameKernel(
    DeviceFrameState* state,
    uint32_t requestedSpawn,
    uint32_t frameIndex)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        state->deadCount = 0;
        state->compactedCount = 0;
        state->spawnStart = 0;
        state->spawnCount = 0;
        state->freeStart = 0;
        state->requestedSpawn = requestedSpawn;
        state->frameIndex = frameIndex;
    }
}

struct SpawnCommand {
    EmitterDesc emitter = {};
    uint32_t systemId = 0;
    uint32_t requestStart = 0;
    uint32_t count = 0;
};

__device__ uint32_t findSpawnCommand(
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t spawnOffset)
{
    uint32_t first = 0;
    uint32_t last = commandCount;
    while (first + 1 < last) {
        const uint32_t middle = first + (last - first) / 2;
        if (commands[middle].requestStart <= spawnOffset) {
            first = middle;
        } else {
            last = middle;
        }
    }

    return first;
}

__global__ void simulateKernel(
    ParticlePool pool,
    DeviceFrameState* state,
    uint8_t* aliveFlags,
    uint32_t* freeIndices,
    SimulationSettings settings,
    float dt)
{
    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t activeIndex = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         activeIndex < count;
         activeIndex += stride) {
        const uint32_t compactIndex = static_cast<uint32_t>(activeIndex);
        const uint32_t particleIndex = activeIndices[compactIndex];
        const float age = pool.age[particleIndex] + dt;
        const float lifetime = pool.lifetime[particleIndex];
        if (age >= lifetime) {
            pool.age[particleIndex] = age;
            aliveFlags[compactIndex] = 0;
            const uint32_t freeSlot = atomicAdd(&state->freeCount, 1u);
            freeIndices[freeSlot] = particleIndex;
            atomicAdd(&state->deadCount, 1u);
            continue;
        }

        float4 velocity = pool.vel[particleIndex];
        float4 position = pool.pos[particleIndex];
        const float dragFactor = fmaxf(0.0f, 1.0f - settings.drag * dt);

        velocity.x = (velocity.x + (settings.gravity.x + settings.wind.x) * dt) * dragFactor;
        velocity.y = (velocity.y + (settings.gravity.y + settings.wind.y) * dt) * dragFactor;
        velocity.z = (velocity.z + (settings.gravity.z + settings.wind.z) * dt) * dragFactor;

        position.x += velocity.x * dt;
        position.y += velocity.y * dt;
        position.z += velocity.z * dt;

        float4 color = pool.color[particleIndex];
        color.w = fmaxf(0.0f, 1.0f - age / lifetime);

        pool.vel[particleIndex] = velocity;
        pool.pos[particleIndex] = position;
        pool.color[particleIndex] = color;
        pool.age[particleIndex] = age;
        aliveFlags[compactIndex] = 1;
    }
}

// CUB's block scan lets each tile reserve one contiguous output range. This
// avoids a CPU readback and avoids one global atomic per surviving particle.
__global__ void compactActiveIndicesKernel(
    DeviceFrameState* state,
    const uint8_t* aliveFlags)
{
    if (state->deadCount == 0) {
        return;
    }

    using BlockScan = cub::BlockScan<uint32_t, kThreadsPerBlock>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ uint32_t outputStart;

    const uint32_t count = state->activeCount;
    const uint32_t* activeIndices = state->activeIndices;
    uint32_t* compactedIndices = state->scratchActiveIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t tileStart = static_cast<uint64_t>(blockIdx.x) * blockDim.x;
         tileStart < count;
         tileStart += stride) {
        const uint64_t activeIndex = tileStart + threadIdx.x;
        const bool inRange = activeIndex < count;
        const uint32_t isAlive = inRange ? static_cast<uint32_t>(aliveFlags[activeIndex]) : 0u;
        uint32_t localOffset = 0;
        uint32_t selectedInTile = 0;
        BlockScan(scanStorage).ExclusiveSum(isAlive, localOffset, selectedInTile);

        if (threadIdx.x == 0) {
            outputStart = selectedInTile == 0
                ? 0u
                : atomicAdd(&state->compactedCount, selectedInTile);
        }
        __syncthreads();

        if (isAlive != 0) {
            compactedIndices[outputStart + localOffset] = activeIndices[activeIndex];
        }
        __syncthreads();
    }
}

__global__ void finalizeCompactionKernel(DeviceFrameState* state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0 && state->deadCount != 0) {
        uint32_t* oldActiveIndices = state->activeIndices;
        state->activeIndices = state->scratchActiveIndices;
        state->scratchActiveIndices = oldActiveIndices;
        state->activeCount = state->compactedCount;
        state->activeBufferIndex ^= 1u;
    }
}

__global__ void prepareSpawnKernel(DeviceFrameState* state)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const uint32_t availableSlots = state->freeCount;
        const uint32_t spawned = min(state->requestedSpawn, availableSlots);

        state->spawnStart = state->activeCount;
        state->spawnCount = spawned;
        state->freeStart = availableSlots;
        state->freeCount = availableSlots - spawned;
        state->activeCount += spawned;
    }
}

__global__ void spawnBatchKernel(
    ParticlePool pool,
    const DeviceFrameState* state,
    const uint32_t* freeIndices,
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t seed)
{
    const uint32_t totalSpawn = state->spawnCount;
    const uint32_t firstActiveIndex = state->spawnStart;
    const uint32_t firstFreeSlot = state->freeStart;
    const uint32_t frameIndex = state->frameIndex;
    uint32_t* activeIndices = state->activeIndices;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;

    for (uint64_t spawnOffset = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         spawnOffset < totalSpawn;
         spawnOffset += stride) {
        const uint32_t compactOffset = static_cast<uint32_t>(spawnOffset);
        const uint32_t activeIndex = firstActiveIndex + compactOffset;
        const uint32_t commandIndex = findSpawnCommand(commands, commandCount, compactOffset);
        const SpawnCommand command = commands[commandIndex];
        const EmitterDesc emitter = command.emitter;
        const uint32_t particleIndex = freeIndices[firstFreeSlot - 1u - compactOffset];
        activeIndices[activeIndex] = particleIndex;

        const uint32_t randomIndex = activeIndex;
        const float vx = emitter.velocity.x + emitter.velocityVariance.x * randomSigned(seed, randomIndex, frameIndex, 1u);
        const float vy = emitter.velocity.y + emitter.velocityVariance.y * randomSigned(seed, randomIndex, frameIndex, 2u);
        const float vz = emitter.velocity.z + emitter.velocityVariance.z * randomSigned(seed, randomIndex, frameIndex, 3u);
        const float lifetimeJitter = emitter.lifetimeVariance * randomSigned(seed, randomIndex, frameIndex, 4u);
        const float particleLifetime = fmaxf(0.001f, emitter.lifetime * (1.0f + lifetimeJitter));

        pool.pos[particleIndex] = make_float4(emitter.position.x, emitter.position.y, emitter.position.z, 1.0f);
        pool.vel[particleIndex] = make_float4(vx, vy, vz, 0.0f);
        pool.color[particleIndex] = make_float4(emitter.color.r, emitter.color.g, emitter.color.b, emitter.color.a);
        pool.age[particleIndex] = 0.0f;
        pool.lifetime[particleIndex] = particleLifetime;
        pool.systemId[particleIndex] = command.systemId;
    }
}

__global__ void snapshotTelemetryKernel(
    const DeviceFrameState* state,
    DeviceTelemetry* telemetry)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        telemetry->aliveCount = state->activeCount;
        telemetry->requestedSpawn = state->requestedSpawn;
        telemetry->spawned = state->spawnCount;
        telemetry->dropped = state->requestedSpawn - state->spawnCount;
        telemetry->deadCount = state->deadCount;
        telemetry->frameIndex = state->frameIndex;
        telemetry->activeBufferIndex = state->activeBufferIndex;
    }
}

struct EmitterState {
    EmitterDesc desc = {};
    float spawnCarry = 0.0f;
    uint32_t pendingBurst = 0;
};

} // namespace

struct ParticleSystem::Impl {
    struct TelemetrySlot {
        DeviceTelemetry* hostTelemetry = nullptr;
        cudaEvent_t frameStart = nullptr;
        cudaEvent_t afterSimulate = nullptr;
        cudaEvent_t afterCompact = nullptr;
        cudaEvent_t afterSpawn = nullptr;
        cudaEvent_t ready = nullptr;
        uint64_t spawnUpperBoundThroughFrame = 0;
        uint32_t frameIndex = 0;
        bool inFlight = false;
    };

    explicit Impl(uint32_t capacity)
    {
        if (capacity == 0) {
            throw std::invalid_argument("ParticleSystem capacity must be greater than zero");
        }

        allocatePool(pool, capacity);
        cudaAlloc(activeIndices, capacity);
        cudaAlloc(scratchActiveIndices, capacity);
        cudaAlloc(freeIndices, capacity);
        cudaAlloc(aliveFlags, capacity);
        cudaAlloc(deviceState, 1);
        cudaAlloc(deviceTelemetryScratch, 1);
        pool.activeIndices = activeIndices;

        VP_CUDA_CHECK(cudaStreamCreate(&stream));
        createTelemetrySlots();
        VP_CUDA_CHECK(cudaEventCreate(&latestFrameStart));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterSimulate));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterCompact));
        VP_CUDA_CHECK(cudaEventCreate(&latestAfterSpawn));

        int device = 0;
        cudaDeviceProp properties = {};
        VP_CUDA_CHECK(cudaGetDevice(&device));
        VP_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        maxDispatchBlocks = std::max(
            1u,
            static_cast<uint32_t>(properties.multiProcessorCount) *
                kTargetBlocksPerMultiprocessor);

        gpuPool.pos = pool.pos;
        gpuPool.vel = pool.vel;
        gpuPool.color = pool.color;
        gpuPool.age = pool.age;
        gpuPool.lifetime = pool.lifetime;
        gpuPool.systemId = pool.systemId;
        gpuPool.activeIndices = reinterpret_cast<uint32_t* const*>(
            reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeIndices));
        gpuPool.aliveCount = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeCount));
        gpuPool.capacity = capacity;

        stats.capacity = capacity;
        initializeFreeList();
    }

    ~Impl()
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }

        destroyTelemetrySlots();
        cudaEventDestroy(latestAfterSpawn);
        cudaEventDestroy(latestAfterCompact);
        cudaEventDestroy(latestAfterSimulate);
        cudaEventDestroy(latestFrameStart);

        cudaFree(deviceTelemetryScratch);
        cudaFree(deviceState);
        cudaFree(spawnCommands);
        cudaFree(aliveFlags);
        cudaFree(freeIndices);
        cudaFree(scratchActiveIndices);
        cudaFree(activeIndices);
        releasePool(pool);

        cudaStreamDestroy(stream);
    }

    void createTelemetrySlots()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            VP_CUDA_CHECK(cudaHostAlloc(
                reinterpret_cast<void**>(&slot.hostTelemetry),
                sizeof(DeviceTelemetry),
                cudaHostAllocPortable));
            VP_CUDA_CHECK(cudaEventCreate(&slot.frameStart));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterSimulate));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterCompact));
            VP_CUDA_CHECK(cudaEventCreate(&slot.afterSpawn));
            VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        }
    }

    void destroyTelemetrySlots()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            cudaEventDestroy(slot.ready);
            cudaEventDestroy(slot.afterSpawn);
            cudaEventDestroy(slot.afterCompact);
            cudaEventDestroy(slot.afterSimulate);
            cudaEventDestroy(slot.frameStart);
            cudaFreeHost(slot.hostTelemetry);
            slot = {};
        }
    }

    void initializeFreeList()
    {
        DeviceFrameState initialState = {};
        initialState.activeIndices = activeIndices;
        initialState.scratchActiveIndices = scratchActiveIndices;
        initialState.activeCount = 0;
        initialState.freeCount = pool.capacity;
        initialState.activeBufferIndex = 0;

        VP_CUDA_CHECK(cudaMemcpyAsync(
            deviceState,
            &initialState,
            sizeof(initialState),
            cudaMemcpyHostToDevice,
            stream));
        initializeFreeListKernel<<<blockCount(pool.capacity), kThreadsPerBlock, 0, stream>>>(
            freeIndices,
            pool.capacity);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaStreamSynchronize(stream));

        pool.aliveCount = 0;
        pool.activeIndices = activeIndices;
        activeWorkUpperBound = 0;
        submittedSpawnUpperBoundTotal = 0;
        nextTelemetrySlot = 0;
        hasSubmittedFrame = false;
        for (TelemetrySlot& slot : telemetrySlots) {
            slot.inFlight = false;
        }
    }

    uint32_t dispatchBlockCount(uint32_t expectedWorkItems) const
    {
        const uint32_t requestedBlocks = std::max(1u, blockCount(expectedWorkItems));
        return std::min(requestedBlocks, maxDispatchBlocks);
    }

    void ensureSpawnCommandStorage(uint32_t commandCount)
    {
        if (commandCount <= spawnCommandCapacity) {
            return;
        }

        VP_CUDA_CHECK(cudaFree(spawnCommands));
        spawnCommands = nullptr;
        cudaAlloc(spawnCommands, commandCount);
        spawnCommandCapacity = commandCount;
    }

    TelemetrySlot* acquireTelemetrySlot()
    {
        for (size_t offset = 0; offset < telemetrySlots.size(); ++offset) {
            const size_t index = (nextTelemetrySlot + offset) % telemetrySlots.size();
            TelemetrySlot& slot = telemetrySlots[index];
            if (!slot.inFlight) {
                nextTelemetrySlot = (index + 1) % telemetrySlots.size();
                return &slot;
            }
        }

        return nullptr;
    }

    void publishTelemetry(
        const DeviceTelemetry& telemetry,
        uint32_t latencyFrames,
        float spawnMs,
        float simulateMs,
        float compactMs,
        float totalMs)
    {
        SimulationStats nextStats = {};
        nextStats.frameIndex = telemetry.frameIndex;
        nextStats.telemetryLatencyFrames = latencyFrames;
        nextStats.valid = true;
        nextStats.capacity = pool.capacity;
        nextStats.aliveCount = telemetry.aliveCount;
        nextStats.requestedSpawn = telemetry.requestedSpawn;
        nextStats.spawned = telemetry.spawned;
        nextStats.dropped = telemetry.dropped;
        nextStats.deadCount = telemetry.deadCount;
        nextStats.spawnMs = spawnMs;
        nextStats.simulateMs = simulateMs;
        nextStats.compactMs = compactMs;
        nextStats.totalMs = totalMs;

        stats = nextStats;
        pool.aliveCount = telemetry.aliveCount;
        pool.activeIndices = telemetry.activeBufferIndex == 0
            ? activeIndices
            : scratchActiveIndices;
    }

    void tightenActiveWorkBound(const TelemetrySlot& slot, const DeviceTelemetry& telemetry)
    {
        if (submittedSpawnUpperBoundTotal < slot.spawnUpperBoundThroughFrame) {
            return;
        }

        const uint64_t potentialSpawnsSinceSample =
            submittedSpawnUpperBoundTotal - slot.spawnUpperBoundThroughFrame;
        const uint64_t refreshedBound = std::min<uint64_t>(
            pool.capacity,
            static_cast<uint64_t>(telemetry.aliveCount) + potentialSpawnsSinceSample);
        activeWorkUpperBound = std::min(
            activeWorkUpperBound,
            static_cast<uint32_t>(refreshedBound));
    }

    void pollTelemetry()
    {
        for (TelemetrySlot& slot : telemetrySlots) {
            if (!slot.inFlight) {
                continue;
            }

            const cudaError_t queryResult = cudaEventQuery(slot.ready);
            if (queryResult == cudaErrorNotReady) {
                continue;
            }
            VP_CUDA_CHECK(queryResult);

            const DeviceTelemetry telemetry = *slot.hostTelemetry;
            float spawnMs = 0.0f;
            float simulateMs = 0.0f;
            float compactMs = 0.0f;
            float totalMs = 0.0f;
            VP_CUDA_CHECK(cudaEventElapsedTime(&spawnMs, slot.afterCompact, slot.afterSpawn));
            VP_CUDA_CHECK(cudaEventElapsedTime(&simulateMs, slot.frameStart, slot.afterSimulate));
            VP_CUDA_CHECK(cudaEventElapsedTime(&compactMs, slot.afterSimulate, slot.afterCompact));
            VP_CUDA_CHECK(cudaEventElapsedTime(&totalMs, slot.frameStart, slot.afterSpawn));

            const uint64_t submittedFrames = frameIndex;
            const uint32_t latencyFrames = submittedFrames > slot.frameIndex
                ? static_cast<uint32_t>(std::min<uint64_t>(
                      submittedFrames - slot.frameIndex - 1u,
                      std::numeric_limits<uint32_t>::max()))
                : 0u;
            publishTelemetry(telemetry, latencyFrames, spawnMs, simulateMs, compactMs, totalMs);
            tightenActiveWorkBound(slot, telemetry);
            slot.inFlight = false;
        }
    }

    void queueTelemetry(TelemetrySlot& slot)
    {
        snapshotTelemetryKernel<<<1, 1, 0, stream>>>(deviceState, deviceTelemetryScratch);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaMemcpyAsync(
            slot.hostTelemetry,
            deviceTelemetryScratch,
            sizeof(DeviceTelemetry),
            cudaMemcpyDeviceToHost,
            stream));
        VP_CUDA_CHECK(cudaEventRecord(slot.ready, stream));
        slot.frameIndex = frameIndex;
        slot.spawnUpperBoundThroughFrame = submittedSpawnUpperBoundTotal;
        slot.inFlight = true;
    }

    void advanceActiveWorkUpperBound(uint32_t potentialSpawnCount)
    {
        activeWorkUpperBound = static_cast<uint32_t>(std::min<uint64_t>(
            pool.capacity,
            static_cast<uint64_t>(activeWorkUpperBound) + potentialSpawnCount));

        const uint64_t maxValue = std::numeric_limits<uint64_t>::max();
        submittedSpawnUpperBoundTotal = maxValue - submittedSpawnUpperBoundTotal < potentialSpawnCount
            ? maxValue
            : submittedSpawnUpperBoundTotal + potentialSpawnCount;
    }

    ParticlePool pool = {};
    GpuParticlePool gpuPool = {};
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t* freeIndices = nullptr;
    uint8_t* aliveFlags = nullptr;
    SpawnCommand* spawnCommands = nullptr;
    uint32_t spawnCommandCapacity = 0;
    DeviceFrameState* deviceState = nullptr;
    DeviceTelemetry* deviceTelemetryScratch = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t latestFrameStart = nullptr;
    cudaEvent_t latestAfterSimulate = nullptr;
    cudaEvent_t latestAfterCompact = nullptr;
    cudaEvent_t latestAfterSpawn = nullptr;
    std::array<TelemetrySlot, kTelemetryRingSize> telemetrySlots = {};
    size_t nextTelemetrySlot = 0;
    uint32_t maxDispatchBlocks = 1;
    uint32_t activeWorkUpperBound = 0;
    uint64_t submittedSpawnUpperBoundTotal = 0;
    SimulationSettings settings = {};
    SimulationStats stats = {};
    std::vector<EmitterState> emitters;
    std::vector<SpawnCommand> hostSpawnCommands;
    uint32_t frameIndex = 0;
    bool hasSubmittedFrame = false;
};

ParticleSystem::ParticleSystem(uint32_t capacity)
    : impl_(new Impl(capacity))
{
}

ParticleSystem::~ParticleSystem()
{
    delete impl_;
}

uint32_t ParticleSystem::addEmitter(const EmitterDesc& desc)
{
    impl_->emitters.push_back(EmitterState{desc});
    try {
        impl_->ensureSpawnCommandStorage(static_cast<uint32_t>(impl_->emitters.size()));
    } catch (...) {
        impl_->emitters.pop_back();
        throw;
    }
    return static_cast<uint32_t>(impl_->emitters.size() - 1);
}

void ParticleSystem::queueBurst(uint32_t systemId, uint32_t count)
{
    if (systemId >= impl_->emitters.size()) {
        return;
    }

    EmitterState& emitter = impl_->emitters[systemId];
    emitter.pendingBurst = static_cast<uint32_t>(std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(),
        static_cast<uint64_t>(emitter.pendingBurst) + count));
}

void ParticleSystem::update(float dt)
{
    impl_->pollTelemetry();
    dt = std::max(0.0f, dt);

    uint64_t requestedSpawnTotal = 0;
    uint32_t commandCoveredSpawn = 0;
    impl_->hostSpawnCommands.clear();
    impl_->hostSpawnCommands.reserve(impl_->emitters.size());

    for (uint32_t systemId = 0; systemId < impl_->emitters.size(); ++systemId) {
        EmitterState& emitter = impl_->emitters[systemId];
        const double exactSpawn = static_cast<double>(emitter.spawnCarry) +
            static_cast<double>(std::max(0.0f, emitter.desc.spawnRate)) * dt;
        const double wholeSpawn = std::floor(exactSpawn);
        const uint32_t continuousSpawn = static_cast<uint32_t>(std::min<double>(
            wholeSpawn,
            static_cast<double>(std::numeric_limits<uint32_t>::max())));
        emitter.spawnCarry = static_cast<float>(exactSpawn - wholeSpawn);

        const uint32_t emitterRequest = static_cast<uint32_t>(std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(),
            static_cast<uint64_t>(continuousSpawn) + emitter.pendingBurst));
        emitter.pendingBurst = 0;
        requestedSpawnTotal = std::min<uint64_t>(
            std::numeric_limits<uint32_t>::max(),
            requestedSpawnTotal + emitterRequest);

        const uint32_t commandCount = std::min(
            emitterRequest,
            impl_->pool.capacity - commandCoveredSpawn);
        if (commandCount == 0) {
            continue;
        }

        impl_->hostSpawnCommands.push_back(SpawnCommand{
            emitter.desc,
            systemId,
            commandCoveredSpawn,
            commandCount});
        commandCoveredSpawn += commandCount;
    }

    const uint32_t requestedSpawn = static_cast<uint32_t>(requestedSpawnTotal);
    const uint32_t simulationBlocks = impl_->dispatchBlockCount(impl_->activeWorkUpperBound);
    const uint32_t spawnBlocks = impl_->dispatchBlockCount(commandCoveredSpawn);
    Impl::TelemetrySlot* telemetrySlot = impl_->acquireTelemetrySlot();

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestFrameStart, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->frameStart, impl_->stream));
    }

    beginFrameKernel<<<1, 1, 0, impl_->stream>>>(
        impl_->deviceState,
        requestedSpawn,
        impl_->frameIndex);
    VP_CUDA_CHECK(cudaGetLastError());

    simulateKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool,
        impl_->deviceState,
        impl_->aliveFlags,
        impl_->freeIndices,
        impl_->settings,
        dt);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSimulate, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSimulate, impl_->stream));
    }

    compactActiveIndicesKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
        impl_->deviceState,
        impl_->aliveFlags);
    VP_CUDA_CHECK(cudaGetLastError());
    finalizeCompactionKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterCompact, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterCompact, impl_->stream));
    }

    if (!impl_->hostSpawnCommands.empty()) {
        const uint32_t commandCount = static_cast<uint32_t>(impl_->hostSpawnCommands.size());
        VP_CUDA_CHECK(cudaMemcpyAsync(
            impl_->spawnCommands,
            impl_->hostSpawnCommands.data(),
            sizeof(SpawnCommand) * commandCount,
            cudaMemcpyHostToDevice,
            impl_->stream));
    }

    prepareSpawnKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
    VP_CUDA_CHECK(cudaGetLastError());
    if (!impl_->hostSpawnCommands.empty()) {
        spawnBatchKernel<<<spawnBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
            impl_->pool,
            impl_->deviceState,
            impl_->freeIndices,
            impl_->spawnCommands,
            static_cast<uint32_t>(impl_->hostSpawnCommands.size()),
            impl_->settings.seed);
        VP_CUDA_CHECK(cudaGetLastError());
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSpawn, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSpawn, impl_->stream));
    }

    impl_->advanceActiveWorkUpperBound(commandCoveredSpawn);
    if (telemetrySlot != nullptr) {
        impl_->queueTelemetry(*telemetrySlot);
    }

    ++impl_->frameIndex;
    impl_->hasSubmittedFrame = true;
}

void ParticleSystem::reset()
{
    impl_->initializeFreeList();
    impl_->stats = {};
    impl_->stats.capacity = impl_->pool.capacity;
    impl_->frameIndex = 0;
    for (EmitterState& emitter : impl_->emitters) {
        emitter.spawnCarry = 0.0f;
        emitter.pendingBurst = 0;
    }
}

void ParticleSystem::synchronize()
{
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    impl_->pollTelemetry();
    if (!impl_->hasSubmittedFrame) {
        return;
    }

    DeviceTelemetry telemetry = {};
    snapshotTelemetryKernel<<<1, 1, 0, impl_->stream>>>(
        impl_->deviceState,
        impl_->deviceTelemetryScratch);
    VP_CUDA_CHECK(cudaGetLastError());
    VP_CUDA_CHECK(cudaMemcpyAsync(
        &telemetry,
        impl_->deviceTelemetryScratch,
        sizeof(telemetry),
        cudaMemcpyDeviceToHost,
        impl_->stream));
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

    float spawnMs = 0.0f;
    float simulateMs = 0.0f;
    float compactMs = 0.0f;
    float totalMs = 0.0f;
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &spawnMs,
        impl_->latestAfterCompact,
        impl_->latestAfterSpawn));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &simulateMs,
        impl_->latestFrameStart,
        impl_->latestAfterSimulate));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &compactMs,
        impl_->latestAfterSimulate,
        impl_->latestAfterCompact));
    VP_CUDA_CHECK(cudaEventElapsedTime(
        &totalMs,
        impl_->latestFrameStart,
        impl_->latestAfterSpawn));
    impl_->publishTelemetry(telemetry, 0, spawnMs, simulateMs, compactMs, totalMs);
    impl_->activeWorkUpperBound = telemetry.aliveCount;
}

const ParticlePool& ParticleSystem::buffers() const
{
    return impl_->pool;
}

const GpuParticlePool& ParticleSystem::gpuBuffers() const
{
    return impl_->gpuPool;
}

const SimulationStats& ParticleSystem::stats() const
{
    return impl_->stats;
}

} // namespace vparticles
