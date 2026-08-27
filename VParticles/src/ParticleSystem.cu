#include "VParticles/ParticleSystem.h"

#include "VParticles/CudaCheck.h"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vparticles {
namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kIndexSortOccupancyPercent = 99;

template <typename T>
void cudaAlloc(T*& pointer, size_t count)
{
    VP_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(T) * count));
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

__global__ void initializeFreeListKernel(uint32_t* freeIndices, uint32_t* freeCount, uint32_t capacity)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < capacity) {
        freeIndices[index] = index;
    }

    if (index == 0) {
        *freeCount = capacity;
    }
}

struct SpawnCommand {
    EmitterDesc emitter = {};
    uint32_t systemId = 0;
    uint32_t activeStart = 0;
    uint32_t count = 0;
};

__device__ uint32_t findSpawnCommand(
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t activeIndex)
{
    uint32_t first = 0;
    uint32_t last = commandCount;
    while (first + 1 < last) {
        const uint32_t middle = first + (last - first) / 2;
        if (commands[middle].activeStart <= activeIndex) {
            first = middle;
        } else {
            last = middle;
        }
    }

    return first;
}

__global__ void spawnBatchKernel(
    ParticlePool pool,
    uint32_t* activeIndices,
    uint32_t* freeIndices,
    uint32_t* freeCount,
    const SpawnCommand* commands,
    uint32_t commandCount,
    uint32_t firstActiveIndex,
    uint32_t totalSpawn,
    uint32_t frameIndex,
    uint32_t seed)
{
    const uint32_t spawnOffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (spawnOffset >= totalSpawn) {
        return;
    }

    const uint32_t activeIndex = firstActiveIndex + spawnOffset;
    const uint32_t commandIndex = findSpawnCommand(commands, commandCount, activeIndex);
    const SpawnCommand command = commands[commandIndex];
    const EmitterDesc emitter = command.emitter;
    const uint32_t freeSlot = atomicSub(freeCount, 1u) - 1u;
    const uint32_t particleIndex = freeIndices[freeSlot];
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

__global__ void simulateKernel(
    ParticlePool pool,
    const uint32_t* activeIndices,
    uint8_t* aliveFlags,
    uint32_t* freeIndices,
    uint32_t* freeCount,
    uint32_t* deadCount,
    SimulationSettings settings,
    float dt,
    uint32_t count)
{
    const uint32_t activeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (activeIndex >= count) {
        return;
    }

    const uint32_t particleIndex = activeIndices[activeIndex];
    float age = pool.age[particleIndex] + dt;
    const float lifetime = pool.lifetime[particleIndex];
    if (age >= lifetime) {
        pool.age[particleIndex] = age;
        aliveFlags[activeIndex] = 0;
        const uint32_t freeSlot = atomicAdd(freeCount, 1u);
        freeIndices[freeSlot] = particleIndex;
        atomicAdd(deadCount, 1u);
        return;
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
    aliveFlags[activeIndex] = 1;
}

uint32_t blockCount(uint32_t itemCount)
{
    return (itemCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

struct EmitterState {
    EmitterDesc desc = {};
    float spawnCarry = 0.0f;
    uint32_t pendingBurst = 0;
};

} // namespace

struct ParticleSystem::Impl {
    explicit Impl(uint32_t capacity)
    {
        if (capacity == 0) {
            throw std::invalid_argument("ParticleSystem capacity must be greater than zero");
        }

        allocatePool(pool, capacity);
        cudaAlloc(activeIndices, capacity);
        cudaAlloc(scratchActiveIndices, capacity);
        cudaAlloc(freeIndices, capacity);
        cudaAlloc(freeCount, 1);
        cudaAlloc(aliveFlags, capacity);
        cudaAlloc(selectedCount, 1);
        cudaAlloc(deadCount, 1);
        pool.activeIndices = activeIndices;

        VP_CUDA_CHECK(cudaStreamCreate(&stream));
        VP_CUDA_CHECK(cudaEventCreate(&frameStart));
        VP_CUDA_CHECK(cudaEventCreate(&afterSpawn));
        VP_CUDA_CHECK(cudaEventCreate(&afterSimulate));
        VP_CUDA_CHECK(cudaEventCreate(&afterCompact));

        stats.capacity = capacity;
        initializeFreeList();
    }

    ~Impl()
    {
        cudaFree(cubTempStorage);
        cudaFree(deadCount);
        cudaFree(selectedCount);
        cudaFree(aliveFlags);
        cudaFree(freeCount);
        cudaFree(freeIndices);
        cudaFree(spawnCommands);
        cudaFree(scratchActiveIndices);
        cudaFree(activeIndices);
        releasePool(pool);

        cudaEventDestroy(afterCompact);
        cudaEventDestroy(afterSimulate);
        cudaEventDestroy(afterSpawn);
        cudaEventDestroy(frameStart);
        cudaStreamDestroy(stream);
    }

    void initializeFreeList()
    {
        initializeFreeListKernel<<<blockCount(pool.capacity), kThreadsPerBlock, 0, stream>>>(
            freeIndices,
            freeCount,
            pool.capacity);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaStreamSynchronize(stream));
        pool.aliveCount = 0;
        pool.activeIndices = activeIndices;
        activeIndexOrderDirty = false;
        freeListMayBeFragmented = false;
    }

    void selectActiveIndices(uint32_t count)
    {
        size_t requestedBytes = 0;
        VP_CUDA_CHECK(cub::DeviceSelect::Flagged(
            nullptr,
            requestedBytes,
            activeIndices,
            aliveFlags,
            scratchActiveIndices,
            selectedCount,
            static_cast<int>(count),
            stream));

        ensureCubStorage(requestedBytes);

        VP_CUDA_CHECK(cub::DeviceSelect::Flagged(
            cubTempStorage,
            cubTempStorageBytes,
            activeIndices,
            aliveFlags,
            scratchActiveIndices,
            selectedCount,
            static_cast<int>(count),
            stream));
    }

    void sortActiveIndices(uint32_t count)
    {
        if (count < 2) {
            return;
        }

        size_t requestedBytes = 0;
        VP_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr,
            requestedBytes,
            activeIndices,
            scratchActiveIndices,
            static_cast<int>(count),
            0,
            sizeof(uint32_t) * 8,
            stream));

        ensureCubStorage(requestedBytes);

        VP_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            cubTempStorage,
            cubTempStorageBytes,
            activeIndices,
            scratchActiveIndices,
            static_cast<int>(count),
            0,
            sizeof(uint32_t) * 8,
            stream));

        std::swap(activeIndices, scratchActiveIndices);
        pool.activeIndices = activeIndices;
    }

    uint32_t compact(uint32_t count)
    {
        if (count == 0) {
            pool.aliveCount = 0;
            return 0;
        }

        uint32_t hostDeadCount = 0;
        VP_CUDA_CHECK(cudaMemcpyAsync(
            &hostDeadCount,
            deadCount,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream));
        VP_CUDA_CHECK(cudaStreamSynchronize(stream));

        if (hostDeadCount == 0) {
            pool.aliveCount = count;
            pool.activeIndices = activeIndices;
            return 0;
        }

        selectActiveIndices(count);

        int hostSelectedCount = 0;
        VP_CUDA_CHECK(cudaMemcpyAsync(
            &hostSelectedCount,
            selectedCount,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        VP_CUDA_CHECK(cudaStreamSynchronize(stream));

        std::swap(activeIndices, scratchActiveIndices);
        pool.activeIndices = activeIndices;
        pool.aliveCount = static_cast<uint32_t>(std::max(0, hostSelectedCount));
        return hostDeadCount;
    }

    void ensureCubStorage(size_t requestedBytes)
    {
        if (requestedBytes <= cubTempStorageBytes) {
            return;
        }

        cudaFree(cubTempStorage);
        cubTempStorage = nullptr;
        cubTempStorageBytes = requestedBytes;
        VP_CUDA_CHECK(cudaMalloc(&cubTempStorage, cubTempStorageBytes));
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

    ParticlePool pool = {};
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t* freeIndices = nullptr;
    uint8_t* aliveFlags = nullptr;
    int* selectedCount = nullptr;
    uint32_t* deadCount = nullptr;
    uint32_t* freeCount = nullptr;
    SpawnCommand* spawnCommands = nullptr;
    uint32_t spawnCommandCapacity = 0;
    bool activeIndexOrderDirty = false;
    bool freeListMayBeFragmented = false;
    void* cubTempStorage = nullptr;
    size_t cubTempStorageBytes = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t frameStart = nullptr;
    cudaEvent_t afterSpawn = nullptr;
    cudaEvent_t afterSimulate = nullptr;
    cudaEvent_t afterCompact = nullptr;
    SimulationSettings settings = {};
    SimulationStats stats = {};
    std::vector<EmitterState> emitters;
    std::vector<SpawnCommand> hostSpawnCommands;
    uint32_t frameIndex = 0;
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
    return static_cast<uint32_t>(impl_->emitters.size() - 1);
}

void ParticleSystem::queueBurst(uint32_t systemId, uint32_t count)
{
    if (systemId >= impl_->emitters.size()) {
        return;
    }

    impl_->emitters[systemId].pendingBurst += count;
}

void ParticleSystem::update(float dt)
{
    dt = std::max(0.0f, dt);

    SimulationStats nextStats = {};
    nextStats.capacity = impl_->pool.capacity;

    VP_CUDA_CHECK(cudaEventRecord(impl_->frameStart, impl_->stream));

    const uint32_t simulationCount = impl_->pool.aliveCount;
    if (simulationCount > 0) {
        VP_CUDA_CHECK(cudaMemsetAsync(impl_->deadCount, 0, sizeof(uint32_t), impl_->stream));
        simulateKernel<<<blockCount(simulationCount), kThreadsPerBlock, 0, impl_->stream>>>(
            impl_->pool,
            impl_->activeIndices,
            impl_->aliveFlags,
            impl_->freeIndices,
            impl_->freeCount,
            impl_->deadCount,
            impl_->settings,
            dt,
            simulationCount);
        VP_CUDA_CHECK(cudaGetLastError());
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->afterSimulate, impl_->stream));
    const uint32_t deadCount = impl_->compact(simulationCount);
    impl_->freeListMayBeFragmented = impl_->freeListMayBeFragmented || deadCount > 0;
    const bool shouldRestoreLocality =
        impl_->activeIndexOrderDirty &&
        impl_->pool.aliveCount >=
            (static_cast<uint64_t>(impl_->pool.capacity) * kIndexSortOccupancyPercent) / 100;
    if (shouldRestoreLocality) {
        impl_->sortActiveIndices(impl_->pool.aliveCount);
        impl_->activeIndexOrderDirty = false;
    }
    VP_CUDA_CHECK(cudaEventRecord(impl_->afterCompact, impl_->stream));

    uint32_t requestedSpawn = 0;
    uint32_t spawned = 0;
    uint32_t writeStart = impl_->pool.aliveCount;
    uint32_t openSlots = impl_->pool.capacity - impl_->pool.aliveCount;
    impl_->hostSpawnCommands.clear();
    impl_->hostSpawnCommands.reserve(impl_->emitters.size());

    for (uint32_t systemId = 0; systemId < impl_->emitters.size(); ++systemId) {
        EmitterState& emitter = impl_->emitters[systemId];
        const float exactSpawn = emitter.spawnCarry + std::max(0.0f, emitter.desc.spawnRate) * dt;
        const uint32_t continuousSpawn = static_cast<uint32_t>(std::floor(exactSpawn));
        emitter.spawnCarry = exactSpawn - static_cast<float>(continuousSpawn);

        const uint32_t emitterRequest = continuousSpawn + emitter.pendingBurst;
        emitter.pendingBurst = 0;
        requestedSpawn += emitterRequest;

        const uint32_t emitterSpawn = std::min(emitterRequest, openSlots);
        if (emitterSpawn == 0) {
            continue;
        }

        impl_->hostSpawnCommands.push_back(SpawnCommand{
            emitter.desc,
            systemId,
            writeStart,
            emitterSpawn});

        writeStart += emitterSpawn;
        openSlots -= emitterSpawn;
        spawned += emitterSpawn;
    }

    if (!impl_->hostSpawnCommands.empty()) {
        const uint32_t commandCount = static_cast<uint32_t>(impl_->hostSpawnCommands.size());
        impl_->ensureSpawnCommandStorage(commandCount);
        VP_CUDA_CHECK(cudaMemcpyAsync(
            impl_->spawnCommands,
            impl_->hostSpawnCommands.data(),
            sizeof(SpawnCommand) * commandCount,
            cudaMemcpyHostToDevice,
            impl_->stream));
        spawnBatchKernel<<<blockCount(spawned), kThreadsPerBlock, 0, impl_->stream>>>(
            impl_->pool,
            impl_->activeIndices,
            impl_->freeIndices,
            impl_->freeCount,
            impl_->spawnCommands,
            commandCount,
            impl_->pool.aliveCount,
            spawned,
            impl_->frameIndex,
            impl_->settings.seed);
        VP_CUDA_CHECK(cudaGetLastError());
    }

    if (spawned > 0 && impl_->freeListMayBeFragmented) {
        impl_->activeIndexOrderDirty = true;
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->afterSpawn, impl_->stream));
    VP_CUDA_CHECK(cudaEventSynchronize(impl_->afterSpawn));

    nextStats.requestedSpawn = requestedSpawn;
    nextStats.spawned = spawned;
    nextStats.dropped = requestedSpawn - spawned;
    nextStats.deadCount = deadCount;
    nextStats.aliveCount = impl_->pool.aliveCount + spawned;
    impl_->pool.aliveCount = nextStats.aliveCount;
    VP_CUDA_CHECK(cudaEventElapsedTime(&nextStats.spawnMs, impl_->afterCompact, impl_->afterSpawn));
    VP_CUDA_CHECK(cudaEventElapsedTime(&nextStats.simulateMs, impl_->frameStart, impl_->afterSimulate));
    VP_CUDA_CHECK(cudaEventElapsedTime(&nextStats.compactMs, impl_->afterSimulate, impl_->afterCompact));
    VP_CUDA_CHECK(cudaEventElapsedTime(&nextStats.totalMs, impl_->frameStart, impl_->afterSpawn));

    impl_->stats = nextStats;
    ++impl_->frameIndex;
}

void ParticleSystem::reset()
{
    impl_->initializeFreeList();
    impl_->stats = {};
    impl_->stats.capacity = impl_->pool.capacity;
    for (EmitterState& emitter : impl_->emitters) {
        emitter.spawnCarry = 0.0f;
        emitter.pendingBurst = 0;
    }
}

const ParticlePool& ParticleSystem::buffers() const
{
    return impl_->pool;
}

const SimulationStats& ParticleSystem::stats() const
{
    return impl_->stats;
}

} // namespace vparticles
