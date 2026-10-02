#pragma once

#include "InternalTypes.h"
#include "ParticleMath.cuh"
#include "SimulationKernels.cuh"
#include "CompactionKernels.cuh"
#include "SpawnKernels.cuh"
#include "TileKernels.cuh"

#include "VParticles/ParticleSystem.h"

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
        bool graphActive = false;
    };

    struct SpawnUploadSlot {
        SpawnCommand* hostCommands = nullptr;
        uint32_t capacity = 0;
        cudaEvent_t ready = nullptr;
        bool inFlight = false;
    };

    struct GraphKey {
        uint32_t simulationBlocks = 0;
        uint32_t spawnBlocks = 0;
        uint32_t migrateBlocks = 0;
        uint32_t commandCount = 0;
        uint32_t recipeCount = 0;
        uint32_t tileCount = 0;
        bool valid = false;

        bool matches(const GraphKey& other) const
        {
            return valid && other.valid &&
                simulationBlocks == other.simulationBlocks &&
                spawnBlocks == other.spawnBlocks &&
                migrateBlocks == other.migrateBlocks &&
                commandCount == other.commandCount &&
                recipeCount == other.recipeCount &&
                tileCount == other.tileCount;
        }
    };

    explicit Impl(uint32_t capacity, StorageMode mode, bool enableCudaGraphs)
    {
        if (capacity == 0) {
            throw std::invalid_argument("ParticleSystem capacity must be greater than zero");
        }

        mode_ = mode;
        if (mode == StorageMode::Packed) {
            allocatePackedPool(packedPool_, capacity);
            // FP32 pool still needed for capacity tracking and telemetry.
            pool.capacity = capacity;
            pool.aliveCount = 0;
        } else {
            allocatePool(pool, capacity);
        }
        cudaAlloc(activeIndices, capacity);
        cudaAlloc(scratchActiveIndices, capacity);
        cudaAlloc(freeIndices, capacity);
        cudaAlloc(aliveFlags, capacity);
        cudaAlloc(deviceState, 1);
        cudaAlloc(deviceTelemetryScratch, 1);
        cudaAlloc(deviceGraphParams, 1);
        pool.activeIndices = activeIndices;

        VP_CUDA_CHECK(cudaStreamCreate(&stream));
        createTelemetrySlots();
        createSpawnUploadSlots();
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

        if (mode != StorageMode::Packed) {
            gpuPool.pos = pool.pos;
            gpuPool.vel = pool.vel;
            gpuPool.color = pool.color;
            gpuPool.age = pool.age;
            gpuPool.lifetime = pool.lifetime;
            gpuPool.systemId = pool.systemId;
            gpuPool.tileId = pool.tileId;
            gpuPool.activeIndices = reinterpret_cast<uint32_t* const*>(
                reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeIndices));
            gpuPool.aliveCount = reinterpret_cast<const uint32_t*>(
                reinterpret_cast<char*>(deviceState) + offsetof(DeviceFrameState, activeCount));
        }
        gpuPool.capacity = capacity;

        settings.tileCount = 1;
        settings.recipeCount = 1;
        settings.recipes[0].gravity = settings.gravity;
        settings.recipes[0].wind = settings.wind;
        settings.recipes[0].drag = settings.drag;
        settings.recipes[0].turbulence = settings.turbulence;
        settings.recipes[0].plane = settings.plane;
        settings.recipes[0].sphere = settings.sphere;
        settings.recipes[0].box = settings.box;
        settings.recipes[0].curves = settings.curves;
        VP_CUDA_CHECK(cudaMemcpyToSymbol(cTiles, settings.tiles, sizeof(TileDesc) * kMaxTiles));
        VP_CUDA_CHECK(cudaMemcpyToSymbol(cRecipes, settings.recipes, sizeof(EffectRecipe) * kMaxRecipes));

        stats.capacity = capacity;
        graphsEnabled = enableCudaGraphs;
        initializeFreeList();
    }

    ~Impl()
    {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }

        destroyTelemetrySlots();
        destroySpawnUploadSlots();
        if (graphExec != nullptr) {
            cudaGraphExecDestroy(graphExec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        cudaEventDestroy(latestAfterSpawn);
        cudaEventDestroy(latestAfterCompact);
        cudaEventDestroy(latestAfterSimulate);
        cudaEventDestroy(latestFrameStart);

        cudaFree(deviceTelemetryScratch);
        cudaFree(deviceGraphParams);
        cudaFree(deviceState);
        cudaFree(spawnCommands);
        cudaFree(aliveFlags);
        cudaFree(freeIndices);
        cudaFree(scratchActiveIndices);
        cudaFree(activeIndices);
        releasePool(pool);
        releasePackedPool(packedPool_);

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

    void createSpawnUploadSlots()
    {
        spawnUploadSlots.resize(kSpawnUploadRingSize);
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        }
    }

    void destroySpawnUploadSlots()
    {
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (slot.ready != nullptr) {
                cudaEventDestroy(slot.ready);
            }
            if (slot.hostCommands != nullptr) {
                cudaFreeHost(slot.hostCommands);
            }
            slot = {};
        }
        spawnUploadSlots.clear();
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
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            slot.inFlight = false;
        }
    }

    void invalidateGraph()
    {
        if (graphExec != nullptr) {
            VP_CUDA_CHECK(cudaGraphExecDestroy(graphExec));
            graphExec = nullptr;
        }
        if (graph != nullptr) {
            VP_CUDA_CHECK(cudaGraphDestroy(graph));
            graph = nullptr;
        }
        graphSpawnMemcpyNode = nullptr;
        graphSpawnMemcpyParams = {};
        currentGraphKey = {};
    }

    void rebuildGraph(const GraphKey& key, const SpawnUploadSlot* uploadSlot)
    {
        invalidateGraph();

        VP_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        beginFrameFromParamsKernel<<<1, 256, 0, stream>>>(
            deviceState,
            deviceTelemetryScratch,
            deviceGraphParams);
        VP_CUDA_CHECK(cudaGetLastError());

        if (mode_ == StorageMode::Packed) {
            const float posScale = settings.posQuantizationScale;
            const float invPosScale = 1.0f / posScale;
            simulatePackedKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
                packedPool_, deviceState, aliveFlags, freeIndices,
                deviceGraphParams, posScale, invPosScale,
                key.recipeCount, key.tileCount);
        } else {
            simulateKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
                pool, deviceState, aliveFlags, freeIndices,
                deviceGraphParams, key.recipeCount, key.tileCount);
        }
        VP_CUDA_CHECK(cudaGetLastError());

        compactActiveIndicesKernel<<<key.simulationBlocks, kThreadsPerBlock, 0, stream>>>(
            deviceState, aliveFlags);
        VP_CUDA_CHECK(cudaGetLastError());
        finalizeCompactionKernel<<<1, 1, 0, stream>>>(deviceState);
        VP_CUDA_CHECK(cudaGetLastError());

        if (key.commandCount != 0) {
            VP_CUDA_CHECK(cudaMemcpyAsync(
                spawnCommands,
                uploadSlot->hostCommands,
                sizeof(SpawnCommand) * key.commandCount,
                cudaMemcpyHostToDevice,
                stream));
        }

        prepareSpawnKernel<<<1, 1, 0, stream>>>(deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        if (key.commandCount != 0) {
            if (mode_ == StorageMode::Packed) {
                const float posScale = settings.posQuantizationScale;
                spawnBatchPackedKernel<<<key.spawnBlocks, kThreadsPerBlock, 0, stream>>>(
                    packedPool_, deviceState, freeIndices, spawnCommands,
                    key.commandCount, settings.seed, posScale);
            } else {
                spawnBatchKernel<<<key.spawnBlocks, kThreadsPerBlock, 0, stream>>>(
                    pool, deviceState, freeIndices, spawnCommands,
                    key.commandCount, settings.seed);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }

        if (settings.tileCount > 1) {
            if (mode_ == StorageMode::Packed) {
                const float posScale = settings.posQuantizationScale;
                const float invPosScale = 1.0f / posScale;
                migrateTilesPackedKernel<<<key.migrateBlocks, kThreadsPerBlock, 0, stream>>>(
                    packedPool_, deviceState, deviceTelemetryScratch, settings.tileCount,
                    posScale, invPosScale);
            } else {
                migrateTilesKernel<<<key.migrateBlocks, kThreadsPerBlock, 0, stream>>>(
                    pool, deviceState, deviceTelemetryScratch, settings.tileCount);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }

        VP_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        VP_CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph));

        if (key.commandCount != 0) {
            size_t nodeCount = 0;
            VP_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodeCount));
            std::vector<cudaGraphNode_t> nodes(nodeCount);
            VP_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &nodeCount));
            for (cudaGraphNode_t node : nodes) {
                cudaGraphNodeType nodeType = cudaGraphNodeTypeKernel;
                VP_CUDA_CHECK(cudaGraphNodeGetType(node, &nodeType));
                if (nodeType == cudaGraphNodeTypeMemcpy) {
                    graphSpawnMemcpyNode = node;
                    VP_CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node, &graphSpawnMemcpyParams));
                    break;
                }
            }
            if (graphSpawnMemcpyNode == nullptr) {
                throw std::runtime_error("CUDA Graph capture did not produce the spawn upload node");
            }
        }

        currentGraphKey = key;
        ++graphRebuildCount;
    }

    void setGraphSpawnUploadSource(const SpawnUploadSlot& uploadSlot)
    {
        if (graphSpawnMemcpyNode == nullptr) {
            return;
        }
        cudaMemcpy3DParms params = graphSpawnMemcpyParams;
        params.srcPtr.ptr = uploadSlot.hostCommands;
        VP_CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams(
            graphExec, graphSpawnMemcpyNode, &params));
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

    void pollSpawnUploadSlots()
    {
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (!slot.inFlight) {
                continue;
            }

            const cudaError_t queryResult = cudaEventQuery(slot.ready);
            if (queryResult == cudaErrorNotReady) {
                continue;
            }
            VP_CUDA_CHECK(queryResult);
            slot.inFlight = false;
        }
    }

    void ensureSpawnUploadCapacity(SpawnUploadSlot& slot, uint32_t commandCount)
    {
        if (commandCount <= slot.capacity) {
            return;
        }

        if (slot.hostCommands != nullptr) {
            VP_CUDA_CHECK(cudaFreeHost(slot.hostCommands));
        }
        slot.hostCommands = nullptr;
        VP_CUDA_CHECK(cudaHostAlloc(
            reinterpret_cast<void**>(&slot.hostCommands),
            sizeof(SpawnCommand) * commandCount,
            cudaHostAllocPortable));
        slot.capacity = commandCount;
    }

    SpawnUploadSlot& acquireSpawnUploadSlot(uint32_t commandCount)
    {
        pollSpawnUploadSlots();
        for (SpawnUploadSlot& slot : spawnUploadSlots) {
            if (!slot.inFlight) {
                ensureSpawnUploadCapacity(slot, commandCount);
                return slot;
            }
        }

        SpawnUploadSlot slot = {};
        VP_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
        spawnUploadSlots.push_back(slot);
        ensureSpawnUploadCapacity(spawnUploadSlots.back(), commandCount);
        return spawnUploadSlots.back();
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
        float totalMs,
        bool graphActive)
    {
        SimulationStats nextStats = {};
        nextStats.frameIndex = telemetry.frameIndex;
        nextStats.telemetryLatencyFrames = latencyFrames;
        nextStats.valid = true;
        nextStats.graphActive = graphActive;
        nextStats.graphRebuildCount = graphRebuildCount;
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

        tileStats.resize(settings.tileCount);
        for (uint32_t t = 0; t < settings.tileCount; ++t) {
            tileStats[t].tileId = t;
            tileStats[t].aliveCount = telemetry.tileAliveCounts[t];
            tileStats[t].origin = {
                settings.tiles[t].originX,
                settings.tiles[t].originY,
                settings.tiles[t].originZ
            };
        }
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

            if (slot.graphActive) {
                // A graph launch has no externally-addressable internal timing
                // events. Publish its complete pipeline duration as simulateMs.
                simulateMs = totalMs;
                compactMs = 0.0f;
                spawnMs = 0.0f;
            }

            const uint64_t submittedFrames = frameIndex;
            const uint32_t latencyFrames = submittedFrames > slot.frameIndex
                ? static_cast<uint32_t>(std::min<uint64_t>(
                      submittedFrames - slot.frameIndex - 1u,
                      std::numeric_limits<uint32_t>::max()))
                : 0u;
            publishTelemetry(
                telemetry, latencyFrames, spawnMs, simulateMs, compactMs, totalMs,
                slot.graphActive);
            tightenActiveWorkBound(slot, telemetry);
            slot.inFlight = false;
        }
    }

    void queueTelemetry(TelemetrySlot& slot)
    {
        snapshotTelemetryKernel<<<1, 1, 0, stream>>>(deviceState, deviceTelemetryScratch, settings.tileCount);
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
    PackedPool packedPool_ = {};
    StorageMode mode_ = StorageMode::FP32;
    GpuParticlePool gpuPool = {};
    uint32_t* activeIndices = nullptr;
    uint32_t* scratchActiveIndices = nullptr;
    uint32_t* freeIndices = nullptr;
    uint8_t* aliveFlags = nullptr;
    SpawnCommand* spawnCommands = nullptr;
    uint32_t spawnCommandCapacity = 0;
    DeviceFrameState* deviceState = nullptr;
    DeviceTelemetry* deviceTelemetryScratch = nullptr;
    GraphParams* deviceGraphParams = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    cudaGraphNode_t graphSpawnMemcpyNode = nullptr;
    cudaMemcpy3DParms graphSpawnMemcpyParams = {};
    GraphKey currentGraphKey = {};
    cudaEvent_t latestFrameStart = nullptr;
    cudaEvent_t latestAfterSimulate = nullptr;
    cudaEvent_t latestAfterCompact = nullptr;
    cudaEvent_t latestAfterSpawn = nullptr;
    std::array<TelemetrySlot, kTelemetryRingSize> telemetrySlots = {};
    std::vector<SpawnUploadSlot> spawnUploadSlots;
    size_t nextTelemetrySlot = 0;
    uint32_t maxDispatchBlocks = 1;
    uint32_t activeWorkUpperBound = 0;
    uint64_t submittedSpawnUpperBoundTotal = 0;
    SimulationSettings settings = {};
    SimulationStats stats = {};
    std::vector<TileStats> tileStats;
    std::vector<EmitterState> emitters;
    std::vector<SpawnCommand> hostSpawnCommands;
    uint32_t frameIndex = 0;
    float simulationTime = 0.0f;
    bool hasSubmittedFrame = false;
    bool graphsEnabled = true;
    bool lastFrameUsedGraph = false;
    uint32_t graphRebuildCount = 0;
};

} // namespace vparticles
