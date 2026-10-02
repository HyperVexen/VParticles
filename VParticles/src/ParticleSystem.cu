#include "VParticles/ParticleSystem.h"

#include "VParticles/CudaCheck.h"
#include "core/InternalTypes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vparticles {

// ── Constant Memory Definitions ───────────────────────────────────────
__constant__ TileDesc cTiles[kMaxTiles];
__constant__ EffectRecipe cRecipes[kMaxRecipes];

} // namespace vparticles

#include "core/ParticleSystemImpl.h"

namespace vparticles {

ParticleSystem::ParticleSystem(uint32_t capacity, StorageMode mode, bool enableCudaGraphs)
    : impl_(new Impl(capacity, mode, enableCudaGraphs))
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
    impl_->currentGraphKey.valid = false;
    return static_cast<uint32_t>(impl_->emitters.size() - 1);
}

void ParticleSystem::clearEmitters()
{
    impl_->emitters.clear();
    impl_->currentGraphKey.valid = false;
}

void ParticleSystem::setEmitter(uint32_t index, const EmitterDesc& desc)
{
    if (index >= impl_->emitters.size()) {
        throw std::out_of_range("Emitter index out of range");
    }
    impl_->emitters[index].desc = desc;
    impl_->currentGraphKey.valid = false;
}

uint32_t ParticleSystem::emitterCount() const
{
    return static_cast<uint32_t>(impl_->emitters.size());
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
    const uint32_t migrateBlocks = impl_->dispatchBlockCount(
        impl_->activeWorkUpperBound + commandCoveredSpawn);
    const uint32_t commandCount = static_cast<uint32_t>(impl_->hostSpawnCommands.size());
    const GraphParams params = {
        requestedSpawn,
        impl_->frameIndex,
        impl_->simulationTime + dt,
        dt};

    // This small copy replaces variable scalar kernel arguments for both
    // submission modes. It is intentionally recorded before timing starts.
    VP_CUDA_CHECK(cudaMemcpyAsync(
        impl_->deviceGraphParams,
        &params,
        sizeof(params),
        cudaMemcpyHostToDevice,
        impl_->stream));

    bool graphFrame = false;
    Impl::SpawnUploadSlot* uploadSlot = nullptr;
    if (impl_->graphsEnabled) {
        if (commandCount != 0) {
            uploadSlot = &impl_->acquireSpawnUploadSlot(commandCount);
            std::copy(
                impl_->hostSpawnCommands.begin(),
                impl_->hostSpawnCommands.end(),
                uploadSlot->hostCommands);
        }

        const Impl::GraphKey graphKey = {
            simulationBlocks,
            spawnBlocks,
            migrateBlocks,
            commandCount,
            impl_->settings.recipeCount,
            impl_->settings.tileCount,
            true};
        if (!impl_->currentGraphKey.matches(graphKey)) {
            impl_->rebuildGraph(graphKey, uploadSlot);
        }
        if (uploadSlot != nullptr) {
            // The captured memcpy stays inside the graph. Its source rotates
            // through the pinned ring, avoiding host writes to an in-flight
            // source buffer while retaining a single graph executable.
            impl_->setGraphSpawnUploadSource(*uploadSlot);
        }
        graphFrame = true;
    }

    Impl::TelemetrySlot* telemetrySlot = impl_->acquireTelemetrySlot();

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestFrameStart, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->frameStart, impl_->stream));
        telemetrySlot->graphActive = graphFrame;
    }

    if (graphFrame) {
        VP_CUDA_CHECK(cudaGraphLaunch(impl_->graphExec, impl_->stream));
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSimulate, impl_->stream));
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterCompact, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSimulate, impl_->stream));
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterCompact, impl_->stream));
        }
    } else {
        beginFrameFromParamsKernel<<<1, 256, 0, impl_->stream>>>(
            impl_->deviceState, impl_->deviceTelemetryScratch, impl_->deviceGraphParams);
        VP_CUDA_CHECK(cudaGetLastError());
        if (impl_->mode_ == StorageMode::Packed) {
            const float posScale = impl_->settings.posQuantizationScale;
            simulatePackedKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                impl_->packedPool_, impl_->deviceState, impl_->aliveFlags, impl_->freeIndices,
                impl_->deviceGraphParams, posScale, 1.0f / posScale,
                impl_->settings.recipeCount, impl_->settings.tileCount);
        } else {
            simulateKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                impl_->pool, impl_->deviceState, impl_->aliveFlags, impl_->freeIndices,
                impl_->deviceGraphParams,
                impl_->settings.recipeCount, impl_->settings.tileCount);
        }
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSimulate, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSimulate, impl_->stream));
        }

        compactActiveIndicesKernel<<<simulationBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
            impl_->deviceState, impl_->aliveFlags);
        VP_CUDA_CHECK(cudaGetLastError());
        finalizeCompactionKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterCompact, impl_->stream));
        if (telemetrySlot != nullptr) {
            VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterCompact, impl_->stream));
        }

        if (commandCount != 0) {
            Impl::SpawnUploadSlot& eagerUploadSlot = impl_->acquireSpawnUploadSlot(commandCount);
            std::copy(impl_->hostSpawnCommands.begin(), impl_->hostSpawnCommands.end(), eagerUploadSlot.hostCommands);
            VP_CUDA_CHECK(cudaMemcpyAsync(
                impl_->spawnCommands, eagerUploadSlot.hostCommands,
                sizeof(SpawnCommand) * commandCount, cudaMemcpyHostToDevice, impl_->stream));
            VP_CUDA_CHECK(cudaEventRecord(eagerUploadSlot.ready, impl_->stream));
            eagerUploadSlot.inFlight = true;
        }
        prepareSpawnKernel<<<1, 1, 0, impl_->stream>>>(impl_->deviceState);
        VP_CUDA_CHECK(cudaGetLastError());
        if (commandCount != 0) {
            if (impl_->mode_ == StorageMode::Packed) {
                const float posScale = impl_->settings.posQuantizationScale;
                spawnBatchPackedKernel<<<spawnBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->packedPool_, impl_->deviceState, impl_->freeIndices, impl_->spawnCommands,
                    commandCount, impl_->settings.seed, posScale);
            } else {
                spawnBatchKernel<<<spawnBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->pool, impl_->deviceState, impl_->freeIndices, impl_->spawnCommands,
                    commandCount, impl_->settings.seed);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }
        if (impl_->settings.tileCount > 1) {
            if (impl_->mode_ == StorageMode::Packed) {
                const float posScale = impl_->settings.posQuantizationScale;
                migrateTilesPackedKernel<<<migrateBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->packedPool_, impl_->deviceState, impl_->deviceTelemetryScratch,
                    impl_->settings.tileCount, posScale, 1.0f / posScale);
            } else {
                migrateTilesKernel<<<migrateBlocks, kThreadsPerBlock, 0, impl_->stream>>>(
                    impl_->pool, impl_->deviceState, impl_->deviceTelemetryScratch,
                    impl_->settings.tileCount);
            }
            VP_CUDA_CHECK(cudaGetLastError());
        }
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->latestAfterSpawn, impl_->stream));
    if (telemetrySlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(telemetrySlot->afterSpawn, impl_->stream));
    }
    if (graphFrame && uploadSlot != nullptr) {
        VP_CUDA_CHECK(cudaEventRecord(uploadSlot->ready, impl_->stream));
        uploadSlot->inFlight = true;
    }

    impl_->advanceActiveWorkUpperBound(commandCoveredSpawn);
    if (telemetrySlot != nullptr) {
        impl_->queueTelemetry(*telemetrySlot);
    }

    impl_->simulationTime += dt;
    impl_->lastFrameUsedGraph = graphFrame;
    ++impl_->frameIndex;
    impl_->hasSubmittedFrame = true;
}

void ParticleSystem::reset()
{
    impl_->initializeFreeList();
    impl_->stats = {};
    impl_->stats.capacity = impl_->pool.capacity;
    impl_->frameIndex = 0;
    impl_->simulationTime = 0.0f;
    for (EmitterState& emitter : impl_->emitters) {
        emitter.spawnCarry = 0.0f;
        emitter.pendingBurst = 0;
    }
}

void ParticleSystem::setSettings(const SimulationSettings& settings)
{
    impl_->settings = settings;
    if (impl_->settings.recipeCount == 0) {
        impl_->settings.recipeCount = 1;
    }
    // Sync legacy/convenience module fields into recipe 0 only when in single-recipe mode
    if (impl_->settings.recipeCount <= 1) {
        impl_->settings.recipes[0].gravity = impl_->settings.gravity;
        impl_->settings.recipes[0].wind = impl_->settings.wind;
        impl_->settings.recipes[0].drag = impl_->settings.drag;
        impl_->settings.recipes[0].turbulence = impl_->settings.turbulence;
        impl_->settings.recipes[0].plane = impl_->settings.plane;
        impl_->settings.recipes[0].sphere = impl_->settings.sphere;
        impl_->settings.recipes[0].box = impl_->settings.box;
        impl_->settings.recipes[0].curves = impl_->settings.curves;
    } else {
        // Multi-recipe mode: sync recipe 0 into legacy fields for convenience readers
        impl_->settings.gravity = impl_->settings.recipes[0].gravity;
        impl_->settings.wind = impl_->settings.recipes[0].wind;
        impl_->settings.drag = impl_->settings.recipes[0].drag;
        impl_->settings.turbulence = impl_->settings.recipes[0].turbulence;
        impl_->settings.plane = impl_->settings.recipes[0].plane;
        impl_->settings.sphere = impl_->settings.recipes[0].sphere;
        impl_->settings.box = impl_->settings.recipes[0].box;
        impl_->settings.curves = impl_->settings.recipes[0].curves;
    }

    if (impl_->settings.tileCount <= 1) {
        impl_->settings.tileCount = 1;
        if (settings.tiles[0].originX == 0.0f && settings.tiles[0].originY == 0.0f && settings.tiles[0].originZ == 0.0f) {
            impl_->settings.tiles[0].originX = settings.tileOrigin.x;
            impl_->settings.tiles[0].originY = settings.tileOrigin.y;
            impl_->settings.tiles[0].originZ = settings.tileOrigin.z;
        } else {
            impl_->settings.tileOrigin = {
                settings.tiles[0].originX,
                settings.tiles[0].originY,
                settings.tiles[0].originZ
            };
        }
    }
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cTiles,
        impl_->settings.tiles,
        sizeof(TileDesc) * kMaxTiles));
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
    impl_->currentGraphKey.valid = false;
}

const SimulationSettings& ParticleSystem::settings() const
{
    return impl_->settings;
}

StorageMode ParticleSystem::storageMode() const
{
    return impl_->mode_;
}

uint32_t ParticleSystem::addRecipe(const EffectRecipe& recipe)
{
    if (impl_->settings.recipeCount >= kMaxRecipes) {
        throw std::runtime_error("Maximum recipe count reached (kMaxRecipes = 64)");
    }
    const uint32_t id = impl_->settings.recipeCount++;
    impl_->settings.recipes[id] = recipe;
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
    impl_->currentGraphKey.valid = false;
    return id;
}

void ParticleSystem::setRecipe(uint32_t recipeId, const EffectRecipe& recipe)
{
    if (recipeId >= kMaxRecipes) {
        throw std::out_of_range("Recipe ID out of range");
    }
    impl_->settings.recipes[recipeId] = recipe;
    if (recipeId >= impl_->settings.recipeCount) {
        impl_->settings.recipeCount = recipeId + 1;
        impl_->currentGraphKey.valid = false;
    }
    VP_CUDA_CHECK(cudaMemcpyToSymbol(
        cRecipes,
        impl_->settings.recipes,
        sizeof(EffectRecipe) * kMaxRecipes));
}

const EffectRecipe& ParticleSystem::recipe(uint32_t recipeId) const
{
    if (recipeId >= kMaxRecipes) {
        throw std::out_of_range("Recipe ID out of range");
    }
    return impl_->settings.recipes[recipeId];
}

uint32_t ParticleSystem::recipeCount() const
{
    return impl_->settings.recipeCount;
}

void ParticleSystem::synchronize()
{
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    impl_->pollTelemetry();
    impl_->pollSpawnUploadSlots();
    if (!impl_->hasSubmittedFrame) {
        return;
    }

    DeviceTelemetry telemetry = {};
    snapshotTelemetryKernel<<<1, 1, 0, impl_->stream>>>(
        impl_->deviceState,
        impl_->deviceTelemetryScratch,
        impl_->settings.tileCount);
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
    if (impl_->lastFrameUsedGraph) {
        simulateMs = totalMs;
        compactMs = 0.0f;
        spawnMs = 0.0f;
    }
    impl_->publishTelemetry(
        telemetry, 0, spawnMs, simulateMs, compactMs, totalMs,
        impl_->lastFrameUsedGraph);
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

const std::vector<TileStats>& ParticleSystem::tileStats() const
{
    return impl_->tileStats;
}

} // namespace vparticles
