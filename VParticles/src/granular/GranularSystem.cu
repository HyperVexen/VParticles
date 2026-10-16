#include "granular/GranularSystem.h"
#include "granular/GranularKernels.cuh"
#include "granular/GranularNeighborSearch.h"
#include "granular/GranularBondManager.h"
#include "VParticles/CudaCheck.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace vparticles {

struct GranularSystem::Impl {
    uint32_t capacity = 0;
    bool enableCudaGraphs = true;
    GranularSettings settings;
    GranularPool pool;
    GranularStats stats;
    std::unique_ptr<IContactSolver> solver;
    std::unique_ptr<GranularNeighborSearch> neighborSearch;
    std::unique_ptr<GranularBondManager> bondManager;

    float accumulator = 0.0f;
    uint64_t frameIndex = 0;
    float simTime = 0.0f;

    cudaStream_t stream = nullptr;
    cudaEvent_t frameStart = nullptr;
    cudaEvent_t frameEnd = nullptr;

    float* d_stats = nullptr;

    // CUDA Graph state
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool graphValid = false;

    Impl(uint32_t cap, bool graphs)
        : capacity(cap), enableCudaGraphs(graphs)
    {
        VP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        VP_CUDA_CHECK(cudaEventCreate(&frameStart));
        VP_CUDA_CHECK(cudaEventCreate(&frameEnd));

        pool.capacity = cap;
        pool.aliveCount = 0;

        // Allocate Device Arrays for GranularPool
        VP_CUDA_CHECK(cudaMalloc(&pool.pos, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.vel, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.pred, sizeof(float4) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.tint, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.materialId, sizeof(uint16_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.flags, sizeof(uint16_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&pool.grainId, sizeof(uint32_t) * cap));
        VP_CUDA_CHECK(cudaMalloc(&d_stats, sizeof(float) * 4));

        VP_CUDA_CHECK(cudaMemsetAsync(pool.vel, 0, sizeof(float4) * cap, stream));
        VP_CUDA_CHECK(cudaMemsetAsync(pool.flags, 0, sizeof(uint16_t) * cap, stream));

        solver = createGaussSeidelContactSolver();
        neighborSearch = std::make_unique<GranularNeighborSearch>(cap);
        bondManager = std::make_unique<GranularBondManager>(cap);
    }

    ~Impl()
    {
        destroyGraph();
        if (d_stats) cudaFree(d_stats);
        if (pool.pos) cudaFree(pool.pos);
        if (pool.vel) cudaFree(pool.vel);
        if (pool.pred) cudaFree(pool.pred);
        if (pool.tint) cudaFree(pool.tint);
        if (pool.materialId) cudaFree(pool.materialId);
        if (pool.flags) cudaFree(pool.flags);
        if (pool.grainId) cudaFree(pool.grainId);

        if (frameStart) cudaEventDestroy(frameStart);
        if (frameEnd) cudaEventDestroy(frameEnd);
        if (stream) cudaStreamDestroy(stream);
    }

    void destroyGraph()
    {
        if (graphExec) {
            cudaGraphExecDestroy(graphExec);
            graphExec = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
        graphValid = false;
    }

    void stepSubsteps(float fixedDt)
    {
        if (pool.aliveCount == 0) return;

        const uint32_t blocks = (pool.aliveCount + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
        const float substepDt = fixedDt / static_cast<float>(settings.substeps);
        const float invSubstepDt = 1.0f / substepDt;
        const float3 grav = make_float3(settings.gravity.x, settings.gravity.y, settings.gravity.z);
        const uint32_t sleepThresholdSubsteps = settings.sleepFrames * settings.substeps;

        for (uint32_t s = 0; s < settings.substeps; ++s) {
            // 1. Predict Pass
            granularPredictKernel<<<blocks, kGranularThreadsPerBlock, 0, stream>>>(
                pool.pos, pool.vel, pool.pred, pool.flags, pool.aliveCount,
                grav, substepDt, settings.materials[0].drag);

            // 2. Contact & Constraint Solve Iterations
            for (uint32_t iter = 0; iter < settings.iterations; ++iter) {
                if (solver) {
                    solver->solve(pool, neighborSearch->neighborData(), settings, substepDt, stream);
                }
                if (bondManager) {
                    bondManager->solveBonds(pool, substepDt, stream);
                }
            }

            // 3. Integrate Pass (with tunneling cap and sleeping deactivation)
            granularIntegrateKernel<<<blocks, kGranularThreadsPerBlock, 0, stream>>>(
                pool.pos, pool.vel, pool.pred, pool.flags, pool.aliveCount,
                invSubstepDt, substepDt, settings.materials[0].damping,
                settings.enableSleeping, settings.sleepSpeedThreshold, sleepThresholdSubsteps);
        }

        // Reduce stats (heights, kinetic energy, asleep count)
        VP_CUDA_CHECK(cudaMemsetAsync(d_stats, 0, sizeof(float) * 4, stream));
        granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, stream>>>(
            pool.pos, pool.vel, pool.flags, pool.aliveCount, d_stats);
    }

    void stepFixed(float fixedDt)
    {
        if (pool.aliveCount == 0) return;

        // Phase 11G: Dynamic Kinematic Moving Collider Trajectory
        if (settings.kinematicColliderEnabled) {
            destroyGraph(); // Kinematic moving collider continuously updates collider bounds
            const float t = simTime;
            if (settings.kinematicColliderType == 0) {
                // Linear oscillating bulldozer plow
                const float sweepX = settings.kinematicOrigin.x + sinf(t * settings.kinematicSpeed) * settings.kinematicAmplitude;
                const float sweepVx = cosf(t * settings.kinematicSpeed) * settings.kinematicSpeed * settings.kinematicAmplitude;
                settings.obstacleSphere.center = {sweepX, settings.kinematicOrigin.y, settings.kinematicOrigin.z};
                settings.obstacleSphere.radius = settings.kinematicRadius;
                settings.obstacleSphere.enabled = 1;
                settings.obstacleSphere.invert = 0;
                settings.kinematicVelocity = {sweepVx, 0.0f, 0.0f};
            } else if (settings.kinematicColliderType == 1) {
                // Rotating circular churner paddle
                const float orbitX = settings.kinematicOrigin.x + cosf(t * settings.kinematicSpeed) * settings.kinematicAmplitude;
                const float orbitZ = settings.kinematicOrigin.z + sinf(t * settings.kinematicSpeed) * settings.kinematicAmplitude;
                const float orbitVx = -sinf(t * settings.kinematicSpeed) * settings.kinematicSpeed * settings.kinematicAmplitude;
                const float orbitVz = cosf(t * settings.kinematicSpeed) * settings.kinematicSpeed * settings.kinematicAmplitude;
                settings.obstacleSphere.center = {orbitX, settings.kinematicOrigin.y, orbitZ};
                settings.obstacleSphere.radius = settings.kinematicRadius;
                settings.obstacleSphere.enabled = 1;
                settings.obstacleSphere.invert = 0;
                settings.kinematicVelocity = {orbitVx, 0.0f, orbitVz};
            }
        }

        // Check if skin displacement was breached
        if (neighborSearch && neighborSearch->checkRebuildTrigger(pool, settings, stream)) {
            destroyGraph();
            neighborSearch->build(pool, settings, stream);
            if (bondManager) {
                bondManager->reorder(neighborSearch->sortedIndices(), pool.grainId, pool.aliveCount, stream);
            }
        }

        if (enableCudaGraphs && pool.aliveCount > 0) {
            if (!graphValid) {
                destroyGraph();
                VP_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                stepSubsteps(fixedDt);
                VP_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                VP_CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
                graphValid = true;
            }
            VP_CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        } else {
            stepSubsteps(fixedDt);
        }
        simTime += fixedDt;
    }
};

GranularSystem::GranularSystem(uint32_t capacity, bool enableCudaGraphs)
    : impl_(std::make_unique<Impl>(capacity, enableCudaGraphs))
{
}

GranularSystem::~GranularSystem() = default;

void GranularSystem::setSettings(const GranularSettings& settings)
{
    impl_->settings = settings;
    impl_->destroyGraph(); // Settings change invalidates graph topology
}

const GranularSettings& GranularSystem::settings() const
{
    return impl_->settings;
}

const GranularPool& GranularSystem::pool() const
{
    return impl_->pool;
}

const GranularStats& GranularSystem::stats() const
{
    return impl_->stats;
}

void GranularSystem::setContactSolver(std::unique_ptr<IContactSolver> solver)
{
    impl_->solver = std::move(solver);
    impl_->destroyGraph();
}

const IContactSolver* GranularSystem::contactSolver() const
{
    return impl_->solver.get();
}

void GranularSystem::spawnLattice(
    const Float3& origin,
    uint32_t count,
    float spacing,
    float jitterRatio,
    const Float3& initialVel,
    uint32_t seed,
    uint16_t matId)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Lattice spawn count exceeds GranularPool capacity");
    }

    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(count))));
    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);
    const float3 initV = make_float3(initialVel.x, initialVel.y, initialVel.z);
    const float density = (matId < impl_->settings.materialCount) ? impl_->settings.materials[matId].density : 1600.0f;

    granularSpawnLatticeKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, spacing, side, side,
        impl_->settings.grainRadius, jitterRatio, initV, seed, matId, density);
    VP_CUDA_CHECK(cudaGetLastError());

    // Reduce stats for the freshly spawned lattice
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph(); // Alive count changed
}

void GranularSystem::spawnColumnCollapse(
    const Float3& origin,
    uint32_t count,
    float spacing,
    float jitterRatio,
    uint32_t seed,
    uint16_t matId)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Column collapse spawn count exceeds GranularPool capacity");
    }

    const uint32_t sideX = std::max(2u, static_cast<uint32_t>(std::round(std::cbrt(static_cast<float>(count) / 3.0f))));
    const uint32_t sideZ = sideX;
    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnColumnKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, spacing, sideX, sideZ,
        impl_->settings.grainRadius, jitterRatio, seed, matId);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnPouringPile(
    const Float3& origin,
    uint32_t count,
    float nozzleRadius,
    float height,
    float jitterRatio,
    uint32_t seed,
    uint16_t matId)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Pouring pile spawn count exceeds GranularPool capacity");
    }

    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnPouringPileKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, nozzleRadius, height,
        impl_->settings.grainRadius, jitterRatio, seed, matId);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnFunnel(
    const Float3& origin,
    uint32_t count,
    float topRadius,
    float orificeRadius,
    float height,
    float jitterRatio,
    uint32_t seed)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Funnel spawn count exceeds GranularPool capacity");
    }

    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnFunnelKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, topRadius, orificeRadius, height,
        impl_->settings.grainRadius, jitterRatio, seed);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnDune(
    const Float3& origin,
    uint32_t count,
    float lengthX,
    float widthZ,
    float slopeAngleDeg,
    float jitterRatio,
    uint32_t seed)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Dune spawn count exceeds GranularPool capacity");
    }

    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnDuneKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, lengthX, widthZ, slopeAngleDeg,
        impl_->settings.grainRadius, jitterRatio, seed);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnCliff(
    const Float3& origin,
    uint32_t count,
    float spacing,
    float jitterRatio,
    uint32_t seed)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Cliff spawn count exceeds GranularPool capacity");
    }

    const uint32_t sideX = 16;
    const uint32_t sideZ = 12;
    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnCliffKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, spacing, sideX, sideZ,
        impl_->settings.grainRadius, jitterRatio, seed);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnBondedBlock(
    const Float3& origin,
    uint32_t count,
    float spacing,
    float anchorRatio,
    float compliance,
    float breakForce,
    uint32_t seed)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("BondedBlock spawn count exceeds GranularPool capacity");
    }

    const uint32_t sideY = 4;
    const uint32_t sideZ = 4;
    const uint32_t sideX = std::max(4u, count / (sideY * sideZ));
    const uint32_t actualCount = std::min(count, sideX * sideY * sideZ);
    const uint32_t blocks = (actualCount + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnBondedBlockKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        actualCount, orig, spacing, sideX, sideY, sideZ, anchorRatio,
        impl_->settings.grainRadius, seed);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, actualCount, impl_->d_stats);

    impl_->pool.aliveCount = actualCount;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }

    if (impl_->bondManager) {
        impl_->bondManager->createLatticeBonds(
            impl_->pool, spacing * 1.5f, compliance, breakForce, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::spawnClumps(
    const Float3& origin,
    uint32_t count,
    float spacing,
    float clumpRadius,
    float compliance,
    float breakForce,
    float jitterRatio,
    uint32_t seed)
{
    if (count > impl_->capacity) {
        throw std::invalid_argument("Clumps spawn count exceeds GranularPool capacity");
    }

    const uint32_t sideX = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(count))));
    const uint32_t sideZ = sideX;
    const uint32_t blocks = (count + kGranularThreadsPerBlock - 1) / kGranularThreadsPerBlock;
    const float3 orig = make_float3(origin.x, origin.y, origin.z);

    granularSpawnClumpsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.pred,
        impl_->pool.tint, impl_->pool.materialId, impl_->pool.flags, impl_->pool.grainId,
        count, orig, spacing, sideX, sideZ, clumpRadius,
        impl_->settings.grainRadius, jitterRatio, seed);
    VP_CUDA_CHECK(cudaGetLastError());

    VP_CUDA_CHECK(cudaMemsetAsync(impl_->d_stats, 0, sizeof(float) * 4, impl_->stream));
    granularReduceStatsKernel<<<blocks, kGranularThreadsPerBlock, 0, impl_->stream>>>(
        impl_->pool.pos, impl_->pool.vel, impl_->pool.flags, count, impl_->d_stats);

    impl_->pool.aliveCount = count;
    if (impl_->neighborSearch) {
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
    }

    if (impl_->bondManager) {
        impl_->bondManager->createClumpBonds(
            impl_->pool, clumpRadius, spacing * 1.5f, compliance, breakForce, seed, impl_->stream);
    }
    impl_->destroyGraph();
}

void GranularSystem::update(float dt)
{
    VP_CUDA_CHECK(cudaEventRecord(impl_->frameStart, impl_->stream));

    // Clamp dt to prevent spiral of death
    dt = std::min(dt, 0.2f);
    impl_->accumulator += dt;

    const float fixedDt = impl_->settings.fixedDt;
    while (impl_->accumulator >= fixedDt) {
        impl_->stepFixed(fixedDt);
        impl_->accumulator -= fixedDt;
    }

    VP_CUDA_CHECK(cudaEventRecord(impl_->frameEnd, impl_->stream));
    impl_->frameIndex++;
}

void GranularSystem::reset()
{
    impl_->pool.aliveCount = 0;
    impl_->accumulator = 0.0f;
    impl_->frameIndex = 0;
    impl_->simTime = 0.0f;
    impl_->stats = {};
    impl_->destroyGraph();
    if (impl_->bondManager) {
        impl_->bondManager->reset();
    }
    VP_CUDA_CHECK(cudaMemsetAsync(impl_->pool.flags, 0, sizeof(uint16_t) * impl_->capacity, impl_->stream));
}

void GranularSystem::synchronize()
{
    VP_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

    float ms = 0.0f;
    if (impl_->frameIndex > 0) {
        VP_CUDA_CHECK(cudaEventElapsedTime(&ms, impl_->frameStart, impl_->frameEnd));
    }

    float hostStats[4] = {};
    VP_CUDA_CHECK(cudaMemcpy(hostStats, impl_->d_stats, sizeof(hostStats), cudaMemcpyDeviceToHost));

    impl_->stats.frameIndex = impl_->frameIndex;
    impl_->stats.simTime = impl_->simTime;
    impl_->stats.capacity = impl_->capacity;
    impl_->stats.aliveCount = impl_->pool.aliveCount;
    impl_->stats.substeps = impl_->settings.substeps;
    impl_->stats.iterations = impl_->settings.iterations;
    impl_->stats.meanHeight = impl_->pool.aliveCount > 0 ? (hostStats[0] / static_cast<float>(impl_->pool.aliveCount)) : 0.0f;
    impl_->stats.maxHeight = hostStats[1];
    impl_->stats.kineticEnergy = hostStats[2];
    impl_->stats.asleepCount = static_cast<uint32_t>(hostStats[3]);
    impl_->stats.stepMs = ms;
    impl_->stats.totalMs += ms;

    if (impl_->bondManager && impl_->pool.aliveCount > 0) {
        impl_->stats.activeBondCount = impl_->bondManager->getActiveBondCount(impl_->pool.aliveCount, impl_->stream);
        impl_->stats.brokenBondCount = impl_->bondManager->getBrokenBondCount(impl_->stream);
    }

    if (impl_->neighborSearch) {
        const auto& nStats = impl_->neighborSearch->stats();
        impl_->stats.neighborRebuildCount = nStats.totalRebuilds;
        impl_->stats.neighborOverflowCount = nStats.lastOverflowCount;
        impl_->stats.neighborRebuildMs = nStats.lastRebuildMs;
        impl_->stats.neighborAmortizedMs = (impl_->stats.frameIndex > 0)
            ? (nStats.lastRebuildMs * static_cast<float>(nStats.totalRebuilds) / static_cast<float>(impl_->stats.frameIndex))
            : nStats.lastRebuildMs;
    }
}

const NeighborData& GranularSystem::neighborData() const
{
    return impl_->neighborSearch->neighborData();
}

const NeighborStats& GranularSystem::neighborStats() const
{
    return impl_->neighborSearch->stats();
}

GranularBondManager* GranularSystem::bondManager() const
{
    return impl_->bondManager.get();
}

void GranularSystem::rebuildNeighbors()
{
    if (impl_->neighborSearch && impl_->pool.aliveCount > 0) {
        impl_->destroyGraph();
        impl_->neighborSearch->build(impl_->pool, impl_->settings, impl_->stream);
        if (impl_->bondManager) {
            impl_->bondManager->reorder(impl_->neighborSearch->sortedIndices(), impl_->pool.grainId, impl_->pool.aliveCount, impl_->stream);
        }
    }
}

} // namespace vparticles
