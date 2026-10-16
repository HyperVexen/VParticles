#include "VParticles/GranularValidation.h"
#include "granular/GranularSystem.h"
#include "VParticles/ParticleSystem.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace vparticles {

namespace {

__global__ void checkMovingObstacleContactKernel(
    const float4* __restrict__ pos,
    const uint16_t* __restrict__ flags,
    uint32_t count,
    float3 sphereCenter,
    float sphereRadius,
    float grainRadius,
    uint32_t* __restrict__ d_awakeCount,
    uint32_t* __restrict__ d_tunnelCount,
    uint32_t* __restrict__ d_nanCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    if (isnan(p.x) || isnan(p.y) || isnan(p.z) || isinf(p.x) || isinf(p.y) || isinf(p.z)) {
        atomicAdd(d_nanCount, 1);
        return;
    }

    if (flags && !(flags[i] & kGrainFlagAsleep)) {
        atomicAdd(d_awakeCount, 1);
    }

    const float dx = p.x - sphereCenter.x;
    const float dy = p.y - sphereCenter.y;
    const float dz = p.z - sphereCenter.z;
    const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    const float rMin = sphereRadius + grainRadius;

    // Severe penetration / tunneling: inside obstacle by > 20% of grain radius
    if (dist < rMin - 0.20f * grainRadius) {
        atomicAdd(d_tunnelCount, 1);
    }
}

__global__ void checkGrainSanityKernel(
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    uint32_t count,
    uint32_t* __restrict__ d_nanCount,
    float* __restrict__ d_maxSpeed)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    const float4 v = vel[i];
    if (isnan(p.x) || isnan(p.y) || isnan(p.z) || isnan(v.x) || isnan(v.y) || isnan(v.z) ||
        isinf(p.x) || isinf(p.y) || isinf(p.z) || isinf(v.x) || isinf(v.y) || isinf(v.z)) {
        atomicAdd(d_nanCount, 1);
        return;
    }

    const float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    atomicMax(reinterpret_cast<int*>(d_maxSpeed), __float_as_int(speed));
}

} // namespace

ScaleValidationResult runScaleValidationSuite(uint32_t capacity)
{
    std::cout << "======================================================================\n"
              << "  SAGA 2 PHASE 11G: SCALE, STRESS TESTING & INTERACTIVE SAND VIEWER\n"
              << "======================================================================\n\n";

    ScaleValidationResult result;

    // =========================================================================
    // GATE 11G-1: Interactive Viewer Latency Budget (< 8.33 ms @ 100K -> 120+ FPS, < 16.6 ms @ 250K -> 60+ FPS)
    // =========================================================================
    std::cout << "[GATE 11G-1] Measuring Interactive Viewer Latency Budget (100K, 250K & 1M grains)...\n";
    {
        // 1. 100K Grains (Default Viewer Capacity) - Preview Quality (4 substeps x 4 iterations)
        {
            const uint32_t count100K = 100000;
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            settings.applyQuality(GranularQuality::Preview);
            settings.enableSleeping = true;

            GranularSystem sys100K(count100K, true);
            sys100K.setSettings(settings);
            sys100K.spawnPouringPile({0.0f, 0.6f, 0.0f}, count100K, 0.08f, 0.6f, 0.15f);
            sys100K.synchronize();

            // Warm up 10 frames
            for (int f = 0; f < 10; ++f) {
                sys100K.update(1.0f / 60.0f);
            }
            sys100K.synchronize();

            // Measure 30 frames
            float totalMs = 0.0f;
            for (int f = 0; f < 30; ++f) {
                sys100K.update(1.0f / 60.0f);
                sys100K.synchronize();
                totalMs += sys100K.stats().stepMs;
            }
            result.simMs100K = totalMs / 30.0f;
        }

        // 2. 250K Grains (Interactive Draft Viewport Mode) - Draft Quality (4 substeps x 2 iterations)
        {
            const uint32_t count250K = 250000;
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            settings.applyQuality(GranularQuality::Draft);
            settings.enableSleeping = true;

            GranularSystem sys250K(count250K, true);
            sys250K.setSettings(settings);
            sys250K.spawnPouringPile({0.0f, 0.6f, 0.0f}, count250K, 0.10f, 0.8f, 0.15f);
            sys250K.synchronize();

            // Warm up 10 frames
            for (int f = 0; f < 10; ++f) {
                sys250K.update(1.0f / 60.0f);
            }
            sys250K.synchronize();

            // Measure 30 frames
            float totalMs = 0.0f;
            for (int f = 0; f < 30; ++f) {
                sys250K.update(1.0f / 60.0f);
                sys250K.synchronize();
                totalMs += sys250K.stats().stepMs;
            }
            result.simMs250K = totalMs / 30.0f;
            result.fpsEquivalent250K = 1000.0f / fmaxf(result.simMs250K, 0.001f);
        }

        // 3. 1M Grains (Scaling Viewport Mode) - Draft Quality
        {
            const uint32_t count1M = 1000000;
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            settings.applyQuality(GranularQuality::Draft);
            settings.enableSleeping = true;

            GranularSystem sys1M(count1M, true);
            sys1M.setSettings(settings);
            sys1M.spawnColumnCollapse({0.0f, 0.0f, 0.0f}, count1M, 0.02f, 0.10f, 2202);
            sys1M.synchronize();

            // Warm up 10 frames
            for (int f = 0; f < 10; ++f) {
                sys1M.update(1.0f / 60.0f);
            }
            sys1M.synchronize();

            // Measure 20 frames
            float totalMs = 0.0f;
            for (int f = 0; f < 20; ++f) {
                sys1M.update(1.0f / 60.0f);
                sys1M.synchronize();
                totalMs += sys1M.stats().stepMs;
            }
            result.simMs1M = totalMs / 20.0f;
            result.fpsEquivalent1M = 1000.0f / fmaxf(result.simMs1M, 0.001f);
        }

        std::cout << "  - 100K Grains Step Latency:  " << std::fixed << std::setprecision(3)
                  << result.simMs100K << " ms (Budget: < 8.33 ms -> "
                  << (1000.0f / result.simMs100K) << " FPS, 120+ FPS target)\n"
                  << "  - 250K Grains Step Latency:  " << result.simMs250K
                  << " ms (Budget: < 16.6 ms -> " << result.fpsEquivalent250K << " FPS, 60+ FPS target)\n"
                  << "  - 1M Grains Step Latency:    " << result.simMs1M
                  << " ms (" << result.fpsEquivalent1M << " FPS viewport playback)\n";

        if (result.simMs100K < 8.33f && result.simMs250K < 16.6f) {
            result.viewerLatencyPassed = true;
            std::cout << "  -> Gate 11G-1 Result: PASSED (60+ to 120+ FPS interactive simulation verified)\n\n";
        } else {
            std::cout << "  -> Gate 11G-1 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11G-2: Multi-Million Active Grain Scaling (Awake Stress Test)
    // =========================================================================
    std::cout << "[GATE 11G-2] Verifying Multi-Million Awake Grain Scaling (1M - 2M awake grains)...\n";
    {
        const uint32_t scaleGrains = 2000000; // 2,000,000 continuously active grains
        result.awakeGrainsTested = scaleGrains;

        // VRAM footprint estimate
        result.memoryAllocatedMB = (scaleGrains * 60 + scaleGrains * 32 * 4 + scaleGrains * 8) / (1024 * 1024);

        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.enableSleeping = false; // 100% continuously awake stress test

        GranularSystem awakeSys(scaleGrains, false);
        awakeSys.setSettings(settings);
        awakeSys.spawnColumnCollapse({0.0f, 0.0f, 0.0f}, scaleGrains, 0.02f, 0.12f, 3303);
        awakeSys.synchronize();

        float totalMs = 0.0f;
        for (int f = 0; f < 30; ++f) {
            awakeSys.update(1.0f / 60.0f);
            awakeSys.synchronize();
            totalMs += awakeSys.stats().stepMs;
        }
        result.simMsAwake = totalMs / 30.0f;

        uint32_t* d_nanCount = nullptr;
        float* d_maxSpeed = nullptr;
        VP_CUDA_CHECK(cudaMalloc(&d_nanCount, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&d_maxSpeed, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_nanCount, 0, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_maxSpeed, 0, sizeof(float)));

        const uint32_t blocks = (scaleGrains + 255) / 256;
        checkGrainSanityKernel<<<blocks, 256>>>(
            awakeSys.pool().pos, awakeSys.pool().vel, scaleGrains, d_nanCount, d_maxSpeed);
        VP_CUDA_CHECK(cudaDeviceSynchronize());

        float maxSpeedHost = 0.0f;
        VP_CUDA_CHECK(cudaMemcpy(&result.awakeNanCount, d_nanCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&maxSpeedHost, d_maxSpeed, sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_nanCount);
        cudaFree(d_maxSpeed);

        std::cout << "  - Awake Active Grains:       " << result.awakeGrainsTested << "\n"
                  << "  - Memory Allocated:          ~" << result.memoryAllocatedMB << " MB VRAM\n"
                  << "  - Continuous Step Latency:   " << result.simMsAwake << " ms\n"
                  << "  - Maximum Grain Speed:       " << maxSpeedHost << " m/s\n"
                  << "  - NaN / Inf Violations:      " << result.awakeNanCount << "\n";

        if (result.awakeNanCount == 0 && maxSpeedHost > 0.1f && maxSpeedHost < 50.0f) {
            result.awakeScalingPassed = true;
            std::cout << "  -> Gate 11G-2 Result: PASSED (multi-million grains scale without sleeping)\n\n";
        } else {
            std::cout << "  -> Gate 11G-2 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11G-3: Kinematic Moving Collider Plowing & Non-Tunneling
    // =========================================================================
    std::cout << "[GATE 11G-3] Verifying Kinematic Moving Collider Plowing & Dynamic Wake-Up...\n";
    {
        const uint32_t testGrains = 100000;
        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.enableSleeping = true;
        settings.sleepSpeedThreshold = 0.02f;

        // Kinematic oscillating plow
        settings.kinematicColliderEnabled = 1;
        settings.kinematicColliderType = 0; // Linear sweep
        settings.kinematicOrigin = {0.0f, 0.10f, 0.0f};
        settings.kinematicRadius = 0.12f;
        settings.kinematicSpeed = 2.5f;
        settings.kinematicAmplitude = 0.35f;

        GranularSystem plowSys(testGrains, false);
        plowSys.setSettings(settings);
        plowSys.spawnPouringPile({0.0f, 0.6f, 0.0f}, testGrains, 0.08f, 0.5f, 0.12f, 4404);
        plowSys.synchronize();

        // Run for 100 frames so the plow sweeps continuously through the sand bed
        for (int f = 0; f < 100; ++f) {
            plowSys.update(1.0f / 60.0f);
        }
        plowSys.synchronize();

        uint32_t* d_awakeCount = nullptr;
        uint32_t* d_tunnelCount = nullptr;
        uint32_t* d_nanCount = nullptr;
        VP_CUDA_CHECK(cudaMalloc(&d_awakeCount, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&d_tunnelCount, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMalloc(&d_nanCount, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_awakeCount, 0, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_tunnelCount, 0, sizeof(uint32_t)));
        VP_CUDA_CHECK(cudaMemset(d_nanCount, 0, sizeof(uint32_t)));

        const uint32_t blocks = (testGrains + 255) / 256;
        float3 plowCenter = make_float3(
            plowSys.settings().obstacleSphere.center.x,
            plowSys.settings().obstacleSphere.center.y,
            plowSys.settings().obstacleSphere.center.z);

        checkMovingObstacleContactKernel<<<blocks, 256>>>(
            plowSys.pool().pos, plowSys.pool().flags, testGrains,
            plowCenter, settings.kinematicRadius, settings.grainRadius,
            d_awakeCount, d_tunnelCount, d_nanCount);
        VP_CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t tunnelCount = 0;
        VP_CUDA_CHECK(cudaMemcpy(&result.plowedGrainsWoken, d_awakeCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&tunnelCount, d_tunnelCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&result.kinematicNanCount, d_nanCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        cudaFree(d_awakeCount);
        cudaFree(d_tunnelCount);
        cudaFree(d_nanCount);

        result.nonTunnelingVerified = (tunnelCount == 0);

        std::cout << "  - Plow Speed / Frequency:    " << settings.kinematicSpeed << " rad/s\n"
                  << "  - Plow Center at Frame 100:  (" << std::fixed << std::setprecision(2)
                  << plowCenter.x << ", " << plowCenter.y << ", " << plowCenter.z << ")\n"
                  << "  - Dynamically Woken Grains:  " << result.plowedGrainsWoken << " grains\n"
                  << "  - Penetration / Tunneling:   " << tunnelCount << " violations\n"
                  << "  - NaN / Inf Violations:      " << result.kinematicNanCount << "\n";

        if (result.plowedGrainsWoken > 5000 && result.nonTunnelingVerified && result.kinematicNanCount == 0) {
            result.kinematicColliderPassed = true;
            std::cout << "  -> Gate 11G-3 Result: PASSED (moving plow wakes grains with non-tunneling)\n\n";
        } else {
            std::cout << "  -> Gate 11G-3 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11G-4: Scenario Hot-Switching Across All 7 Presets
    // =========================================================================
    std::cout << "[GATE 11G-4] Verifying Real-Time Switching Across All 7 Sand Scenarios...\n";
    {
        ParticleSystem testSys(100000, StorageMode::FP32, false);
        testSys.setSimulationMode(SimulationMode::Granular);

        GranularSettings defaultSettings;
        defaultSettings.grainRadius = 0.01f;
        defaultSettings.substeps = 4;
        defaultSettings.iterations = 4;
        defaultSettings.groundPlane.enabled = 1;
        defaultSettings.materials[0].cohesion = 60.0f;

        const char* scenarioNames[7] = {
            "pile", "column_collapse", "funnel", "dune", "cliff", "bonded_block", "clumps"
        };

        bool allScenariosClean = true;
        for (int s = 0; s < 7; ++s) {
            testSys.reset();
            testSys.setGranularSettings(defaultSettings);

            switch (s) {
            case 0: testSys.spawnGranularPouringPile({0.0f, 0.6f, 0.0f}, 50000, 0.08f, 0.5f, 0.12f); break;
            case 1: testSys.spawnGranularColumnCollapse({0.0f, 0.0f, 0.0f}, 50000, 0.02f, 0.12f); break;
            case 2: testSys.spawnGranularFunnel({0.0f, 0.4f, 0.0f}, 50000, 0.25f, 0.04f, 0.5f, 0.10f); break;
            case 3: testSys.spawnGranularDune({0.0f, 0.0f, 0.0f}, 50000, 1.0f, 0.8f, 25.0f, 0.10f); break;
            case 4: testSys.spawnGranularCliff({0.0f, 0.0f, 0.0f}, 50000, 0.02f, 0.12f); break;
            case 5: testSys.spawnGranularBondedBlock({-0.2f, 0.2f, 0.0f}, 2048, 0.02f, 0.15f, 1e-5f, 40.0f); break;
            case 6: testSys.spawnGranularClumps({0.0f, 0.5f, 0.0f}, 30000, 0.02f, 0.04f, 1e-5f, 25.0f, 0.10f); break;
            }

            testSys.synchronize();
            uint32_t count = testSys.granularStats().aliveCount;

            // Run 20 frames
            for (int f = 0; f < 20; ++f) {
                testSys.update(1.0f / 60.0f);
            }
            testSys.synchronize();

            const auto& st = testSys.granularStats();
            bool clean = (st.aliveCount == count && !isnan(st.meanHeight) && !isnan(st.kineticEnergy));
            if (!clean) {
                allScenariosClean = false;
            }

            std::cout << "  - Preset [" << scenarioNames[s] << "]: Alive " << st.aliveCount
                      << " | Mean Y: " << std::fixed << std::setprecision(3) << st.meanHeight
                      << " m | KE: " << st.kineticEnergy << (clean ? " -> OK" : " -> FAILED") << "\n";
            result.scenariosVerified++;
        }

        if (allScenariosClean && result.scenariosVerified == 7) {
            result.scenarioSwitchingPassed = true;
            std::cout << "  -> Gate 11G-4 Result: PASSED (all 7 sand presets stable across hot-switching)\n\n";
        } else {
            std::cout << "  -> Gate 11G-4 Result: FAILED\n\n";
        }
    }

    result.passed = result.viewerLatencyPassed && result.awakeScalingPassed &&
                    result.kinematicColliderPassed && result.scenarioSwitchingPassed;

    std::cout << "======================================================================\n";
    if (result.passed) {
        std::cout << "  PHASE 11G VALIDATION VERDICT: ALL GATES PASSED!\n";
    } else {
        std::cout << "  PHASE 11G VALIDATION VERDICT: GATES FAILED\n";
    }
    std::cout << "======================================================================\n\n";

    return result;
}

} // namespace vparticles
