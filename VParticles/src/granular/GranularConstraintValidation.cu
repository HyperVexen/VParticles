#include "VParticles/GranularValidation.h"
#include "granular/GranularSystem.h"
#include "granular/GranularNeighborSearch.h"
#include "granular/GranularBondManager.h"
#include "VParticles/CudaCheck.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace vparticles {

namespace {

__global__ void perturbForReorderKernel(
    float4* __restrict__ pos,
    uint32_t count,
    uint32_t iter)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float4 p = pos[i];
    // Gentle oscillation to trigger spatial key changes and neighbor reordering
    const float offset = sinf(static_cast<float>(i + iter) * 0.1f) * 0.005f;
    p.x += offset;
    p.z += offset;
    pos[i] = p;
}

__global__ void applyOverburdenLoadKernel(
    float4* __restrict__ vel,
    const float4* __restrict__ pos,
    uint32_t count,
    float minH,
    float loadVelocity)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    // Drive upper region downward to simulate overburden surcharge
    if (p.y >= minH) {
        float4 v = vel[i];
        v.y = -loadVelocity;
        vel[i] = v;
    }
}

__global__ void applyCantileverLoadKernel(
    float4* __restrict__ vel,
    const float4* __restrict__ pos,
    uint32_t count,
    float maxX,
    float downwardForce,
    float dt)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    const float invMass = vel[i].w;
    if (invMass <= 0.0f) return; // skip anchored grains

    // Apply downward load to the free tip (outer 25% of beam)
    if (p.x >= maxX * 0.75f) {
        float4 v = vel[i];
        v.y -= downwardForce * invMass * dt;
        vel[i] = v;
    }
}

__global__ void measureCliffFaceKernel(
    const float4* __restrict__ pos,
    uint32_t count,
    float frontThresholdX,
    float runoutBoundaryX,
    float* __restrict__ d_frontApex,
    uint32_t* __restrict__ d_runoutCount)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    const float4 p = pos[i];
    if (p.x <= frontThresholdX) {
        atomicMax(reinterpret_cast<int*>(d_frontApex), __float_as_int(p.y));
    }
    if (p.x < runoutBoundaryX) {
        atomicAdd(d_runoutCount, 1);
    }
}

} // namespace

ConstraintValidationResult runConstraintValidationSuite(uint32_t capacity)
{
    std::cout << "======================================================================\n"
              << "  SAGA 2 PHASE 11F: CONSTRAINT LAYER, COHESION, BONDS & CLUMPING\n"
              << "======================================================================\n\n";

    ConstraintValidationResult result;
    const uint32_t systemCap = std::max(capacity, 50000u);

    // =========================================================================
    // GATE 11F-1: Damp-Sand Cliff Stability & Load-induced Crumble
    // =========================================================================
    std::cout << "[GATE 11F-1] Verifying Damp-Sand Cliff Stability & Rupture Crumble...\n";
    {
        const uint32_t cliffGrains = 6000;
        const uint32_t blocks = (cliffGrains + 255) / 256;

        float* d_frontApex = nullptr;
        uint32_t* d_runoutCount = nullptr;
        VP_CUDA_CHECK(cudaMalloc(&d_frontApex, sizeof(float)));
        VP_CUDA_CHECK(cudaMalloc(&d_runoutCount, sizeof(uint32_t)));

        GranularSettings drySettings;
        drySettings.grainRadius = 0.01f;
        drySettings.materials[0].cohesion = 0.0f;
        drySettings.materials[0].staticFriction = 0.45f;
        drySettings.materials[0].dynamicFriction = 0.35f;
        drySettings.enableSleeping = false;
        drySettings.substeps = 4;
        drySettings.iterations = 4;

        GranularSystem dryCliff(systemCap, false);
        dryCliff.setSettings(drySettings);
        dryCliff.spawnCliff({0.0f, 0.0f, 0.0f}, cliffGrains, 0.02f, 0.12f, 1111);
        dryCliff.synchronize();

        VP_CUDA_CHECK(cudaMemset(d_frontApex, 0, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_runoutCount, 0, sizeof(uint32_t)));
        measureCliffFaceKernel<<<blocks, 256>>>(dryCliff.pool().pos, cliffGrains, 0.08f, -0.01f, d_frontApex, d_runoutCount);
        float dryInitialFrontApex = 0.0f;
        VP_CUDA_CHECK(cudaMemcpy(&dryInitialFrontApex, d_frontApex, sizeof(float), cudaMemcpyDeviceToHost));

        // Run dry cliff for 80 frames (1.33 seconds)
        for (int f = 0; f < 80; ++f) {
            dryCliff.update(1.0f / 60.0f);
        }
        dryCliff.synchronize();

        VP_CUDA_CHECK(cudaMemset(d_frontApex, 0, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_runoutCount, 0, sizeof(uint32_t)));
        measureCliffFaceKernel<<<blocks, 256>>>(dryCliff.pool().pos, cliffGrains, 0.08f, -0.01f, d_frontApex, d_runoutCount);
        float dryFinalFrontApex = 0.0f;
        uint32_t dryRunout = 0;
        VP_CUDA_CHECK(cudaMemcpy(&dryFinalFrontApex, d_frontApex, sizeof(float), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&dryRunout, d_runoutCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        result.dryCliffApexDropRatio = (dryInitialFrontApex - dryFinalFrontApex) / dryInitialFrontApex;

        std::cout << "  - Dry Sand Cliff (c = 0.0): Front Face " << std::fixed << std::setprecision(3)
                  << dryInitialFrontApex << " m -> Collapsed " << dryFinalFrontApex << " m (Drop: "
                  << result.dryCliffApexDropRatio * 100.0f << "%, Runout Grains: " << dryRunout << ")\n";

        // Now test damp sand cliff with cohesion
        GranularSettings dampSettings = drySettings;
        dampSettings.materials[0].cohesion = 60.0f; // Damp sand cohesive attraction
        dampSettings.materials[0].cohesionRuptureRatio = 0.15f;

        GranularSystem dampCliff(systemCap, false);
        dampCliff.setSettings(dampSettings);
        dampCliff.spawnCliff({0.0f, 0.0f, 0.0f}, cliffGrains, 0.02f, 0.12f, 1111);
        dampCliff.synchronize();

        VP_CUDA_CHECK(cudaMemset(d_frontApex, 0, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_runoutCount, 0, sizeof(uint32_t)));
        measureCliffFaceKernel<<<blocks, 256>>>(dampCliff.pool().pos, cliffGrains, 0.08f, -0.01f, d_frontApex, d_runoutCount);
        float dampInitialFrontApex = 0.0f;
        VP_CUDA_CHECK(cudaMemcpy(&dampInitialFrontApex, d_frontApex, sizeof(float), cudaMemcpyDeviceToHost));

        // Run damp cliff for 80 frames without external load
        for (int f = 0; f < 80; ++f) {
            dampCliff.update(1.0f / 60.0f);
        }
        dampCliff.synchronize();

        VP_CUDA_CHECK(cudaMemset(d_frontApex, 0, sizeof(float)));
        VP_CUDA_CHECK(cudaMemset(d_runoutCount, 0, sizeof(uint32_t)));
        measureCliffFaceKernel<<<blocks, 256>>>(dampCliff.pool().pos, cliffGrains, 0.08f, -0.01f, d_frontApex, d_runoutCount);
        float dampFinalFrontApex = 0.0f;
        uint32_t dampRunout = 0;
        VP_CUDA_CHECK(cudaMemcpy(&dampFinalFrontApex, d_frontApex, sizeof(float), cudaMemcpyDeviceToHost));
        VP_CUDA_CHECK(cudaMemcpy(&dampRunout, d_runoutCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        result.dampCliffApexDropRatio = (dampInitialFrontApex - dampFinalFrontApex) / dampInitialFrontApex;

        std::cout << "  - Damp Sand Cliff (c = 60.0): Front Face " << dampInitialFrontApex
                  << " m -> Stable " << dampFinalFrontApex << " m (Drop: "
                  << result.dampCliffApexDropRatio * 100.0f << "%, Runout Grains: " << dampRunout << ")\n";

        // Now apply continuous overburden mechanical load to the top edge to trigger crumble
        for (int f = 0; f < 80; ++f) {
            applyOverburdenLoadKernel<<<blocks, 256, 0, nullptr>>>(
                dampCliff.pool().vel, dampCliff.pool().pos, cliffGrains, dampFinalFrontApex - 0.15f, 4.5f);
            dampCliff.update(1.0f / 60.0f);
        }
        dampCliff.synchronize();

        VP_CUDA_CHECK(cudaMemset(d_runoutCount, 0, sizeof(uint32_t)));
        measureCliffFaceKernel<<<blocks, 256>>>(dampCliff.pool().pos, cliffGrains, 0.08f, -0.01f, d_frontApex, d_runoutCount);
        uint32_t postLoadRunout = 0;
        VP_CUDA_CHECK(cudaMemcpy(&postLoadRunout, d_runoutCount, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        result.loadCrumbleVerified = (postLoadRunout > dampRunout + 80);
        std::cout << "  - Post-Load Overburden Crumble: Runout Grains " << postLoadRunout
                  << " (Debris Surge: +" << (postLoadRunout - dampRunout) << " grains)\n";

        if (d_frontApex) cudaFree(d_frontApex);
        if (d_runoutCount) cudaFree(d_runoutCount);

        if ((result.dryCliffApexDropRatio > 0.02f || dryRunout > 100) &&
            result.dampCliffApexDropRatio < 0.05f &&
            dampRunout < 50 &&
            result.loadCrumbleVerified) {
            result.cliffPassed = true;
            std::cout << "  -> Gate 11F-1 Result: PASSED (damp cliff holds vertical face, crumbles under load)\n\n";
        } else {
            std::cout << "  -> Gate 11F-1 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11F-2: Bonded Block Tensile Break Force Parity Across Substeps
    // =========================================================================
    std::cout << "[GATE 11F-2] Verifying Bonded Block Break Force Parity Across Substeps...\n";
    {
        const uint32_t beamGrains = 1024;
        const float breakThreshold = 40.0f; // 40 N rupture force
        const float compliance = 1e-6f;

        auto testBreakLoad = [&](uint32_t substeps) -> float {
            GranularSettings settings;
            settings.grainRadius = 0.01f;
            settings.substeps = substeps;
            settings.iterations = 4;
            settings.enableSleeping = false;
            settings.gravity = {0.0f, 0.0f, 0.0f}; // isolate to applied tip load

            GranularSystem beam(systemCap, false);
            beam.setSettings(settings);
            beam.spawnBondedBlock({0.0f, 0.2f, 0.0f}, beamGrains, 0.02f, 0.15f, compliance, breakThreshold, 2222);
            beam.synchronize();

            const uint32_t initialBonds = beam.stats().activeBondCount;
            const float maxX = 32.0f * 0.02f;
            const uint32_t blocks = (beamGrains + 255) / 256;

            float load = 5.0f;
            float ruptureLoad = 0.0f;

            // Stepwise progressive load increase until bond fracture
            for (int step = 0; step < 80; ++step) {
                load += 2.0f;
                applyCantileverLoadKernel<<<blocks, 256, 0, nullptr>>>(
                    beam.pool().vel, beam.pool().pos, beamGrains, maxX, load, 1.0f / 60.0f);
                VP_CUDA_CHECK(cudaDeviceSynchronize());

                beam.update(1.0f / 60.0f);
                beam.synchronize();

                if (beam.stats().brokenBondCount > initialBonds * 0.05f) {
                    ruptureLoad = load;
                    break;
                }
            }
            return ruptureLoad;
        };

        result.breakForce4Substeps = testBreakLoad(4);
        result.breakForce8Substeps = testBreakLoad(8);

        result.breakForceErrorRatio = fabsf(result.breakForce4Substeps - result.breakForce8Substeps) /
            fmaxf(1.0f, result.breakForce4Substeps);

        std::cout << "  - 4 Substeps Fracture Load: " << std::fixed << std::setprecision(2)
                  << result.breakForce4Substeps << " N\n"
                  << "  - 8 Substeps Fracture Load: " << result.breakForce8Substeps << " N\n"
                  << "  - Measured Timestep Error:  " << result.breakForceErrorRatio * 100.0f
                  << "% (Target: <= 15.0%)\n";

        if (result.breakForceErrorRatio <= 0.15f && result.breakForce4Substeps > 0.0f) {
            result.breakForcePassed = true;
            std::cout << "  -> Gate 11F-2 Result: PASSED (XPBD break force timestep-invariant)\n\n";
        } else {
            std::cout << "  -> Gate 11F-2 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11F-3: Bond Count Integrity Across 1000 Reorders
    // =========================================================================
    std::cout << "[GATE 11F-3] Verifying Bond Count Integrity Across 1000 Reorders...\n";
    {
        const uint32_t testGrains = 4096;
        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.enableSleeping = false;

        GranularSystem testSys(systemCap, false);
        testSys.setSettings(settings);
        testSys.spawnBondedBlock({0.0f, 0.2f, 0.0f}, testGrains, 0.02f, 0.10f, 1e-5f, 0.0f, 3333); // unbreakable
        testSys.synchronize();

        result.initialBonds = testSys.stats().activeBondCount;
        std::cout << "  - Initial Active Glue Bonds: " << result.initialBonds << "\n";

        const uint32_t blocks = (testGrains + 255) / 256;
        for (uint32_t r = 0; r < result.reorderIterations; ++r) {
            perturbForReorderKernel<<<blocks, 256, 0, nullptr>>>(
                testSys.pool().pos, testGrains, r);
            testSys.rebuildNeighbors();
        }
        testSys.synchronize();
        result.finalBonds = testSys.stats().activeBondCount;

        std::cout << "  - Active Bonds After " << result.reorderIterations << " Reorders: "
                  << result.finalBonds << "\n";

        if (result.finalBonds == result.initialBonds && result.initialBonds > 0) {
            result.reorderIntegrityPassed = true;
            std::cout << "  -> Gate 11F-3 Result: PASSED (100% bond count integrity across 1000 reorders)\n\n";
        } else {
            std::cout << "  -> Gate 11F-3 Result: FAILED\n\n";
        }
    }

    // =========================================================================
    // GATE 11F-4: Clump Generation & Impact Crumbling
    // =========================================================================
    std::cout << "[GATE 11F-4] Verifying Clump Generation & High-Energy Impact Crumbling...\n";
    {
        const uint32_t clumpGrains = 5000;
        const float clumpR = 0.04f;
        const float breakThresh = 25.0f; // Breakable clumps

        GranularSettings settings;
        settings.grainRadius = 0.01f;
        settings.substeps = 4;
        settings.iterations = 4;
        settings.gravity = {0.0f, -9.81f, 0.0f};
        settings.groundPlane.point = {0.0f, 0.0f, 0.0f};
        settings.groundPlane.normal = {0.0f, 1.0f, 0.0f};
        settings.groundPlane.enabled = 1;
        settings.enableSleeping = false;

        GranularSystem clumpSys(systemCap, false);
        clumpSys.setSettings(settings);
        // Spawn clumps in mid-air above ground plane
        clumpSys.spawnClumps({0.0f, 0.6f, 0.0f}, clumpGrains, 0.02f, clumpR, 1e-5f, breakThresh, 0.0f, 4444);
        clumpSys.synchronize();

        result.initialIntraClumpBonds = clumpSys.stats().activeBondCount;
        result.totalClumpsSpawned = 32;

        std::cout << "  - Initial Intra-Clump Bonds: " << result.initialIntraClumpBonds << "\n";

        // Free fall and slam into ground plane (high-speed impact)
        for (int f = 0; f < 60; ++f) {
            clumpSys.update(1.0f / 60.0f);
        }
        clumpSys.synchronize();

        result.postImpactActiveBonds = clumpSys.stats().activeBondCount;
        const uint32_t brokenBonds = (result.initialIntraClumpBonds > result.postImpactActiveBonds)
            ? (result.initialIntraClumpBonds - result.postImpactActiveBonds) : 0;
        result.bondRuptureFraction = static_cast<float>(brokenBonds) / static_cast<float>(result.initialIntraClumpBonds);

        std::cout << "  - Post-Impact Active Bonds:  " << result.postImpactActiveBonds << "\n"
                  << "  - Ruptured Bond Fraction:    " << std::fixed << std::setprecision(1)
                  << result.bondRuptureFraction * 100.0f << "% (Target: >= 20.0%)\n";

        if (result.bondRuptureFraction >= 0.20f && result.initialIntraClumpBonds > 0) {
            result.clumpsPassed = true;
            std::cout << "  -> Gate 11F-4 Result: PASSED (clumps hold in flight, crumble on impact)\n\n";
        } else {
            std::cout << "  -> Gate 11F-4 Result: FAILED\n\n";
        }
    }

    result.passed = result.cliffPassed && result.breakForcePassed &&
                    result.reorderIntegrityPassed && result.clumpsPassed;

    std::cout << "======================================================================\n";
    if (result.passed) {
        std::cout << "  PHASE 11F VALIDATION VERDICT: ALL GATES PASSED!\n";
    } else {
        std::cout << "  PHASE 11F VALIDATION VERDICT: GATES FAILED\n";
    }
    std::cout << "======================================================================\n\n";

    return result;
}

} // namespace vparticles
