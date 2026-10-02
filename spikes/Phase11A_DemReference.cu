// Phase 11A only: deliberately simple DEM calibration reference.
//
// This is not production code. It uses an O(N^2) all-pairs gather kernel so
// that the contact law is easy to audit. It must stay out of VParticlesCore
// and is built only via the explicitly requested Phase11ADemReference target.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 128;

struct DemParams {
    uint32_t grainCount;
    float dt;
    float radius;
    float stiffness;
    float normalDamping;
    float friction;
    float tangentialDamping;
    float linearDamping;
    float gravity;
};

struct DemConfig {
    uint32_t grainCount = 5'000;
    uint32_t frames = 180;
    uint32_t substeps = 8;
    uint32_t reportEvery = 30;
    uint32_t seed = 11'011;
    float frameDt = 1.0f / 60.0f;
    float radius = 0.025f;
    float stiffness = 2'500.0f;
    float normalDamping = 8.0f;
    float friction = 0.45f;
    float tangentialDamping = 8.0f;
    float linearDamping = 2.0f;
    std::string csvPath;
};

struct Summary {
    float meanHeight = 0.0f;
    float maxHeight = 0.0f;
    float kineticEnergy = 0.0f;
    float maxOverlap = 0.0f;
    float stepMs = 0.0f;
};

void check(cudaError_t result, const char* expression)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(result));
    }
}

#define DEM_CUDA_CHECK(expression) check((expression), #expression)

__device__ inline float3 makeFloat3(const float4 value)
{
    return make_float3(value.x, value.y, value.z);
}

__device__ inline float dot3(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 add3(const float3 a, const float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 sub3(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 mul3(const float3 value, const float scalar)
{
    return make_float3(value.x * scalar, value.y * scalar, value.z * scalar);
}

// A gather formulation: each thread writes exactly one grain. Pair forces are
// evaluated twice, once from each grain's perspective, avoiding atomics and
// retaining deterministic force accumulation order for a fixed launch shape.
__global__ void demStepKernel(
    const float4* __restrict__ currentPos,
    const float4* __restrict__ currentVel,
    float4* __restrict__ nextPos,
    float4* __restrict__ nextVel,
    DemParams params)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= params.grainCount) {
        return;
    }

    const float4 pos4 = currentPos[i];
    const float4 vel4 = currentVel[i];
    const float3 position = makeFloat3(pos4);
    const float3 velocity = makeFloat3(vel4);
    // The linear term is intentionally explicit: this spike models a damped
    // reference pile, not a lossless ballistic benchmark. Contact damping
    // remains the calibration knob for restitution.
    float3 force = mul3(velocity, -params.linearDamping);
    force.y -= params.gravity; // unit mass
    float maxOverlap = 0.0f;
    const float contactDistance = params.radius * 2.0f;
    const float contactDistanceSq = contactDistance * contactDistance;

    for (uint32_t j = 0; j < params.grainCount; ++j) {
        if (j == i) {
            continue;
        }

        const float3 otherPosition = makeFloat3(currentPos[j]);
        const float3 displacement = sub3(otherPosition, position);
        const float distanceSq = dot3(displacement, displacement);
        if (distanceSq >= contactDistanceSq || distanceSq <= 1.0e-12f) {
            continue;
        }

        const float distance = sqrtf(distanceSq);
        const float3 normal = mul3(displacement, 1.0f / distance);
        const float overlap = contactDistance - distance;
        maxOverlap = fmaxf(maxOverlap, overlap);
        const float3 relativeVelocity = sub3(velocity, makeFloat3(currentVel[j]));
        const float normalVelocity = dot3(relativeVelocity, normal);
        const float normalForce = fmaxf(0.0f,
            params.stiffness * overlap + params.normalDamping * normalVelocity);
        force = sub3(force, mul3(normal, normalForce));

        const float3 tangentialVelocity = sub3(relativeVelocity, mul3(normal, normalVelocity));
        const float tangentialSpeedSq = dot3(tangentialVelocity, tangentialVelocity);
        if (tangentialSpeedSq > 1.0e-12f) {
            const float tangentialSpeed = sqrtf(tangentialSpeedSq);
            const float tangentialForce = fminf(
                params.friction * normalForce,
                params.tangentialDamping * tangentialSpeed);
            force = sub3(force, mul3(tangentialVelocity, tangentialForce / tangentialSpeed));
        }
    }

    // Plane contact at y = 0. This mirrors the same spring/dashpot/Coulomb
    // model and supplies the settling reference for the first calibration run.
    const float groundOverlap = params.radius - position.y;
    if (groundOverlap > 0.0f) {
        maxOverlap = fmaxf(maxOverlap, groundOverlap);
        const float normalForce = fmaxf(0.0f,
            params.stiffness * groundOverlap - params.normalDamping * velocity.y);
        force.y += normalForce;
        const float tangentSpeed = sqrtf(velocity.x * velocity.x + velocity.z * velocity.z);
        if (tangentSpeed > 1.0e-6f) {
            const float tangentForce = fminf(params.friction * normalForce,
                params.tangentialDamping * tangentSpeed);
            force.x -= tangentForce * velocity.x / tangentSpeed;
            force.z -= tangentForce * velocity.z / tangentSpeed;
        }
    }

    const float3 nextVelocity = add3(velocity, mul3(force, params.dt));
    const float3 nextPosition = add3(position, mul3(nextVelocity, params.dt));
    nextPos[i] = make_float4(nextPosition.x, nextPosition.y, nextPosition.z, maxOverlap);
    nextVel[i] = make_float4(nextVelocity.x, nextVelocity.y, nextVelocity.z, 0.0f);
}

__global__ void reduceSummaryKernel(
    const float4* __restrict__ positions,
    const float4* __restrict__ velocities,
    uint32_t grainCount,
    float* __restrict__ sums)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= grainCount) {
        return;
    }

    const float4 position = positions[i];
    const float4 velocity = velocities[i];
    atomicAdd(&sums[0], position.y);
    atomicMax(reinterpret_cast<int*>(&sums[1]), __float_as_int(position.y));
    atomicAdd(&sums[2], 0.5f * (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z));
    atomicMax(reinterpret_cast<int*>(&sums[3]), __float_as_int(position.w));
}

uint32_t parseUint(const char* value, uint32_t fallback)
{
    try {
        return static_cast<uint32_t>(std::stoul(value));
    } catch (...) {
        return fallback;
    }
}

float parseFloat(const char* value, float fallback)
{
    try {
        return std::stof(value);
    } catch (...) {
        return fallback;
    }
}

DemConfig parseArgs(int argc, char** argv)
{
    DemConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--grains" && i + 1 < argc) config.grainCount = parseUint(argv[++i], config.grainCount);
        else if (argument == "--frames" && i + 1 < argc) config.frames = parseUint(argv[++i], config.frames);
        else if (argument == "--substeps" && i + 1 < argc) config.substeps = parseUint(argv[++i], config.substeps);
        else if (argument == "--report-every" && i + 1 < argc) config.reportEvery = parseUint(argv[++i], config.reportEvery);
        else if (argument == "--seed" && i + 1 < argc) config.seed = parseUint(argv[++i], config.seed);
        else if (argument == "--radius" && i + 1 < argc) config.radius = parseFloat(argv[++i], config.radius);
        else if (argument == "--stiffness" && i + 1 < argc) config.stiffness = parseFloat(argv[++i], config.stiffness);
        else if (argument == "--damping" && i + 1 < argc) config.normalDamping = parseFloat(argv[++i], config.normalDamping);
        else if (argument == "--friction" && i + 1 < argc) config.friction = parseFloat(argv[++i], config.friction);
        else if (argument == "--tangential-damping" && i + 1 < argc) config.tangentialDamping = parseFloat(argv[++i], config.tangentialDamping);
        else if (argument == "--linear-damping" && i + 1 < argc) config.linearDamping = parseFloat(argv[++i], config.linearDamping);
        else if (argument == "--csv" && i + 1 < argc) config.csvPath = argv[++i];
        else if (argument == "--help") {
            std::cout << "Phase11ADemReference options: --grains N --frames N --substeps N "
                         "--radius R --stiffness K --damping C --friction MU "
                         "--tangential-damping C --linear-damping C --seed N --report-every N --csv FILE\n";
            std::exit(0);
        }
    }
    if (config.grainCount == 0 || config.substeps == 0 || config.radius <= 0.0f ||
        config.stiffness <= 0.0f || config.frameDt <= 0.0f) {
        throw std::runtime_error("All DEM scale, timestep, and contact parameters must be positive");
    }
    return config;
}

std::vector<float4> makeInitialPositions(const DemConfig& config)
{
    std::vector<float4> positions(config.grainCount);
    std::mt19937 random(config.seed);
    std::uniform_real_distribution<float> jitter(-0.07f, 0.07f);
    const float spacing = 2.05f * config.radius;
    const uint32_t side = static_cast<uint32_t>(std::ceil(std::cbrt(static_cast<float>(config.grainCount))));
    for (uint32_t i = 0; i < config.grainCount; ++i) {
        const uint32_t x = i % side;
        const uint32_t z = (i / side) % side;
        const uint32_t y = i / (side * side);
        positions[i] = make_float4(
            (static_cast<float>(x) - 0.5f * static_cast<float>(side - 1)) * spacing + jitter(random) * config.radius,
            config.radius + 0.30f + static_cast<float>(y) * spacing + jitter(random) * config.radius,
            (static_cast<float>(z) - 0.5f * static_cast<float>(side - 1)) * spacing + jitter(random) * config.radius,
            0.0f);
    }
    return positions;
}

Summary collectSummary(
    const float4* positions,
    const float4* velocities,
    uint32_t grainCount,
    cudaStream_t stream)
{
    float* deviceSums = nullptr;
    DEM_CUDA_CHECK(cudaMalloc(&deviceSums, sizeof(float) * 4));
    DEM_CUDA_CHECK(cudaMemsetAsync(deviceSums, 0, sizeof(float) * 4, stream));
    reduceSummaryKernel<<<(grainCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        positions, velocities, grainCount, deviceSums);
    DEM_CUDA_CHECK(cudaGetLastError());
    float sums[4] = {};
    DEM_CUDA_CHECK(cudaMemcpyAsync(sums, deviceSums, sizeof(sums), cudaMemcpyDeviceToHost, stream));
    DEM_CUDA_CHECK(cudaStreamSynchronize(stream));
    DEM_CUDA_CHECK(cudaFree(deviceSums));
    return Summary{ sums[0] / static_cast<float>(grainCount), sums[1], sums[2], sums[3], 0.0f };
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const DemConfig config = parseArgs(argc, argv);
        DemParams params = {
            config.grainCount,
            config.frameDt / static_cast<float>(config.substeps),
            config.radius,
            config.stiffness,
            config.normalDamping,
            config.friction,
            config.tangentialDamping,
            config.linearDamping,
            9.81f,
        };

        std::cout << "Phase 11A DEM reference (throwaway calibration spike)\n"
                  << "grains=" << config.grainCount << " frames=" << config.frames
                  << " substeps=" << config.substeps << " radius=" << config.radius
                  << " stiffness=" << config.stiffness << " damping=" << config.normalDamping
                  << " friction=" << config.friction
                  << " tangentDamping=" << config.tangentialDamping
                  << " linearDamping=" << config.linearDamping
                  << " seed=" << config.seed << "\n";

        const std::vector<float4> initialPositions = makeInitialPositions(config);
        std::vector<float4> initialVelocities(config.grainCount, make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        float4 *positionsA = nullptr, *positionsB = nullptr, *velocitiesA = nullptr, *velocitiesB = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr, stop = nullptr;
        DEM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DEM_CUDA_CHECK(cudaEventCreate(&start));
        DEM_CUDA_CHECK(cudaEventCreate(&stop));
        DEM_CUDA_CHECK(cudaMalloc(&positionsA, sizeof(float4) * config.grainCount));
        DEM_CUDA_CHECK(cudaMalloc(&positionsB, sizeof(float4) * config.grainCount));
        DEM_CUDA_CHECK(cudaMalloc(&velocitiesA, sizeof(float4) * config.grainCount));
        DEM_CUDA_CHECK(cudaMalloc(&velocitiesB, sizeof(float4) * config.grainCount));
        DEM_CUDA_CHECK(cudaMemcpyAsync(positionsA, initialPositions.data(), sizeof(float4) * config.grainCount, cudaMemcpyHostToDevice, stream));
        DEM_CUDA_CHECK(cudaMemcpyAsync(velocitiesA, initialVelocities.data(), sizeof(float4) * config.grainCount, cudaMemcpyHostToDevice, stream));

        std::ofstream csv;
        if (!config.csvPath.empty()) {
            csv.open(config.csvPath);
            if (!csv) throw std::runtime_error("Unable to open CSV path");
            csv << "frame,meanHeight,maxHeight,kineticEnergy,maxOverlap,stepMs\n";
        }

        float accumulatedStepMs = 0.0f;
        for (uint32_t frame = 0; frame < config.frames; ++frame) {
            DEM_CUDA_CHECK(cudaEventRecord(start, stream));
            for (uint32_t substep = 0; substep < config.substeps; ++substep) {
                demStepKernel<<<(config.grainCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    positionsA, velocitiesA, positionsB, velocitiesB, params);
                DEM_CUDA_CHECK(cudaGetLastError());
                std::swap(positionsA, positionsB);
                std::swap(velocitiesA, velocitiesB);
            }
            DEM_CUDA_CHECK(cudaEventRecord(stop, stream));
            DEM_CUDA_CHECK(cudaEventSynchronize(stop));
            float frameMs = 0.0f;
            DEM_CUDA_CHECK(cudaEventElapsedTime(&frameMs, start, stop));
            accumulatedStepMs += frameMs / static_cast<float>(config.substeps);

            if (frame % config.reportEvery == 0 || frame + 1 == config.frames) {
                Summary summary = collectSummary(positionsA, velocitiesA, config.grainCount, stream);
                summary.stepMs = frameMs / static_cast<float>(config.substeps);
                std::cout << "frame=" << std::setw(4) << frame
                          << " meanHeight=" << std::fixed << std::setprecision(4) << summary.meanHeight
                          << " maxHeight=" << summary.maxHeight
                          << " kineticEnergy=" << summary.kineticEnergy
                          << " maxOverlap=" << summary.maxOverlap
                          << " stepMs=" << summary.stepMs << '\n';
                if (csv) {
                    csv << frame << ',' << summary.meanHeight << ',' << summary.maxHeight << ','
                        << summary.kineticEnergy << ',' << summary.maxOverlap << ',' << summary.stepMs << '\n';
                }
            }
        }

        std::cout << "averageStepMs=" << std::fixed << std::setprecision(4)
                  << accumulatedStepMs / static_cast<float>(config.frames) << '\n';
        DEM_CUDA_CHECK(cudaFree(positionsA));
        DEM_CUDA_CHECK(cudaFree(positionsB));
        DEM_CUDA_CHECK(cudaFree(velocitiesA));
        DEM_CUDA_CHECK(cudaFree(velocitiesB));
        DEM_CUDA_CHECK(cudaEventDestroy(start));
        DEM_CUDA_CHECK(cudaEventDestroy(stop));
        DEM_CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase11ADemReference failed: " << error.what() << '\n';
        return 1;
    }
}
