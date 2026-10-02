#include "VParticles/ParticleSystem.h"

#include "VParticles/CudaCheck.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ── GPU device info ──────────────────────────────────────────────────

struct GpuDeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    int smCount = 0;
    size_t totalMemoryMB = 0;
    int memoryClockMHz = 0;
    int memoryBusWidth = 0;
    int driverVersion = 0;
    int runtimeVersion = 0;
};

GpuDeviceInfo queryGpuDeviceInfo()
{
    int device = 0;
    VP_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props = {};
    VP_CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    GpuDeviceInfo info;
    info.name = props.name;
    info.major = props.major;
    info.minor = props.minor;
    info.smCount = props.multiProcessorCount;
    info.totalMemoryMB = static_cast<size_t>(props.totalGlobalMem / (1024u * 1024u));
    info.memoryBusWidth = props.memoryBusWidth;

    int memoryClockKHz = 0;
    cudaDeviceGetAttribute(&memoryClockKHz, cudaDevAttrMemoryClockRate, device);
    info.memoryClockMHz = memoryClockKHz / 1000;

    VP_CUDA_CHECK(cudaDriverGetVersion(&info.driverVersion));
    VP_CUDA_CHECK(cudaRuntimeGetVersion(&info.runtimeVersion));

    return info;
}

void printGpuDeviceInfo(const GpuDeviceInfo& info)
{
    std::cout << "GPU: " << info.name
              << "  SM " << info.major << '.' << info.minor
              << "  " << info.smCount << " SMs"
              << "  " << info.totalMemoryMB << " MB VRAM"
              << "  " << info.memoryBusWidth << "-bit bus"
              << "  " << info.memoryClockMHz << " MHz mem\n"
              << "CUDA driver " << info.driverVersion / 1000
              << '.' << (info.driverVersion % 1000) / 10
              << "  runtime " << info.runtimeVersion / 1000
              << '.' << (info.runtimeVersion % 1000) / 10
              << "\n\n";
}

// ── Workload types ───────────────────────────────────────────────────

enum class Workload { Ramp, Recycle, Saturation, Burst };

const char* workloadName(Workload workload)
{
    switch (workload) {
    case Workload::Ramp:       return "ramp";
    case Workload::Recycle:    return "recycle";
    case Workload::Saturation: return "saturation";
    case Workload::Burst:      return "burst";
    }
    return "unknown";
}

Workload parseWorkload(const char* value, Workload fallback)
{
    if (value == nullptr) {
        return fallback;
    }
    const std::string text(value);
    if (text == "ramp")       return Workload::Ramp;
    if (text == "recycle")    return Workload::Recycle;
    if (text == "saturation") return Workload::Saturation;
    if (text == "burst")      return Workload::Burst;
    return fallback;
}

std::vector<Workload> parseWorkloadList(
    const char* value,
    const std::vector<Workload>& fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    std::vector<Workload> result;
    const std::string text(value);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string token = comma == std::string::npos
            ? text.substr(start)
            : text.substr(start, comma - start);
        if (token == "ramp")            result.push_back(Workload::Ramp);
        else if (token == "recycle")    result.push_back(Workload::Recycle);
        else if (token == "saturation") result.push_back(Workload::Saturation);
        else if (token == "burst")      result.push_back(Workload::Burst);
        else return fallback;

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return result.empty() ? fallback : result;
}

std::string joinWorkloadList(const std::vector<Workload>& workloads)
{
    std::string joined;
    for (size_t i = 0; i < workloads.size(); ++i) {
        if (i != 0) {
            joined += ',';
        }
        joined += workloadName(workloads[i]);
    }
    return joined;
}

// ── Percentiles and bandwidth ────────────────────────────────────────

struct TimingPercentiles {
    float p50 = 0.0f;
    float p95 = 0.0f;
    float p99 = 0.0f;
};

float percentile(const std::vector<float>& sorted, float p)
{
    if (sorted.empty()) {
        return 0.0f;
    }
    const float index = p * static_cast<float>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(index);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const float fraction = index - static_cast<float>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

TimingPercentiles computePercentiles(std::vector<float>& timings)
{
    if (timings.empty()) {
        return {};
    }
    std::sort(timings.begin(), timings.end());
    return {
        percentile(timings, 0.50f),
        percentile(timings, 0.95f),
        percentile(timings, 0.99f),
    };
}

// Effective simulation bandwidth based on per-particle memory traffic.
// FP32 mode:
//   read:  float4 pos(16) + float4 vel(16) + float4 color(16)
//          + age(4) + lifetime(4) + activeIndex(4) = 60
//   write: float4 pos(16) + float4 vel(16) + float4 color(16)
//          + age(4) + aliveFlag(1) = 53
//   Total: 113 bytes/particle
// Packed mode:
//   read:  posXYZ(6) + velXYZ(6) + age(2) + lifetime(2) + color(4)
//          + size(2) + activeIndex(4) = 26
//   write: posXYZ(6) + velXYZ(6) + age(2) + color(4) + size(2)
//          + aliveFlag(1) = 21
//   Total: ~57 bytes/particle (with padding/alignment ~57)
constexpr uint32_t kSimulateBytesPerParticleFP32 = 113;
constexpr uint32_t kSimulateBytesPerParticlePacked = 57;

float simulateBandwidthGBs(uint32_t aliveCount, float simulateMs, uint32_t bytesPerParticle)
{
    if (simulateMs <= 0.0f || aliveCount == 0) {
        return 0.0f;
    }
    const double bytes = static_cast<double>(aliveCount) * bytesPerParticle;
    const double seconds = static_cast<double>(simulateMs) * 1.0e-3;
    return static_cast<float>(bytes / seconds / 1.0e9);
}

// Estimated GPU memory for the pool and supporting buffers.
// FP32 per slot: pos(16) + vel(16) + color(16) + age(4) + lifetime(4)
//              + systemId(4) + tileId(2) + activeIndices*2(8) + freeIndices(4)
//              + aliveFlags(1) = 75 bytes.
// Packed per slot: posXYZ(6) + velXYZ(6) + age(2) + lifetime(2) + color(4)
//                + size(2) + systemId(2) + tileId(2) + activeIndices*2(8) + freeIndices(4)
//                + aliveFlags(1) = 39 bytes.
constexpr uint32_t kPoolBytesPerSlotFP32 = 75;
constexpr uint32_t kPoolBytesPerSlotPacked = 39;

float estimatedPoolMemoryMB(uint32_t capacity, bool packed)
{
    const uint32_t bytesPerSlot = packed ? kPoolBytesPerSlotPacked : kPoolBytesPerSlotFP32;
    return static_cast<float>(
        static_cast<double>(capacity) * bytesPerSlot / (1024.0 * 1024.0));
}

// ── Benchmark config and result ──────────────────────────────────────

struct BenchmarkConfig {
    uint32_t capacity = 1'000'000;
    uint32_t frames = 240;
    uint32_t emitterCount = 1;
    float spawnRate = 250'000.0f;
    float dt = 1.0f / 60.0f;
    Workload workload = Workload::Ramp;
    float turbulenceStrength = 0.0f;
    float turbulenceFrequency = 0.2f;
    bool enableGroundPlane = false;
    float groundPlaneHeight = 0.0f;
    bool enableSphereCollision = false;
    float sphereRadius = 5.0f;
    bool enableBoxCollision = false;
    bool enableCurves = false;
    bool usePacked = false;
    bool useGraph = true;
    uint32_t tileCount = 1;
    float tileScale = 100.0f;
    uint32_t recipeCount = 1;
};

struct BenchmarkResult {
    BenchmarkConfig config = {};
    vparticles::SimulationStats finalStats = {};
    std::vector<vparticles::TileStats> tileStats = {};
    uint32_t telemetrySamples = 0;
    uint32_t maxTelemetryLatencyFrames = 0;
    TimingPercentiles spawnPercentiles = {};
    TimingPercentiles simulatePercentiles = {};
    TimingPercentiles compactPercentiles = {};
    TimingPercentiles totalPercentiles = {};
    float peakBandwidthGBs = 0.0f;
    float medianBandwidthGBs = 0.0f;
};

// ── Argument parsing ─────────────────────────────────────────────────

uint32_t parseUintArg(const char* value, uint32_t fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || parsed > std::numeric_limits<uint32_t>::max()) {
        return fallback;
    }
    return static_cast<uint32_t>(parsed);
}

float parseFloatArg(const char* value, float fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    return end != value ? parsed : fallback;
}

std::vector<uint32_t> parseUintListArg(
    const char* value,
    const std::vector<uint32_t>& fallback)
{
    if (value == nullptr) {
        return fallback;
    }

    std::vector<uint32_t> parsedValues;
    const std::string text(value);
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string token = comma == std::string::npos
            ? text.substr(start)
            : text.substr(start, comma - start);
        if (token.empty()) {
            return fallback;
        }

        char* end = nullptr;
        const unsigned long parsed = std::strtoul(token.c_str(), &end, 10);
        if (end == token.c_str() ||
            *end != '\0' ||
            parsed == 0 ||
            parsed > std::numeric_limits<uint32_t>::max()) {
            return fallback;
        }
        parsedValues.push_back(static_cast<uint32_t>(parsed));

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    return parsedValues.empty() ? fallback : parsedValues;
}

std::string joinUintList(const std::vector<uint32_t>& values)
{
    std::string joined;
    for (uint32_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            joined += ',';
        }
        joined += std::to_string(values[index]);
    }
    return joined;
}

// ── Spawn rate helpers ───────────────────────────────────────────────

float scaledRampSpawnRate(uint32_t capacity, uint32_t frames, float dt)
{
    const double duration = static_cast<double>(frames) * static_cast<double>(dt);
    if (duration <= 0.0) {
        return 0.0f;
    }

    const double spawnRate = static_cast<double>(capacity) / duration;
    return static_cast<float>(std::min<double>(
        spawnRate,
        static_cast<double>(std::numeric_limits<float>::max())));
}

// Compute the natural spawn rate for each workload type when the user
// has not specified an explicit --spawn-rate.
float workloadSpawnRate(Workload workload, uint32_t capacity, uint32_t frames, float dt)
{
    switch (workload) {
    case Workload::Ramp:
        return scaledRampSpawnRate(capacity, frames, dt);
    case Workload::Recycle:
        // Fill in ~1 second, then sustain churn at capacity.
        return static_cast<float>(capacity);
    case Workload::Saturation:
        // 4x oversupply: pool fills in ~25% of run, rest is all drops.
        return scaledRampSpawnRate(capacity, frames, dt) * 4.0f;
    case Workload::Burst:
        // No continuous spawning; bursts are queued separately.
        return 0.0f;
    }
    return scaledRampSpawnRate(capacity, frames, dt);
}

// ── Recipe setup ─────────────────────────────────────────────────────

void setupRecipes(vparticles::SimulationSettings& settings, const BenchmarkConfig& config)
{
    for (uint32_t r = 0; r < config.recipeCount && r < vparticles::kMaxRecipes; ++r) {
        vparticles::EffectRecipe recipe = {};
        switch (r % 8) {
        case 0: // Smoke / Dust: gentle rise, drag, turbulence
            recipe.gravity = {0.0f, 1.5f, 0.0f};
            recipe.drag = 0.25f;
            recipe.turbulence = {3.5f, 0.25f, 0.6f};
            recipe.curves = {0.5f, 3.0f, 5.0f, {0.7f, 0.7f, 0.7f, 0.6f}, {0.2f, 0.2f, 0.2f, 0.0f}, 1};
            break;
        case 1: // Sparks / Debris: heavy gravity, low drag, bouncy ground plane
            recipe.gravity = {0.0f, -18.0f, 0.0f};
            recipe.drag = 0.02f;
            recipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.85f, 0.1f, 1};
            recipe.curves = {1.0f, 0.8f, 0.1f, {1.0f, 0.9f, 0.3f, 1.0f}, {0.8f, 0.2f, 0.05f, 0.0f}, 1};
            break;
        case 2: // Fire / Embers: fast rise, turbulence, obstacle sphere
            recipe.gravity = {0.0f, 4.0f, 0.0f};
            recipe.drag = 0.1f;
            recipe.turbulence = {2.5f, 0.4f, 1.0f};
            recipe.sphere = {{0.0f, 8.0f, 0.0f}, 4.0f, 0.5f, 0.2f, 1, 0};
            recipe.curves = {0.8f, 2.2f, 0.2f, {1.0f, 0.6f, 0.1f, 1.0f}, {0.2f, 0.02f, 0.01f, 0.0f}, 1};
            break;
        case 3: // Water Fountain / Spray: gravity, wind, box bounds
            recipe.gravity = {0.0f, -9.81f, 0.0f};
            recipe.wind = {4.0f, 0.0f, 1.0f};
            recipe.drag = 0.06f;
            recipe.box = {{-15.0f, 0.0f, -15.0f}, {15.0f, 25.0f, 15.0f}, 0.5f, 0.2f, 1};
            recipe.curves = {0.4f, 1.2f, 0.3f, {0.2f, 0.6f, 1.0f, 0.9f}, {0.8f, 0.9f, 1.0f, 0.0f}, 1};
            break;
        case 4: // Plasma / Vortex: zero-G, high turbulence, containment sphere
            recipe.gravity = {0.0f, 0.0f, 0.0f};
            recipe.drag = 0.03f;
            recipe.turbulence = {5.0f, 0.5f, 1.2f};
            recipe.sphere = {{0.0f, 5.0f, 0.0f}, 12.0f, 0.85f, 0.05f, 1, 1};
            recipe.curves = {1.0f, 2.5f, 0.4f, {0.8f, 0.1f, 1.0f, 1.0f}, {0.1f, 0.0f, 0.4f, 0.0f}, 1};
            break;
        case 5: // Shrapnel: extreme gravity, fast ground plane bounce
            recipe.gravity = {0.0f, -28.0f, 0.0f};
            recipe.drag = 0.01f;
            recipe.plane = {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.7f, 0.3f, 1};
            recipe.curves = {1.2f, 1.0f, 0.4f, {0.5f, 0.5f, 0.55f, 1.0f}, {0.2f, 0.2f, 0.2f, 0.5f}, 1};
            break;
        case 6: // Magic Shimmer: gentle descent, fine turbulence
            recipe.gravity = {0.0f, -1.0f, 0.0f};
            recipe.drag = 0.12f;
            recipe.turbulence = {2.0f, 0.8f, 0.8f};
            recipe.curves = {0.3f, 1.0f, 0.1f, {0.2f, 1.0f, 0.8f, 0.8f}, {0.05f, 0.3f, 0.2f, 0.0f}, 1};
            break;
        case 7: // Firework Burst: wind drift, fast fade curves
            recipe.gravity = {0.0f, -6.0f, 0.0f};
            recipe.wind = {-2.0f, 0.0f, 2.0f};
            recipe.drag = 0.05f;
            recipe.curves = {0.6f, 1.8f, 0.2f, {1.0f, 0.2f, 0.4f, 1.0f}, {0.3f, 0.05f, 0.1f, 0.0f}, 1};
            break;
        }
        settings.recipes[r] = recipe;
    }
}

// ── Emitter setup ────────────────────────────────────────────────────

void addDefaultEmitters(vparticles::ParticleSystem& system, const BenchmarkConfig& config)
{
    float lifetime = 4.0f;
    float lifetimeVariance = 0.2f;

    switch (config.workload) {
    case Workload::Ramp:
        // Defaults: lifetime = 4s, particles outlive most ramp runs.
        break;
    case Workload::Recycle:
        // Short lifetime for sustained churn once pool is full.
        lifetime = 1.0f;
        lifetimeVariance = 0.1f;
        break;
    case Workload::Saturation:
        // Long lifetime: nothing dies, every spawn after capacity is a drop.
        lifetime = static_cast<float>(config.frames) * config.dt * 2.0f;
        lifetimeVariance = 0.0f;
        break;
    case Workload::Burst:
        // Moderate lifetime: particles from earlier bursts recycle.
        lifetime = 2.0f;
        lifetimeVariance = 0.2f;
        break;
    }

    const float perEmitterRate = config.spawnRate / static_cast<float>(config.emitterCount);

    for (uint32_t i = 0; i < config.emitterCount; ++i) {
        vparticles::EmitterDesc emitter;
        emitter.spawnRate = perEmitterRate;
        emitter.lifetime = lifetime;
        emitter.lifetimeVariance = lifetimeVariance;
        emitter.velocity = {0.0f, 8.0f, 0.0f};
        emitter.velocityVariance = {2.0f, 2.0f, 2.0f};
        if (config.recipeCount > 1) {
            emitter.recipeId = static_cast<uint16_t>(i % config.recipeCount);
        }
        if (config.tileCount > 1) {
            emitter.tileId = static_cast<uint16_t>(i % config.tileCount);
            const auto& t = system.settings().tiles[emitter.tileId];
            emitter.position = {t.originX, t.originY, t.originZ};
        }
        system.addEmitter(emitter);
    }
}

// ── Per-frame stats printing ─────────────────────────────────────────

void printStats(const vparticles::SimulationStats& stats, uint32_t bytesPerParticle)
{
    const float bw = simulateBandwidthGBs(stats.aliveCount, stats.simulateMs, bytesPerParticle);
    std::cout << "frame " << std::setw(4) << stats.frameIndex
              << " alive=" << std::setw(9) << stats.aliveCount
              << " spawned=" << std::setw(7) << stats.spawned
              << " dropped=" << std::setw(7) << stats.dropped
              << " dead=" << std::setw(7) << stats.deadCount
              << std::fixed << std::setprecision(3)
              << " spawnMs=" << stats.spawnMs
              << " simMs=" << stats.simulateMs
              << " compactMs=" << stats.compactMs
              << " totalMs=" << stats.totalMs
              << " submit=" << (stats.graphActive ? "graph" : "eager");
    if (bw > 0.0f) {
        std::cout << std::setprecision(1) << " BW=" << bw;
    }
    if (stats.telemetryLatencyFrames != 0) {
        std::cout << " lag=" << stats.telemetryLatencyFrames;
    }
    std::cout << '\n';
}

// ── Benchmark runner ─────────────────────────────────────────────────

constexpr uint32_t kBurstIntervalFrames = 30;

BenchmarkResult runBenchmark(const BenchmarkConfig& config, uint32_t sampleEvery)
{
    const vparticles::StorageMode mode = config.usePacked
        ? vparticles::StorageMode::Packed
        : vparticles::StorageMode::FP32;
    const uint32_t bytesPerParticle = config.usePacked
        ? kSimulateBytesPerParticlePacked
        : kSimulateBytesPerParticleFP32;
    vparticles::ParticleSystem system(config.capacity, mode, config.useGraph);

    vparticles::SimulationSettings settings = system.settings();
    settings.tileCount = config.tileCount;
    settings.posQuantizationScale = config.tileScale;
    settings.recipeCount = config.recipeCount;
    if (config.tileCount > 1) {
        const uint32_t gridDim = static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(config.tileCount))));
        const float spacing = 150.0f;
        for (uint32_t t = 0; t < config.tileCount; ++t) {
            const uint32_t row = t / gridDim;
            const uint32_t col = t % gridDim;
            settings.tiles[t].originX = (static_cast<float>(col) - 0.5f * static_cast<float>(gridDim - 1)) * spacing;
            settings.tiles[t].originY = 0.0f;
            settings.tiles[t].originZ = (static_cast<float>(row) - 0.5f * static_cast<float>(gridDim - 1)) * spacing;
        }
    }
    if (config.recipeCount > 1) {
        setupRecipes(settings, config);
    } else {
        settings.turbulence.strength = config.turbulenceStrength;
        settings.turbulence.frequency = config.turbulenceFrequency;
        if (config.enableGroundPlane) {
            settings.plane.enabled = true;
            settings.plane.point = {0.0f, config.groundPlaneHeight, 0.0f};
            settings.plane.normal = {0.0f, 1.0f, 0.0f};
            settings.plane.bounce = 0.6f;
            settings.plane.friction = 0.15f;
        }
        if (config.enableSphereCollision) {
            settings.sphere.enabled = true;
            settings.sphere.center = {0.0f, 5.0f, 0.0f};
            settings.sphere.radius = config.sphereRadius;
            settings.sphere.bounce = 0.7f;
            settings.sphere.friction = 0.1f;
        }
        if (config.enableBoxCollision) {
            settings.box.enabled = true;
            settings.box.minBounds = {-15.0f, 0.0f, -15.0f};
            settings.box.maxBounds = {15.0f, 30.0f, 15.0f};
            settings.box.bounce = 0.6f;
            settings.box.friction = 0.1f;
        }
        if (config.enableCurves) {
            settings.curves.enabled = true;
            settings.curves.startSize = 0.5f;
            settings.curves.peakSize = 2.0f;
            settings.curves.endSize = 0.1f;
            settings.curves.startColor = {1.0f, 0.8f, 0.2f, 1.0f};
            settings.curves.endColor = {0.8f, 0.1f, 0.05f, 0.0f};
        }
    }
    system.setSettings(settings);

    addDefaultEmitters(system, config);

    BenchmarkResult result = {};
    result.config = config;

    // Per-frame telemetry collection for percentile computation.
    std::vector<float> spawnTimings;
    std::vector<float> simulateTimings;
    std::vector<float> compactTimings;
    std::vector<float> totalTimings;
    std::vector<float> bandwidthSamples;
    spawnTimings.reserve(config.frames);
    simulateTimings.reserve(config.frames);
    compactTimings.reserve(config.frames);
    totalTimings.reserve(config.frames);
    bandwidthSamples.reserve(config.frames);

    // Burst workload: queue capacity/4 particles every kBurstIntervalFrames,
    // split evenly among emitters.
    const uint32_t burstPerEmitter = config.workload == Workload::Burst
        ? std::max(1u, config.capacity / (4u * config.emitterCount))
        : 0u;

    uint64_t lastReportedFrame = std::numeric_limits<uint64_t>::max();
    uint64_t lastTelemetryFrame = std::numeric_limits<uint64_t>::max();

    for (uint32_t frame = 0; frame < config.frames; ++frame) {
        if (config.workload == Workload::Burst &&
            frame % kBurstIntervalFrames == 0) {
            for (uint32_t sid = 0; sid < config.emitterCount; ++sid) {
                system.queueBurst(sid, burstPerEmitter);
            }
        }

        system.update(config.dt);
        const vparticles::SimulationStats& stats = system.stats();

        if (stats.valid && stats.frameIndex != lastTelemetryFrame) {
            ++result.telemetrySamples;
            result.maxTelemetryLatencyFrames = std::max(
                result.maxTelemetryLatencyFrames,
                stats.telemetryLatencyFrames);
            lastTelemetryFrame = stats.frameIndex;

            spawnTimings.push_back(stats.spawnMs);
            simulateTimings.push_back(stats.simulateMs);
            compactTimings.push_back(stats.compactMs);
            totalTimings.push_back(stats.totalMs);
            bandwidthSamples.push_back(
                simulateBandwidthGBs(stats.aliveCount, stats.simulateMs, bytesPerParticle));
        }

        if (sampleEvery != 0 &&
            stats.valid &&
            stats.frameIndex != lastReportedFrame &&
            stats.frameIndex % sampleEvery == 0) {
            printStats(stats, bytesPerParticle);
            lastReportedFrame = stats.frameIndex;
        }
    }

    // Explicit reporting boundary: flush the final delayed sample so the
    // terminal line and percentile distribution include an exact snapshot.
    system.synchronize();
    result.finalStats = system.stats();

    if (result.finalStats.valid &&
        result.finalStats.frameIndex != lastTelemetryFrame) {
        spawnTimings.push_back(result.finalStats.spawnMs);
        simulateTimings.push_back(result.finalStats.simulateMs);
        compactTimings.push_back(result.finalStats.compactMs);
        totalTimings.push_back(result.finalStats.totalMs);
        bandwidthSamples.push_back(
            simulateBandwidthGBs(result.finalStats.aliveCount,
                                 result.finalStats.simulateMs,
                                 bytesPerParticle));
    }

    if (sampleEvery != 0 &&
        result.finalStats.valid &&
        result.finalStats.frameIndex != lastReportedFrame) {
        printStats(result.finalStats, bytesPerParticle);
    }

    // Compute timing percentiles across all collected telemetry samples.
    result.spawnPercentiles = computePercentiles(spawnTimings);
    result.simulatePercentiles = computePercentiles(simulateTimings);
    result.compactPercentiles = computePercentiles(compactTimings);
    result.totalPercentiles = computePercentiles(totalTimings);

    // Compute bandwidth statistics.
    if (!bandwidthSamples.empty()) {
        std::sort(bandwidthSamples.begin(), bandwidthSamples.end());
        result.peakBandwidthGBs = bandwidthSamples.back();
        result.medianBandwidthGBs = percentile(bandwidthSamples, 0.50f);
    }

    result.tileStats = system.tileStats();

    return result;
}

// ── Matrix terminal output ───────────────────────────────────────────

void printMatrixHeader()
{
    std::cout << std::setw(11) << "workload"
              << std::setw(11) << "capacity"
              << std::setw(9)  << "emitters"
              << std::setw(8)  << "recipes"
              << std::setw(11) << "alive"
              << std::setw(9)  << "spawned"
              << std::setw(9)  << "dropped"
              << std::setw(9)  << "dead"
              << std::setw(9)  << "simP50"
              << std::setw(9)  << "simP95"
              << std::setw(9)  << "simP99"
              << std::setw(9)  << "totP50"
              << std::setw(9)  << "totP95"
              << std::setw(9)  << "peakBW"
              << std::setw(9)  << "medBW"
              << std::setw(7)  << "lag"
              << '\n';
}

void printMatrixRow(const BenchmarkResult& result)
{
    const vparticles::SimulationStats& stats = result.finalStats;
    std::cout << std::setw(11) << workloadName(result.config.workload)
              << std::setw(11) << result.config.capacity
              << std::setw(9)  << result.config.emitterCount
              << std::setw(8)  << result.config.recipeCount
              << std::setw(11) << stats.aliveCount
              << std::setw(9)  << stats.spawned
              << std::setw(9)  << stats.dropped
              << std::setw(9)  << stats.deadCount
              << std::fixed << std::setprecision(3)
              << std::setw(9)  << result.simulatePercentiles.p50
              << std::setw(9)  << result.simulatePercentiles.p95
              << std::setw(9)  << result.simulatePercentiles.p99
              << std::setw(9)  << result.totalPercentiles.p50
              << std::setw(9)  << result.totalPercentiles.p95
              << std::setprecision(1)
              << std::setw(9)  << result.peakBandwidthGBs
              << std::setw(9)  << result.medianBandwidthGBs
              << std::setw(7)  << result.maxTelemetryLatencyFrames
              << '\n';
}

// ── CSV output ───────────────────────────────────────────────────────

void writeCsvHeader(std::ostream& output)
{
    output << "workload,capacity,emitters,recipes,tiles,frames,spawnRate,dt,"
           << "submission,graphRebuilds,frame,alive,requestedSpawn,spawned,dropped,dead,"
           << "spawnMs,simulateMs,compactMs,totalMs,"
           << "spawnP50,spawnP95,spawnP99,"
           << "simP50,simP95,simP99,"
           << "compactP50,compactP95,compactP99,"
           << "totalP50,totalP95,totalP99,"
           << "peakBW_GBs,medianBW_GBs,"
           << "poolMB,telemetrySamples,maxTelemetryLag\n";
}

void writeCsvRow(std::ostream& output, const BenchmarkResult& result)
{
    const vparticles::SimulationStats& stats = result.finalStats;
    output << workloadName(result.config.workload) << ','
           << result.config.capacity << ','
           << result.config.emitterCount << ','
           << result.config.recipeCount << ','
           << result.config.tileCount << ','
           << result.config.frames << ','
           << result.config.spawnRate << ','
           << result.config.dt << ','
           << (result.config.useGraph ? "graph" : "eager") << ','
           << stats.graphRebuildCount << ','
           << stats.frameIndex << ','
           << stats.aliveCount << ','
           << stats.requestedSpawn << ','
           << stats.spawned << ','
           << stats.dropped << ','
           << stats.deadCount << ','
           << stats.spawnMs << ','
           << stats.simulateMs << ','
           << stats.compactMs << ','
           << stats.totalMs << ','
           << result.spawnPercentiles.p50 << ','
           << result.spawnPercentiles.p95 << ','
           << result.spawnPercentiles.p99 << ','
           << result.simulatePercentiles.p50 << ','
           << result.simulatePercentiles.p95 << ','
           << result.simulatePercentiles.p99 << ','
           << result.compactPercentiles.p50 << ','
           << result.compactPercentiles.p95 << ','
           << result.compactPercentiles.p99 << ','
           << result.totalPercentiles.p50 << ','
           << result.totalPercentiles.p95 << ','
           << result.totalPercentiles.p99 << ','
           << result.peakBandwidthGBs << ','
           << result.medianBandwidthGBs << ','
           << estimatedPoolMemoryMB(result.config.capacity, result.config.usePacked) << ','
           << result.telemetrySamples << ','
           << result.maxTelemetryLatencyFrames << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        BenchmarkConfig config;
        bool runMatrix = false;
        bool explicitSpawnRate = false;
        std::string csvPath;
        std::vector<uint32_t> matrixCapacities = {1'000'000, 5'000'000, 10'000'000};
        std::vector<uint32_t> matrixEmitters = {1, 8, 64};
        std::vector<Workload> matrixWorkloads = {
            Workload::Ramp,
            Workload::Recycle,
            Workload::Saturation,
            Workload::Burst,
        };

        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--capacity" && index + 1 < argc) {
                config.capacity = parseUintArg(argv[++index], config.capacity);
            } else if (arg == "--frames" && index + 1 < argc) {
                config.frames = parseUintArg(argv[++index], config.frames);
            } else if (arg == "--emitters" && index + 1 < argc) {
                config.emitterCount = std::max(1u, parseUintArg(argv[++index], config.emitterCount));
            } else if (arg == "--spawn-rate" && index + 1 < argc) {
                config.spawnRate = parseFloatArg(argv[++index], config.spawnRate);
                explicitSpawnRate = true;
            } else if (arg == "--dt" && index + 1 < argc) {
                config.dt = parseFloatArg(argv[++index], config.dt);
            } else if (arg == "--workload" && index + 1 < argc) {
                config.workload = parseWorkload(argv[++index], config.workload);
            } else if (arg == "--matrix") {
                runMatrix = true;
            } else if (arg == "--matrix-capacities" && index + 1 < argc) {
                matrixCapacities = parseUintListArg(argv[++index], matrixCapacities);
            } else if (arg == "--matrix-emitters" && index + 1 < argc) {
                matrixEmitters = parseUintListArg(argv[++index], matrixEmitters);
            } else if (arg == "--matrix-workloads" && index + 1 < argc) {
                matrixWorkloads = parseWorkloadList(argv[++index], matrixWorkloads);
            } else if (arg == "--turbulence" && index + 1 < argc) {
                config.turbulenceStrength = parseFloatArg(argv[++index], config.turbulenceStrength);
            } else if (arg == "--turb-freq" && index + 1 < argc) {
                config.turbulenceFrequency = parseFloatArg(argv[++index], config.turbulenceFrequency);
            } else if (arg == "--ground-plane") {
                config.enableGroundPlane = true;
                if (index + 1 < argc && argv[index + 1][0] != '-') {
                    config.groundPlaneHeight = parseFloatArg(argv[++index], config.groundPlaneHeight);
                }
            } else if (arg == "--sphere-collision") {
                config.enableSphereCollision = true;
                if (index + 1 < argc && argv[index + 1][0] != '-') {
                    config.sphereRadius = parseFloatArg(argv[++index], config.sphereRadius);
                }
            } else if (arg == "--box-collision") {
                config.enableBoxCollision = true;
            } else if (arg == "--curves") {
                config.enableCurves = true;
            } else if (arg == "--packed") {
                config.usePacked = true;
            } else if (arg == "--no-graph") {
                config.useGraph = false;
            } else if (arg == "--tiles" && index + 1 < argc) {
                config.tileCount = std::max(1u, parseUintArg(argv[++index], config.tileCount));
            } else if (arg == "--recipes" && index + 1 < argc) {
                config.recipeCount = std::min(vparticles::kMaxRecipes,
                    std::max(1u, parseUintArg(argv[++index], config.recipeCount)));
            } else if (arg == "--tile-scale" && index + 1 < argc) {
                config.tileScale = parseFloatArg(argv[++index], config.tileScale);
            } else if (arg == "--csv" && index + 1 < argc) {
                csvPath = argv[++index];
            }
        }

        const GpuDeviceInfo gpuInfo = queryGpuDeviceInfo();
        printGpuDeviceInfo(gpuInfo);

        if (!runMatrix) {
            if (!explicitSpawnRate) {
                config.spawnRate = workloadSpawnRate(
                    config.workload, config.capacity, config.frames, config.dt);
            }

            std::cout << "VParticles compute benchmark\n"
                      << "workload=" << workloadName(config.workload)
                      << " capacity=" << config.capacity
                      << " frames=" << config.frames
                      << " emitters=" << config.emitterCount
                      << " spawnRate=" << std::fixed << std::setprecision(0)
                      << config.spawnRate
                      << " dt=" << std::setprecision(6) << config.dt
                      << " poolMB=" << std::setprecision(0)
                      << estimatedPoolMemoryMB(config.capacity, config.usePacked)
                      << " storage=" << (config.usePacked ? "packed" : "fp32")
                      << " submission=" << (config.useGraph ? "[graph]" : "[eager]")
                      << " recipes=" << config.recipeCount
                      << " tiles=" << config.tileCount;
            if (config.turbulenceStrength > 0.0f || config.enableGroundPlane ||
                config.enableSphereCollision || config.enableBoxCollision || config.enableCurves) {
                std::cout << " modules=[";
                if (config.turbulenceStrength > 0.0f) std::cout << " turb=" << config.turbulenceStrength;
                if (config.enableGroundPlane) std::cout << " ground(y=" << config.groundPlaneHeight << ")";
                if (config.enableSphereCollision) std::cout << " sphere(r=" << config.sphereRadius << ")";
                if (config.enableBoxCollision) std::cout << " box";
                if (config.enableCurves) std::cout << " curves";
                std::cout << " ]";
            }
            std::cout << "\n\n";

            const BenchmarkResult result = runBenchmark(config, 30);

            std::cout << "\n--- summary ---\n"
                      << std::fixed << std::setprecision(3)
                      << "simulate  p50=" << result.simulatePercentiles.p50
                      << "  p95=" << result.simulatePercentiles.p95
                      << "  p99=" << result.simulatePercentiles.p99 << " ms\n"
                      << "total     p50=" << result.totalPercentiles.p50
                      << "  p95=" << result.totalPercentiles.p95
                      << "  p99=" << result.totalPercentiles.p99 << " ms\n"
                      << std::setprecision(1)
                      << "bandwidth peak=" << result.peakBandwidthGBs
                      << "  median=" << result.medianBandwidthGBs << " GB/s\n"
                      << "telemetry " << result.telemetrySamples << " samples"
                       << "  maxLag=" << result.maxTelemetryLatencyFrames
                       << "  graphRebuilds=" << result.finalStats.graphRebuildCount << '\n';

            if (result.config.tileCount > 1) {
                std::cout << "\n--- tile breakdown (" << result.config.tileCount << " tiles) ---\n";
                for (const auto& ts : result.tileStats) {
                    if (ts.aliveCount > 0 || result.config.tileCount <= 16) {
                        std::cout << "  tile " << std::setw(3) << ts.tileId
                                  << " origin=(" << std::setw(7) << std::fixed << std::setprecision(1)
                                  << ts.origin.x << ", " << ts.origin.y << ", " << ts.origin.z << ")"
                                  << " alive=" << std::setw(9) << ts.aliveCount << "\n";
                    }
                }
            }

            if (!csvPath.empty()) {
                std::ofstream csvFile(csvPath);
                if (!csvFile) {
                    throw std::runtime_error(
                        "Unable to open CSV output path: " + csvPath);
                }
                writeCsvHeader(csvFile);
                writeCsvRow(csvFile, result);
                std::cout << "CSV written to " << csvPath << '\n';
            }

            return 0;
        }

        // ── Matrix mode ──────────────────────────────────────────────

        std::ofstream csvFile;
        if (!csvPath.empty()) {
            csvFile.open(csvPath);
            if (!csvFile) {
                throw std::runtime_error(
                    "Unable to open CSV output path: " + csvPath);
            }
            writeCsvHeader(csvFile);
        }

        std::cout << "VParticles benchmark matrix\n"
                  << "workloads=" << joinWorkloadList(matrixWorkloads)
                  << "  capacities=" << joinUintList(matrixCapacities)
                  << "  emitters=" << joinUintList(matrixEmitters)
                  << "  frames=" << config.frames
                  << "  dt=" << config.dt
                  << "  submission=" << (config.useGraph ? "[graph]" : "[eager]")
                  << "\n\n";
        printMatrixHeader();

        for (const Workload workload : matrixWorkloads) {
            for (const uint32_t capacity : matrixCapacities) {
                for (const uint32_t emitterCount : matrixEmitters) {
                    BenchmarkConfig scenario = config;
                    scenario.capacity = capacity;
                    scenario.emitterCount = std::max(1u, emitterCount);
                    scenario.workload = workload;
                    if (!explicitSpawnRate) {
                        scenario.spawnRate = workloadSpawnRate(
                            workload, capacity, scenario.frames, scenario.dt);
                    }

                    const BenchmarkResult result = runBenchmark(scenario, 0);
                    printMatrixRow(result);
                    if (csvFile) {
                        writeCsvRow(csvFile, result);
                    }
                }
            }
            // Visual separator between workload groups.
            std::cout << '\n';
        }

        if (csvFile) {
            std::cout << "CSV written to " << csvPath << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VParticles failed: " << error.what() << '\n';
        return 1;
    }
}
