# VParticles Compute Status Report

Updated: 2026-08-27

## Scope

VParticles is currently a CUDA-only, compute-only particle simulation engine. Rendering, Blender integration, editor tooling, and external DCC bridges are intentionally out of scope until the simulation path is proven at scale.

The current target is game VFX simulation with 10M+ live particles on NVIDIA GPUs.

## Completed Work

- Replaced the legacy application surface with a headless C++20/CUDA benchmark.
- Created one global FP32 structure-of-arrays pool for position, velocity, color, age, lifetime, and `systemId`.
- Added deterministic hash-based spawn variation without a per-particle RNG-state buffer.
- Implemented fused gravity, wind, drag, Euler integration, age, lifetime kill, and alpha fade in one simulation kernel.
- Implemented a persistent GPU free-list for stable particle slots.
- Simulate through a dense active-index list; dead particles return their slot to the free-list.
- Keep active counters, free-list counters, and active-list selection state resident on the GPU.
- Use a CUB block-scan compaction kernel to compact only `uint32_t` active indices after deaths, rather than copying every particle attribute.
- Batch all emitter work into one host spawn-command upload and one GPU spawn launch per frame.
- Added a delayed telemetry ring for per-stage GPU timing and particle lifecycle counters.
- Added `ParticleSystem::synchronize()` as an explicit reporting and validation boundary.

## Latest Validation

Smoke-test commands run after the GPU-resident telemetry rewrite:

```text
.\out\build\x64-Release\VParticles.exe --capacity 10000 --frames 20 --spawn-rate 30000
.\out\build\x64-Release\VParticles.exe --capacity 10000 --frames 30 --spawn-rate 60000
.\out\build\x64-Release\VParticles.exe --capacity 10000 --frames 300 --spawn-rate 30000
.\out\build\x64-Release\VParticles.exe --capacity 100000 --frames 120 --emitters 64 --spawn-rate 250000
```

Final synchronized samples:

| Scenario | Frame | Live | Spawned | Dead | Dropped | Spawn | Simulate | Compact | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10K ramp | 19 | 10,000 | 500 | 0 | 0 | 0.046 ms | 0.054 ms | 0.027 ms | 0.127 ms |
| 10K saturation | 29 | 10,000 | 0 | 0 | 1,000 | 0.027 ms | 0.037 ms | 0.024 ms | 0.087 ms |
| 10K recycle | 299 | 10,000 | 50 | 50 | 450 | 0.038 ms | 0.059 ms | 0.010 ms | 0.107 ms |
| 100K, 64 emitters | 119 | 100,000 | 0 | 0 | 4,160 | 0.011 ms | 0.174 ms | 0.009 ms | 0.194 ms |

Interpretation:

- Delayed telemetry reports completed GPU frames without forcing `update()` to wait.
- The explicit final `synchronize()` publishes an exact host-visible endpoint for benchmark output.
- Spawn drops, dead counts, and replacement spawns behave correctly at capacity.
- Full-scale 1M/5M/10M measurements need to be refreshed with the benchmark matrix.

## Current Architecture

```text
simulate active particles
  -> return dead slots to GPU free-list
  -> compact active indices on the GPU when deaths occur
  -> upload batched spawn commands
  -> reserve and spawn into reclaimed slots
  -> publish delayed telemetry
```

The active-index/free-list design avoids expensive full-attribute gathers during lifecycle updates. Host-visible `ParticlePool` counters are completed telemetry snapshots; CUDA consumers that need current active-list state should use `ParticleSystem::gpuBuffers()`.

## Known Limits

- Host-visible stats are delayed by design. Call `ParticleSystem::synchronize()` only at explicit validation or reporting boundaries that need an exact snapshot.
- High-occupancy locality repair and `systemId` grouping are deferred until Nsight confirms whether sorting active indices is worth the cost.
- Spawn commands are still uploaded from a small host vector. A pinned upload ring is a useful next production polish item.
- The current particle format is FP32. It prioritizes a measurable baseline over memory compression.
- All particles use one fixed simulation recipe. `systemId` exists, but particles are not yet grouped by effect recipe.
- The benchmark does not yet emit GPU model, effective memory bandwidth, or automated scenario summaries.
- No Nsight capture has yet validated occupancy, memory transactions, or warp divergence.

## Next Work

### 1. Benchmark Matrix

Add reproducible 1M, 5M, and 10M scenarios with 1, 8, and 64 emitters. Record ramp, capacity saturation, and sustained recycling numbers with timing percentiles.

### 2. Nsight Bottleneck Pass

Confirm whether the next ceiling is memory bandwidth, active-list compaction, command upload, or spawn distribution at high occupancy.

### 3. Split Simulation Core From Effect Logic

Keep allocator, active list, compaction, telemetry, and scheduling generic. Move effect behavior into compact module/recipe data so forces and lifecycle modules can grow without rewriting the core.

### 4. Group By Effect Recipe

When multiple simulation recipes are added, use CUB radix sort to group active indices by `systemId`. Do this only after measuring warp divergence; a single recipe does not need this cost.

### 5. Fused Simulation Modules

Add turbulence, simple collisions, color-over-life, and size-over-life as compact parameter tables consumed by fused kernels. Avoid turning each module into another full-pool memory pass.

### 6. Packed State Experiment

Only after the FP32 and lifecycle baselines are profiled, compare tile-local quantized position, FP16 velocity, normalized uint16 age/lifetime, and RGBA8 color. Decode into FP32 registers for simulation, then encode back. The decision depends on measured bandwidth savings and acceptable VFX error, not compression alone.

### 7. CUDA Graphs And Rendering

Capture the stable production pipeline with CUDA Graphs after the control path settles. Rendering and any Blender remapping remain later adapter work, not simulation-core work.
