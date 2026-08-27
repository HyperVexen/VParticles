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
- Use CUB select to compact only `uint32_t` active indices after deaths, rather than copying every particle attribute.
- Add a high-occupancy radix-sort repair pass to restore coalesced SoA access after recycled slots fragment the active list.
- Batch all emitter work into one host spawn-command upload and one GPU spawn launch per frame.
- Added per-stage GPU timing and particle lifecycle counters to the benchmark.

## Latest Benchmark

User-supplied benchmark configuration:

```text
capacity=1,000,000
frames=240
emitters=1
spawnRate=250,000 particles/second
dt=1/60 second
```

Latest recorded frame:

| Frame | Live | Spawned | Dead | Dropped | Spawn | Simulate | Lifecycle | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 239 | 951,364 | 4,167 | 1,999 | 0 | 0.046 ms | 0.861 ms | 0.191 ms | 1.099 ms |

Interpretation:

- The engine is well below a 16.67 ms frame budget in this 1M-capacity test.
- Particle death begins before the end of the run and lifecycle processing stays below 0.2 ms in the final sample.
- No spawn requests were dropped through frame 239.
- This is a strong near-capacity ramp result, but it is not yet a sustained high-churn result. Long runs must still test a full pool recycling particles every frame.

## Current Architecture

```text
simulate active particles
  -> return dead slots to GPU free-list
  -> compact active indices when deaths occur
  -> restore index locality at high occupancy
  -> upload batched spawn commands
  -> spawn into reclaimed slots
```

The active-index/free-list design avoids expensive full-attribute gathers during lifecycle updates. The locality repair pass is necessary because free-list reuse eventually scatters slot indices and makes SoA reads less coalesced.

## Known Limits

- `update()` currently performs CPU synchronization to obtain exact death and active counts for CUB selection and benchmark stats. This is correct and useful for profiling, but it is not the final production submission path.
- The current particle format is FP32. It prioritizes a measurable baseline over memory compression.
- All particles use one fixed simulation recipe. `systemId` exists, but particles are not yet grouped by effect recipe.
- The benchmark does not yet emit GPU model, effective memory bandwidth, or automated scenario summaries.
- No Nsight capture has yet validated occupancy, memory transactions, or warp divergence.

## Next Work

### 1. GPU-Resident Frame Control

Build a production update mode that keeps active counters and lifecycle decisions on the GPU. CPU stats should use a delayed staging/ring-buffer readback instead of forcing a per-frame wait.

This must preserve variable-work dispatch. Launching a capacity-wide simulation every frame merely to avoid a readback would waste the gains already established by the active-index path.

### 2. Benchmark Matrix And Lifecycle Choice

Add reproducible 100K, 1M, 5M, and 10M scenarios with ramp, sustained recycling, burst, and multi-emitter loads. Compare active-index/free-list and dense-SoA lifecycle modes, then choose defaults from measurements and Nsight traces.

### 3. Group By Effect Recipe

When multiple simulation recipes are added, use CUB radix sort to group active indices by `systemId`. Do this only after measuring warp divergence; a single recipe does not need this cost.

### 4. Fused Simulation Modules

Add turbulence, simple collisions, color-over-life, and size-over-life as compact parameter tables consumed by fused kernels. Avoid turning each module into another full-pool memory pass.

### 5. Packed State Experiment

Only after the FP32 and lifecycle baselines are profiled, compare tile-local quantized position, FP16 velocity, normalized uint16 age/lifetime, and RGBA8 color. Decode into FP32 registers for simulation, then encode back. The decision depends on measured bandwidth savings and acceptable VFX error, not compression alone.

### 6. CUDA Graphs And Rendering

Capture the stable production pipeline with CUDA Graphs after the control path settles. Rendering and any Blender remapping remain later adapter work, not simulation-core work.
