# VParticles Roadmap

## North Star

Build VParticles as a standalone, high-scale GPU particle simulation engine for game VFX. The first target is simulation and compute only: no renderer, no Blender bridge, no editor integration, and no external DCC assumptions.

Target scale: 10M+ live particles on NVIDIA GPUs.

Current implementation status and the latest validation notes are recorded in [Project Status Report](docs/PROJECT_STATUS_REPORT.md).

## Core Decisions

- Use one shared global particle pool, not one buffer per effect.
- Store particle data as structure-of-arrays for coalesced GPU memory access.
- Keep the simulation bandwidth-aware from day one.
- Use CUDA for the first production backend.
- Hide raw CUDA allocation, launches, streams, and CUB scratch management behind thin internal wrappers.
- Start with FP32 storage for correctness and profiling.
- Add packed/quantized storage only after the baseline is measurable.
- Profile with Nsight before replacing simple mechanisms with complex ones.

## Phase 0: Compute-Only Foundation

- Recreate `VParticles/include` and `VParticles/src`.
- Remove renderer/UI assumptions from the first executable.
- Build a console or headless benchmark target first.
- Add core modules:
  - particle pool
  - emitter registry
  - simulation settings
  - update pipeline
  - CUDA error/runtime helpers
  - performance counters
  - deterministic test hooks

## Phase 1: Single Pool, Single Emitter

- Implement one global FP32 SoA particle pool:
  - position
  - velocity
  - color
  - age
  - lifetime
  - system id
- Add one point emitter.
- Add continuous spawn and burst spawn.
- Use atomic counters for initial slot allocation.
- Implement gravity, drag, and Euler integration.
- Implement kill/compact for active-index lifecycle management.
- Add a stats readback path for testing and benchmarks.

## Phase 2: Correctness And Profiling

The previous synchronized 1M ramp baseline must be refreshed after the GPU-resident telemetry rewrite:

- Re-run 1M, 5M, and 10M cases through the benchmark matrix.
- Include one-emitter, eight-emitter, and 64-emitter workloads.
- Include sustained recycle tests, not only near-capacity ramps.

- Add deterministic spawn randomness using particle id plus frame number.
- Prefer counter-based randomness such as Philox over persistent RNG state arrays.
- Add benchmark scenarios:
  - 100K particles
  - 1M particles
  - 5M particles
  - 10M particles
- Track:
  - spawn time
  - simulate time
  - compact time
  - total update time
  - effective memory bandwidth
  - alive count
  - spawn drops when capacity is exhausted

## Phase 3: Multiple Emitters, One Pool

Initial batching implementation complete:

- `EmitterDesc` and per-particle `systemId` allocation.
- Compact host-side spawn commands uploaded once per frame.
- One batched spawn pass and one simulation pass per frame, independent of emitter count.

Future grouping work:

- Add combined compact/group behavior:
  - dead particles removed
  - live particles grouped by `systemId`
- Use CUB radix sort when grouping is needed.
- Use block-scan active-list compaction when grouping is not needed.
- Verify warp divergence in Nsight instead of assuming it.

## Phase 4: Lifecycle Strategy

Initial implementation complete:

- Stable SoA particle slots with a dense active-index list.
- Persistent GPU free-list:
  - death pushes freed indices
  - spawn pops open slots
- A GPU-resident frame state owns active/free counters and active-list buffer selection.
- A CUB block-scan compaction kernel compacts only `uint32_t` active indices after particle death.

Next refinement:

- Add a benchmark switch for dense-SoA versus active-index lifecycle strategies.
- Revisit locality repair and `systemId` grouping as GPU-side sort/group passes after Nsight traces.
- Choose the default from realistic churn profiles and GPU traces rather than a fixed assumption.

## Milestone Complete: GPU-Resident Frame Control

- Keep production active counters and lifecycle decisions on the GPU.
- Move benchmark/debug statistics to delayed staging or a ring-buffer readback path.
- Remove the per-frame CPU wait from production submission without falling back to a capacity-wide simulation dispatch.
- Keep an exact synchronized boundary available for validation and benchmark final snapshots.

## Next Milestone: Benchmark Matrix

- Automate 1M, 5M, and 10M scenarios across 1, 8, and 64 emitters.
- Include ramp, sustained-recycle, saturation, and burst workloads.
- Record GPU model, memory use, effective bandwidth, and timing percentiles.
- Capture Nsight traces before changing lifecycle defaults or adding packed state.

## Phase 5: Fused Simulation Modules

- Add a small fixed set of hand-written CUDA modules:
  - gravity
  - drag
  - curl-noise turbulence
  - simple plane/sphere/box collision
  - color over life
  - size over life
- Fuse simulation work aggressively to reduce full-pool read/write passes.
- Store per-system curves and module parameters in compact GPU tables.

## Phase 6: Packed State Mode

Add alternate storage formats for scale after the FP32 baseline is stable:

```text
position: tile-local quantized int16/int32
velocity: fp16 or signed normalized 16-bit
age/lifetime: normalized uint16
color: rgba8
radius/size: fp16 or normalized uint16
flags: uint8/uint16
```

- Decode packed state into FP32 registers inside kernels.
- Simulate in FP32.
- Encode back to packed storage.
- Compare bandwidth, occupancy, visual error, and max particle count against FP32.

## Phase 7: Spatial Scale

- Add simulation tiles with world-space origins.
- Store large-world positions as tile origin plus local quantized position.
- Add tile-level bounds and culling metadata for future renderer integration.
- Keep this independent from any renderer API.

## Phase 8: Launch Overhead Reduction

- Capture the stable update sequence with CUDA Graphs:
  - begin frame
  - simulate
  - compact/group
  - reserve/spawn
  - telemetry
- Replay graphs for steady-state frames.
- Rebuild graphs only when pipeline structure changes.

## Phase 9: Data-Driven Effects

- Keep fixed kernels until they become limiting.
- Later, add NVRTC effect recipes:
  - compose module graph into CUDA source
  - compile per `systemId` or recipe class
  - cache compiled kernels
- Do this after the module set and data model settle.

## Phase 10: Rendering And External Remapping

- Add rendering only after compute performance is proven.
- Keep kernels accepting raw device pointers and counts.
- Future render interop options:
  - OpenGL buffer registration through CUDA interop
  - Vulkan external memory and semaphore interop
- Add external integrations as adapters, not core dependencies.

## First Build Target

The first implementation target is:

```text
Headless benchmark executable
+ one global FP32 SoA particle pool
+ one point emitter
+ spawn kernel
+ simulate kernel
+ GPU-resident active-list compaction
+ delayed telemetry stats readback
+ benchmark output
```

Once that is correct and profiled, expand to multiple emitters, grouping, packed storage, and CUDA Graphs.
