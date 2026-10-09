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

## Milestone Complete: Benchmark Matrix

- Automated 1M, 5M, and 10M scenarios across 1, 8, and 64 emitters.
- Included ramp, sustained-recycle, saturation, and burst workloads.
- Recorded GPU model, estimated memory use, effective bandwidth, and timing percentiles (p50/p95/p99).
- Added `--matrix`, `--workload`, `--matrix-workloads`, and `--csv` arguments.

## Milestone Complete: Nsight Bottleneck Pass

- Captured Nsight traces at 1M, 5M, and 10M alive particles across ramp and sustained-recycle workloads.
- Verified that `simulateKernel` consumes 95.5% - 96.2% of GPU execution time and achieves 160-170 GB/s on the 168 GB/s physical memory bus.
- Confirmed that active-list compaction (0.1% - 0.3% / ~2.1 µs) and multi-emitter spawn batching (2.6% - 3.8% / ~74 µs) have negligible overhead.
- Confirmed that radix-sorting active indices by `systemId` is not justified for unified recipes.
- Recorded detailed metrics and kernel profiles in [docs/NSIGHT_PROFILING_REPORT.md](docs/NSIGHT_PROFILING_REPORT.md).

## Milestone Complete: Phase 5 - Fused Simulation Modules

- Implemented divergence-free (div v = 0) 3D procedural curl-noise turbulence.
- Implemented analytical collision solvers (ground/arbitrary planes, bounding/obstacle spheres, AABB boxes) with bounce restitution and tangential surface friction.
- Implemented color-over-life and size-over-life curves utilizing existing pos.w and color channels (zero additional memory footprint).
- Fully fused inside simulateKernel with zero extra full-pool passes.
- Module parameters passed via CUDA constant memory, preserving 160-185 GB/s memory bandwidth.
- Verified at 1M and 10M scales with all modules active.

## Milestone Complete: Phase 6 - Packed State Mode

- Implemented compressed SoA storage (26 bytes/particle in pool, ~37 bytes/slot total vs 73 bytes/slot FP32).
- Quantized tile-local int16 positions (posX, posY, posZ) with configurable scale factor.
- IEEE fp16 velocities (velX, velY, velZ), lifetime, and size via <cuda_fp16.h>.
- Normalized uint16 age (age / lifetime * 65535).
- Packed RGBA8 color in single uint32.
- Fully fused decode (packed DRAM -> FP32 registers) and encode (FP32 registers -> packed DRAM) inside simulatePackedKernel and spawnBatchPackedKernel.
- Simulation arithmetic remains 100% FP32 in registers — zero loss in simulation physics or collision accuracy.
- Runtime dual-mode selectable via StorageMode::FP32 or StorageMode::Packed (CLI --packed).
- Reduced simulate memory footprint by ~50% and achieved 1.93x - 2.32x speedup on simulation pass.
- Successfully scaled to 20,000,000 particles at 7.4 ms simulation time on RTX 3050 Laptop GPU.

## Milestone Complete: Phase 7 - Spatial Scale

- Implemented multi-tile world representation supporting up to 256 tiles with world-space origins.
- Tile descriptor table (`TileDesc`) stored in CUDA `__constant__` memory (`cTiles`, 4 KB).
- Added `tileId` (`uint16_t`) attribute to `ParticlePool`, `GpuParticlePool`, and `PackedPool` (cost: +2 bytes/slot; 39 bytes/slot packed, 75 bytes/slot FP32).
- Extended `EmitterDesc` with `uint16_t tileId` allowing emitters to spawn particles directly into designated local tile origins.
- Implemented `migrateTilesPackedKernel` and `migrateTilesKernel` featuring zero-copy tile migration:
  - Particles find closest tile center in Voronoi space across the constant cache.
  - Zero DRAM writes for non-migrating particles (99.9% of particles in steady-state).
  - Fast block-reduction histogram using shared memory (`sTileCounts`) with only 1 global atomic per tile per block.
  - Bypassed entirely when `tileCount == 1` for zero single-tile overhead.
- Added tile-level metadata readback via `TileStats` on the delayed telemetry ring, exposed asynchronously via `ParticleSystem::tileStats()`.
- Validated at 1M, 5M, 10M, and 20M particles across 1, 4, and 16 tiles:
  - 1M particles across 4 tiles: 0.174 ms simulate p50, 174.9 GB/s peak bandwidth.
  - 5M particles across 16 tiles: 0.288 ms simulate p50, 1.64 ms final frame, 178.8 GB/s peak bandwidth.
  - 10M particles across 16 tiles: 0.415 ms simulate p50, 3.32 ms final frame, 180.4 GB/s peak bandwidth.
  - 20M particles across 16 tiles: 0.751 ms simulate p50, 8.71 ms final frame, 744 MB pool VRAM, 178.6 GB/s peak bandwidth.

## Milestone Complete: Phase 8 - Launch Overhead Reduction

- Captured the stable compute update sequence with CUDA Graphs: frame initialization, simulation, compaction, batched spawn upload, spawn, and optional tile migration.
- Replayed steady-state frames with one `cudaGraphLaunch` call; telemetry remains a delayed, conventional readback.
- Added a device-side per-frame parameter buffer, avoiding graph rebuilds for timestep, simulation time, frame index, and requested-spawn changes.
- Rebuilt only when dispatch topology, command count, settings, or captured storage changes.
- Added `--no-graph` eager fallback, graph-mode benchmark labeling, and graph rebuild observability.
- Recorded timing semantics and reproducible A/B evidence in [Phase 8 CUDA Graphs](docs/PHASE_8_CUDA_GRAPHS.md).

## Milestone Complete: Phase 9 - Data-Driven Effects

- Implemented `EffectRecipe` architecture supporting up to 64 distinct recipes in `__constant__ EffectRecipe cRecipes[kMaxRecipes]`.
- Decoupled effect definition (gravity, wind, drag, turbulence, plane/sphere/box collisions, curves, lifetime scale) from global simulation core.
- Stored recipe ID in existing `pool.systemId` SoA array (zero DRAM footprint increase).
- Optimized single-recipe workloads: bypassed `systemId` DRAM read when `recipeCount <= 1`, preserving 100% bandwidth parity.
- Validated at 1M, 10M, and 20M particles across 1, 4, and 8 recipes: 20M particles simulating 8 distinct recipes concurrently in 8.8 ms on RTX 3050 Laptop GPU.
- Added `--recipes <N>` CLI option and detailed documentation in [docs/PHASE_9_DATA_DRIVEN_EFFECTS.md](docs/PHASE_9_DATA_DRIVEN_EFFECTS.md).

## Milestone Complete: Phase 10A - Interactive OpenGL Viewer

- Built standalone interactive viewer executable (`VParticlesViewer.exe`) without modifying compute-only benchmark executable (`VParticles.exe`).
- Zero-copy CUDA-OpenGL buffer interop via `cudaGraphicsGLRegisterBuffer`.
- Mapped particle position/size (`float4 pos`) and color (`float4 color`) directly as OpenGL VBOs.
- Active-index indexed rendering (renders only live particles using `activeIndices` buffer without CPU roundtrips).
- Point sprite rendering with perspective sizing and smooth circular alpha / additive blending.
- Interactive camera orbit, pan, and zoom with real-time FPS, simulation time, and live particle count overlay.
- Real-time preset switching to showcase multi-recipe visual diversity (smoke, fire, sparks, plasma, fountain) at 1M-5M scale.
- Validated at 150 FPS for 1,000,000 live particles (0.22 ms simulation time) on RTX 3050 Laptop GPU.
- Detailed report in [docs/PHASE_10A_OPENGL_VIEWER.md](docs/PHASE_10A_OPENGL_VIEWER.md).

## Milestone Complete: Phase 10B - Modern Engine Interop (Vulkan / DX12)

- Engine-facing external memory import (`cudaExternalMemoryHandleDesc`, `cudaImportExternalMemory`, `cudaExternalMemoryGetMappedBuffer`).
- Cross-API timeline semaphore synchronization (`cudaImportExternalSemaphore`, `cudaWaitExternalSemaphoresAsync`, `cudaSignalExternalSemaphoresAsync`).
- Zero-copy resource sharing with DirectX 12 (`ID3D12Resource` NT handles, `D3D12_FENCE_FLAG_SHARED`) and Vulkan (`VK_KHR_external_memory`, timeline semaphores).
- GPU-driven indirect draw command generation (`D3D12DrawArguments` / `VkDrawIndirectCommand`) directly on GPU without CPU readback.
- Validated via `VParticlesD3D12InteropTest.exe` on RTX 3050 Laptop GPU at 3,564 FPS (0.28 ms/frame) with 100% data and hardware fence verification.
- Integration guide and architecture report in [docs/PHASE_10B_MODERN_ENGINE_INTEROP.md](docs/PHASE_10B_MODERN_ENGINE_INTEROP.md).

## Milestone Complete: Phase 11A - Granular Foundations & Spikes

- Built empirical spikes for spatial neighbour queries and contact solving on NVIDIA GeForce RTX 3050 Laptop GPU:
  - `Phase11ANeighborSpike.exe`: Verlet neighbour search with skin-triggered rebuilds, achieving 0.667 ms amortized rebuild at 1M grains (< 1.5 ms budget) with 100% bit-exact determinism across 5 runs.
  - `Phase11ASolverSpike.exe`: XPBD Jacobi vs. Cell-Colored Gauss-Seidel contact solvers, achieving 0.740 ms pass at 250K grains (< 1.0 ms budget) and 3.230 ms pass at 1M grains (< 3.5 ms budget).
  - Validated shockwave transmission through 50-grain column (1.28 m/s) and settling pour monotonic kinetic energy decay (95,047 -> 118).
- Gate G0 closed as **PASSED**; see [docs/PHASE_11A_GRANULAR_FOUNDATIONS.md](docs/PHASE_11A_GRANULAR_FOUNDATIONS.md).

## Milestone Complete: Phase 11B - Granular Pool, Fixed-Step Runtime, and Solver Interface

- Implemented `GranularPool` with dense FP32 Structure-of-Arrays (SoA) layout (60 bytes/slot: `pos`, `vel`, `pred`, `tint`, `materialId`, `flags`, `grainId`).
- Implemented fixed-step accumulator runtime (`GranularSystem`) preventing spiral-of-death with substepped symplectic Euler predict/integrate and CUDA Graph capture replay.
- Created extensible `IContactSolver` abstraction with baseline analytical `JacobiContactSolver`.
- Integrated dual-mode execution (`SimulationMode::Effects` vs `SimulationMode::Granular`) into `ParticleSystem`.
- Extended `VParticles.exe` with granular CLI suite: `--sand`, `--grain-radius`, `--radius-jitter`, `--drop-height`, `--substeps`, `--iterations`, `--quality`, `--sand-scenario`, `--analytic-check`.
- **Phase 11B Exit Criterion Verified**: 1,000,000 grains free-fall ballistic trajectory matched analytic solution within **0.510%** error (Target: $< 1.0\%$).
- **Zero Effects Regression**: Complete 36-workload `--matrix` suite verified with full bandwidth parity up to 221.2 GB/s.
- Detailed report in [docs/PHASE_11B_GRANULAR_POOL_RUNTIME.md](docs/PHASE_11B_GRANULAR_POOL_RUNTIME.md).

## Milestone Complete: Phase 11C - Spatial Neighbor Structure & Verlet Lists

- Production-hardened GPU uniform spatial hashing + CUB radix sort + Verlet list pipeline (`GranularNeighborSearch`).
- Dynamic 2-pass parallel bounding box reduction adapting grid to particle domain without static bounds (< 0.05 ms).
- Linear 32-bit spatial key sorting preserving L1/L2 cache locality with zero-copy dense SoA pointer swapping.
- Fixed 32-slot Verlet lists with skin buffer (`skinRatio = 0.20`) and $O(N)$ displacement check rebuild trigger.
- **Gate 11C-1 PASSED:** 10,000 grains 100% bit-exact pair match vs $O(N^2)$ brute-force ground truth (57,198 pairs discovered, 0 false positives, 0 missed).
- **Gate 11C-2 PASSED:** 1,000,000 grains amortized rebuild latency 0.896 ms / substep (budget: $\le 1.5$ ms).
- **Gate 11C-3 PASSED:** 100% bit-exact determinism across 5 runs on 100,000 grains (state hash `0xa36c93921c6809c9`).
- **Gate 11C-4 PASSED:** Explicit device overflow counter and telemetry tracking (clamped to 32 slots, zero corruption).
- **Gate 11C-5 PASSED:** Zero regression in default effects mode (36 / 36 workloads passed, up to 226.5 GB/s bandwidth).
- Detailed report in [docs/PHASE_11C_NEIGHBOR_STRUCTURE.md](docs/PHASE_11C_NEIGHBOR_STRUCTURE.md).

## Phase 11D: Cell-Colored Gauss-Seidel Contact Solver & Friction (Next)

- Implement 8-color cell-partitioned Gauss-Seidel XPBD contact solver utilizing Phase 11C neighbor lists and color tables.
- Add Coulomb friction model (static + dynamic friction cones) with compliance-weighted stabilization.
- Validate non-penetration, column shockwave propagation, resting sand pile angle of repose ($30^\circ - 35^\circ$), and solver latency budget ($\le 3.5$ ms at 1M grains).

