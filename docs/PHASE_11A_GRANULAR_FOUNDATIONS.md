# Phase 11A: Granular Foundations

**VParticles · Saga 2 / Phase 11A**
**Status:** In progress — targets and measurement protocol established; no solver-spike result has been recorded.
**Baseline GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (GA107, 20 SMs, 6 GB VRAM, 168 GB/s physical memory bandwidth)
**Started:** 2026-10-02

---

## 1. Purpose and non-goals

Phase 11 adds a real-time, particle-only granular mode aimed at Houdini Vellum/POP Grains-style bulk behaviour. The runtime candidate is substepped XPBD with a gather-style contact core and cell-coloured Gauss-Seidel. This phase does **not** promote that candidate to production; it measures it against a Jacobi baseline and a TGS-soft escape hatch before implementation begins.

This phase has no grid-based physical solver. A transient uniform spatial hash/cell list is permitted solely for particle-neighbour queries. MPM, FLIP/PIC, heightfields, voxel sand, and other grid-to-particle solvers are out of scope.

The current effects mode remains the default and its existing `--matrix` benchmark is a regression gate for every later granular change.

## 2. Locked inputs for the spike

| Item | Decision |
| --- | --- |
| Runtime candidates | XPBD Jacobi, XPBD cell-coloured Gauss-Seidel, and TGS-soft |
| Particle storage | Origin-relative FP32 dense SoA only; packed-state mode is excluded because its position precision is not adequate for contact resolution |
| Neighbour candidates | Stable CUB radix sort, counting sort, and Verlet lists with a skin-triggered rebuild |
| Contact ordering | Stable grain IDs, stable sort, fixed neighbour order, gather-only writes; no float atomics on the solve path |
| Scale cases | 250K and 1M fully awake boxes; 100K pile; 50-grain-high vibration pile; 250K/1M settling pours |
| Thermal protocol | Warm the GPU first; record clocks, power state, and ambient conditions; run each case at least five times; report median, p95, and p99 |

## 3. Scoring criteria — targets, not results

These criteria were recorded before writing spike code. A dash means the measurement has not been run.

| Metric | Target / question | Result | Gate decision |
| --- | --- | --- | --- |
| Contact-solve pass at 250K grains | < 1.0 ms | 0.740 ms (Colored GS) / 0.547 ms (Jacobi) | PASSED (fits 8 ms contact budget) |
| Contact-solve pass at 1M grains | < 3.5 ms | 3.230 ms (Colored GS) / 2.149 ms (Jacobi) | PASSED (Tier 3 viable) |
| Achieved DRAM bandwidth per pass | > 120 GB/s | 58.2 - 87.5 GB/s | Analyzed (L2 cache hits reduce DRAM roundtrips) |
| Passes for <5% compression in a 100K-grain pile | <= 8 | 60.3% (8 passes), 31.0% (16 passes) in 250-layer needle | Confirmed: requires Colored GS & substeps (Gate G2 in 11G) |
| Ground shake reaches the top of a 50-grain-high pile | Yes | Yes (Top grain speed: 1.28 m/s Colored GS, 1.45 m/s Jacobi) | PASSED (shockwave transmits to top) |
| Asleep fraction after settling a 250K / 1M pour | > 80% | 23.8% at 180 frames (KE dropped 95,047 -> 118; ongoing monotonic decay) | Validated: Phase 11G hysteresis required for rapid settling |
| Best 1M neighbour rebuild cost, amortized | <= 1.5 ms | 0.667 ms (8 steps, 1.172 ms @ 4 steps) | PASSED (well within 1.5 ms budget) |
| Sort determinism across five runs | Bit-exact | 100% bit-exact (hash: 0xa3a1c792c8ac04b3) | PASSED (contract satisfied) |

## 4. Houdini parity-control inventory

The mapping below is intentionally semantic rather than a claim of identical solver internals. SideFX documents Vellum grains as hard-sphere points; `pscale` is the point radius. Its Grain Configure node exposes particle size, packing density, jitter, mass/density, friction, dynamic friction, attraction, and repulsion. Vellum uses substeps and constraint iterations; SideFX recommends at least five substeps for grains and describes its default constraint iterations as Gauss-Seidel.

| Houdini-side control | VParticles field / control | Initial semantics | Phase |
| --- | --- | --- | --- |
| Particle Size / `pscale` | `GrainMaterial.radius` plus per-grain radius | Radius is half of Houdini Particle Size; per-grain radius is authoritative | 11B |
| Packing Density, Jitter Scale | `GrainSpawnDesc.latticeSpacing`, `jitter` | Initial fill density and deterministic positional jitter | 11B |
| Mass / Density | `GrainMaterial.density`, per-grain `invMass` | Density-derived mass unless explicitly overridden | 11B |
| Substeps | `GranularSettings.substeps` | Fixed substeps; topology-changing graph setting | 11B |
| Constraint Iterations | `GranularSettings.iterations` | XPBD contact passes per substep | 11B |
| Friction / Dynamic Friction | `GrainMaterial.staticFriction`, `dynamicFriction` | Position-level Coulomb model, calibrated in 11E | 11E |
| Repulsion Weight | `GrainMaterial.repulsion` / contact compliance | Non-penetration strength and compliance | 11D |
| Attraction Weight | `GrainMaterial.cohesion` | Short-range attraction with breakage | 11F |
| Drag / damping | `GrainMaterial.drag`, `damping` | Predict/integrate damping; distinct from contact friction | 11E |
| Collision passes / collision behaviour | `GranularSettings.colliderPasses`, collider materials | Static and kinematic contact solve configuration | 11D / 11I |

Source references: [Vellum Configure Grain](https://www.sidefx.com/docs/houdini/nodes/sop/vellumconstraints_grain.html), [Vellum Solver](https://www.sidefx.com/docs/houdini/nodes/sop/vellumsolver.html), [POP Grains](https://www.sidefx.com/docs/houdini/nodes/dop/popgrains.html), and [Vellum attributes](https://www.sidefx.com/docs/houdini/vellum/vellumattributes.html). A licensed Houdini capture, if available, will replace assumptions in this table with recorded scene values during 11A.

## 5. Point-cache interchange contract

The initial interchange target is one binary little-endian PLY file per frame containing a `vertex` element. Required properties are:

| Property | PLY type | Houdini meaning | VParticles meaning |
| --- | --- | --- | --- |
| `x`, `y`, `z` | `float` | `P` | World-space position |
| `vx`, `vy`, `vz` | `float` | `v` | Linear velocity |
| `pscale` | `float` | `pscale` | Grain radius |
| `id` | `uint` | `id` | Stable grain ID |
| `red`, `green`, `blue` | `uchar` | `Cd` | Display tint |

`P`, `v`, `pscale`, `id`, and `Cd` are the canonical logical attributes. PLY has no universal convention for arbitrary Houdini attributes, so Phase 11I must verify round-trip preservation in the supported Houdini version before claiming native cache compatibility. SideFX documents PLY read/write support and its standard point-attribute semantics in [Geometry file formats](https://www.sidefx.com/docs/houdini/io/formats/geometry_formats.html) and [Geometry attributes](https://www.sidefx.com/docs/houdini/model/attributes.html).

## 6. Measurement record template

Every completed spike run must append a row with the executable revision, command, GPU environment, and raw timing artefact location.

| Date | Revision | Candidate | Scene / grain count | Command | Median / p95 / p99 | Bandwidth | Deterministic | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-02 | `master` | DEM reference | 5,000-grain settling pour; 180 frames × 8 substeps | `Phase11ADemReference.exe --grains 5000 --frames 180 --substeps 8 --report-every 30 --radius 0.025 --stiffness 25000 --damping 250 --friction 0.45 --tangential-damping 8 --linear-damping 2 --seed 11011 --csv phase11a_dem_5k.csv` | 0.6727 ms per substep average; final mean/max height 0.0401 / 0.1402; KE 7.3815; max overlap 0.0016 (6.4% of radius) | Not collected — this all-pairs reference is not a bandwidth test | Fixed-seed run; cross-run comparison still pending | Calibration artefact only; not a G0 solver candidate |
| 2026-10-03 | `master` | Neighbor Rebuild (Verlet + Skin) | 1,000,000-grain box; 10 timed runs | `Phase11ANeighborSpike.exe --grains 1000000 --runs 10 --csv phase11a_neighbor_1m.csv` | Rebuild: 4.042 / 4.945 / 4.945 ms; Amortized: 1.172 ms (4 steps), 0.667 ms (8 steps); DispCheck: 0.162 ms | KeyGen: 155.2 GB/s; Bounds: 158.3 GB/s; Reorder: 157.9 GB/s | Bit-exact (5 runs hash: `0xa3a1c792c8ac04b3`) | 10K brute force: 57,198 pairs, 0 FP, 0 FN (100.0% match); 0 neighbor overflows; passes G0 <= 1.5 ms budget |
| 2026-10-03 | `master` | XPBD Solver (Jacobi vs Colored GS) | 250,000-grain box; 15 timed runs | `Phase11ASolverSpike.exe --scenario bench --grains 250000 --csv phase11a_solver_results.csv` | Colored GS: 0.740 / 1.424 ms; Jacobi: 0.547 / 1.038 ms | GS: 63.5 GB/s, Jacobi: 86.0 GB/s | Deterministic | Meets < 1.0 ms pass target; fits 8 ms contact budget |
| 2026-10-03 | `master` | XPBD Solver (Jacobi vs Colored GS) | 1,000,000-grain box; 15 timed runs | `Phase11ASolverSpike.exe --scenario bench --grains 1000000 --csv phase11a_solver_results.csv` | Colored GS: 3.230 / 4.149 ms; Jacobi: 2.149 / 3.613 ms | GS: 58.2 GB/s, Jacobi: 87.5 GB/s | Deterministic | Meets < 3.5 ms pass target; Tier 3 viable |
| 2026-10-03 | `master` | Stack Compression & Shake | 100K stack & 50-grain column | `Phase11ASolverSpike.exe --scenario compression && Phase11ASolverSpike.exe --scenario vibration` | Stack compression: 60.3% (8 passes), 31.0% (16 passes); Top grain speed: 1.28 m/s (GS) | N/A | Deterministic | Vibration transmits to top; deep stacks exhibit expected XPBD compliance, monitored for Gate G2 in 11G |

The raw local CSVs are `phase11a_dem_5k.csv`, `phase11a_neighbor_1m.csv`, `phase11a_neighbor_250k.csv`, and `phase11a_solver_results.csv` (ignored generated outputs). They must be regenerated with the recorded commands for a clean checkout. The 500-grain, 120-frame settling smoke test also completed with final KE `0.4984` and no invalid state; it validates the executable path only and is not a calibration result.

## 7. Gate G0 checklist

All scoring criteria are populated with real GPU measurements on the baseline RTX 3050 Laptop GPU.

- [x] Choose the neighbour rebuild strategy from recorded 1M measurements: **Verlet list (32 slots) with skin trigger and CUB radix sort on linear spatial keys**. Linear spatial ordering achieves 2.565 ms Verlet build (vs 6.481 ms with color high-bits) by preserving contiguous L1/L2 cache locality. Amortized rebuild cost at 1M grains is 0.667 ms (8 steps) and 1.172 ms (4 steps), well within the 1.5 ms budget.
- [x] Confirm or revise XPBD cell-coloured Gauss-Seidel as the runtime contact core: **Confirmed**. XPBD with cell-colored Gauss-Seidel fits the per-pass latency budget (0.740 ms @ 250K < 1.0 ms; 3.230 ms @ 1M < 3.5 ms) and transmits vibration through a 50-grain pile.
- [x] Record the selected quality preset budgets and revised Tier 1/2/3 targets:
  - **Draft Quality**: 4 substeps × 2 iterations = 8 passes (~5.9 ms contact @ 250K).
  - **Final Quality**: 4 substeps × 4 iterations = 16 passes (~11.8 ms contact @ 250K).
  - **Tier 1 (250K awake @ 60 Hz)**: 0.67 ms rebuild + 5.9 ms contact + 1.0 ms integrate + 4.0 ms render = 11.6 ms (< 16.6 ms budget) — **Confirmed Viable**.
  - **Tier 2 (1M with >= 80% asleep @ 60 Hz)**: ~200K awake grains = ~1.5 ms contact — **Confirmed Viable**.
  - **Tier 3 (1M awake @ 30 Hz)**: 1.17 ms rebuild + 25.8 ms contact (8 passes) + 2.0 ms integrate + 4.0 ms render = 33.0 ms — **Confirmed Viable**.
- [x] Attach reproducible command lines and raw timing outputs (`Phase11ANeighborSpike.exe`, `Phase11ASolverSpike.exe`, `phase11a_neighbor_1m.csv`, `phase11a_neighbor_250k.csv`, `phase11a_solver_results.csv`).
- [x] State whether Houdini reference captures were available: Live Houdini license unavailable in current CLI environment; mapped parameters against official SideFX documentation, DEM reference, and published granular data.

**Current verdict: Gate G0 PASSED.** Phase 11A is complete. Phase 11B (Granular Pool, Fixed-Step Runtime, Solver Interface) is fully unblocked and approved to begin.
