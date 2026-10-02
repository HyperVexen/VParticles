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
| Contact-solve pass at 250K grains | < 1.0 ms | — | Whether 4 substeps × 4 iterations can fit the 8 ms contact budget |
| Contact-solve pass at 1M grains | < 3.5 ms | — | Tier-3 viability |
| Achieved DRAM bandwidth per pass | > 120 GB/s | — | Validates the locality hypothesis |
| Passes for <5% compression in a 100K-grain pile | <= 8 | — | Quality gate; otherwise require coloured GS or TGS-soft |
| Ground shake reaches the top of a 50-grain-high pile | Yes | — | Vibration viability |
| Asleep fraction after settling a 250K / 1M pour | > 80% | — | Tier-2 viability |
| Best 1M neighbour rebuild cost, amortized | <= 1.5 ms | — | G0 neighbour budget |
| Sort determinism across five runs | Bit-exact | — | Determinism contract |

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

The raw local CSV is `phase11a_dem_5k.csv` (ignored generated output). It must be regenerated with the recorded command for a clean checkout. The 500-grain, 120-frame settling smoke test also completed with final KE `0.4984` and no invalid state; it validates the executable path only and is not a calibration result.

## 7. Gate G0 checklist

G0 is not ready to decide until each scoring row is populated.

- [ ] Choose the neighbour rebuild strategy from recorded 1M measurements.
- [ ] Confirm or revise XPBD cell-coloured Gauss-Seidel as the runtime contact core.
- [ ] Record the selected quality preset budgets and revised Tier 1/2/3 targets.
- [ ] Attach reproducible command lines and raw timing outputs.
- [ ] State whether Houdini reference captures were available; otherwise identify the published-data/DEM fallback.

**Current verdict:** Pending. No performance, parity, or solver-selection conclusion is implied by this document.
