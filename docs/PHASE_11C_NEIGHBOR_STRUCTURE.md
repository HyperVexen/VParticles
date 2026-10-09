# Phase 11C: Spatial Neighbor Structure & Verlet Lists

**Status:** COMPLETE (All 5 Exit Gates Passed)  
**Date:** October 2026  
**Hardware Target:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, 20 SMs, Ampere SM 8.6, 96-bit GDDR6)  

---

## 1. Overview & Architecture

Phase 11C establishes the GPU spatial neighbor indexing pipeline for Saga 2 (Houdini-class particle-only granular sand). Under the project's strict architectural principles:
- **Particle-Only Physics:** The uniform grid is strictly an indexing and neighbor lookup data structure. No physical mass, velocity, stress, or momentum fields are ever accumulated on the grid (preventing MPM mesh/grid dissipation).
- **Contiguous Memory Order:** Linear 32-bit Morton/spatial cell keys preserve spatial coherency in L1/L2 caches, eliminating thread divergence during contact queries.
- **Fixed-Capacity Verlet Lists:** Fixed 32 neighbor slots per grain (`kMaxNeighborsPerGrain = 32`) with a Verlet skin buffer ($\delta_{\text{skin}} = 0.20 \cdot r$).
- **Displacement-Triggered Rebuilds:** $O(N)$ skin displacement checks only trigger expensive spatial sorting and neighbor reconstruction when a grain moves $> \delta_{\text{skin}} / 2$.
- **Zero-Copy Reordering:** GranularPool Structure of Arrays (SoA) pointers are swapped with scratch buffers to avoid redundant host-device memory movement.

```
+--------------------------------------------------------------------------------+
|                             NEIGHBOR PIPELINE                                  |
+--------------------------------------------------------------------------------+
  [GranularPool (pos, vel, pred, tint, matId, flags, grainId)]
                          |
                          v
  1. Two-Pass Min/Max Reduction (Compute dynamic particle domain bounds)
                          |
                          v
  2. generateKeysKernel (Compute linear 32-bit cell key per grain)
                          |
                          v
  3. cub::DeviceRadixSort::SortPairs (Linear spatial sort: keys + indices)
                          |
                          v
  4. findCellBoundsKernel (Contiguous [cellStart, cellEnd) lookup table)
                          |
                          v
  5. reorderGranularPoolKernel (Permute dense SoA + pointer swap)
                          |
                          v
  6. buildVerletListKernel (27-cell stencil traversal, 32 slots per grain)
                          |
                          v
  7. 8-Color Cell Partitioning (Permutation table prepared for Phase 11D GS)
                          |
                          v
  8. checkDisplacementKernel (Fast O(N) check per substep: trigger rebuild)
+--------------------------------------------------------------------------------+
```

---

## 2. Exit Gate Validation Results

All verification tests were executed directly on the physical target GPU (**NVIDIA GeForce RTX 3050 6GB Laptop GPU**, Driver 13.3, SM 8.6).

```
======================================================================
  SAGA 2 PHASE 11C: NEIGHBOR STRUCTURE & VERLET LIST VALIDATION SUITE
======================================================================
```

### Gate 11C-1: Ground-Truth 10K Neighbor Pair Match against $O(N^2)$ Brute Force
- **Configuration:** 10,000 grains, disordered lattice with radius jitter ($\pm 5\%$), skin ratio 20%.
- **Validation Method:** Full device-side $O(N^2)$ brute-force distance calculation (`bruteForceNeighborsKernel`) compared pair-by-pair against `GranularNeighborSearch`.
- **Results:**
  - Grains evaluated: 10,000
  - Grid neighbor pairs discovered: **57,198**
  - Brute-force ground truth pairs: **57,198**
  - False positives (extra pairs): **0**
  - False negatives (missed pairs): **0**
  - Max neighbors per grain: 6 (limit: 32)
  - Avg neighbors per grain: 5.72
  - Device overflow counter: 0
  - **Verdict: PASSED (100% bit-exact pair match)**

### Gate 11C-2: 1,000,000 Grains Latency & Amortization
- **Configuration:** 1,000,000 grains, 20% skin ratio.
- **Breakdown Timing:**
  - Key generation: `0.289 ms`
  - CUB Radix sort: `0.582 ms`
  - Cell bounds finding: `0.061 ms`
  - Dense SoA reordering: `0.765 ms`
  - 27-cell Verlet list build: `6.246 ms`
  - Full rebuild latency: `8.506 ms`
  - Displacement check latency: `0.045 ms`
  - Amortized latency per substep (10 substeps / rebuild): **`0.896 ms`**
  - Latency budget: $\le 1.5$ ms / substep
  - **Verdict: PASSED (`0.896 ms` $\le 1.5$ ms budget)**

### Gate 11C-3: 100% Bit-Exact Determinism Across 5 Runs
- **Configuration:** 100,000 grains, 5 separate runs from scratch with fixed seed.
- **Verification:** 64-bit FNV-1a state hash computed over neighbor counts, neighbor lists, positions, and grain IDs.
- **Results:**
  - Run 1 Hash: `0xa36c93921c6809c9`
  - Run 2 Hash: `0xa36c93921c6809c9`
  - Run 3 Hash: `0xa36c93921c6809c9`
  - Run 4 Hash: `0xa36c93921c6809c9`
  - Run 5 Hash: `0xa36c93921c6809c9`
  - **Verdict: PASSED (100% bit-exact across all 5 runs)**

### Gate 11C-4: Device Overflow Counter & Telemetry Tracking
- **Test:** Synthetic high-density cluster (64 grains placed in $0.0035$ m volume such that all 64 overlap, creating 63 potential neighbors per grain $> 32$).
- **Results:**
  - Dense overlapping grains: 64
  - Reported overflow count: **1,984** (`(63 - 32) * 64`)
  - Neighbor slots clamped: **YES (max 32)**
  - Memory corruption / out-of-bounds: **0 (intact)**
  - Telemetry tracking: Exposed via `NeighborStats::lastOverflowCount` and `GranularStats::neighborOverflowCount`.
  - **Verdict: PASSED (overflow explicitly counted, memory protected)**

### Gate 11C-5: Zero Regression in Default Effects Mode
- **Test:** `--matrix --frames 60` across all 36 workload combinations (capacities 1M, 5M, 10M; emitters 1, 8, 64; workloads ramp, recycle, saturation, burst).
- **Results:**
  - **36 / 36 workloads PASSED**
  - Peak memory bandwidth: **226.5 GB/s**
  - Median simulation latency: `0.193 ms` (1M ramp)
  - Zero performance degradation or regressions.
  - **Verdict: PASSED**

---

## 3. Key Technical Decisions & Learnings

1. **Separable Compilation & CUDA Kernel Linkage:**
   - In modular CUDA projects with `CUDA_SEPARABLE_COMPILATION ON`, headers defining `__global__` functions must declare them `static __global__ void` to ensure internal translation-unit linkage and avoid `nvlink` duplicate symbol collisions.
2. **Dynamic Bounding Box Reduction:**
   - Instead of static domain boundaries, a two-pass parallel reduction dynamically computes tight bounding boxes in $< 0.05$ ms, automatically sizing the grid cell count without wasting cell table memory.
3. **Radius Jitter Safety Margin in Cell Sizing:**
   - To guarantee zero false negatives under disordered packing, cell size is dimensioned by $h \ge 2 \cdot r_{\text{max}} + \delta_{\text{skin}}$, where $r_{\text{max}} = r \cdot (1 + \text{jitter})$. This strictly guarantees that any contacting pair lies within the 27-cell stencil.
4. **Zero-Copy SoA Permutation:**
   - Reordering the 7 SoA channels into scratch buffers and swapping pointers in `GranularPool` completely eliminates an expensive secondary write pass.
