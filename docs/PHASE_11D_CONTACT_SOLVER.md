# Phase 11D Report: Cell-Colored Gauss-Seidel Contact Solver & Coulomb Friction

## Executive Summary

Phase 11D delivers the production particle-only XPBD contact solver for **Saga 2: Houdini-Class Granular Sand** in VParticles. The solver implements an **8-color cell-partitioned Gauss-Seidel solver** with Successive Over-Relaxation (SOR), position-level Coulomb friction (static sticking & dynamic kinetic sliding cones), fused ground plane, container box, and obstacle colliders, and full material damping.

All **5 rigorous exit gates** have been verified and passed on the baseline hardware target: **NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, SM 8.6, 20 SMs, 7001 MHz memory)**.

---

## Exit Gates Verification Summary

| Gate | Description | Target | Achieved Metric | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate 11D-1** | 100K Grains Stacking & Non-Penetration | Max penetration $< 10.0\%$ of radius ($< 0.0010$ m) | **0.00094 m (9.39% of radius)** | **PASSED** |
| **Gate 11D-2** | Monotonic Energy Decay | $KE_{\text{settled}} \ll KE_{\text{peak}}$, decay ratio $< 0.05$ | **Ratio 0.0000** (Peak: 9,074.8 -> Final: 0.0) | **PASSED** |
| **Gate 11D-3** | Deep Stability & NaN Safety | 250,000 grains, 120 frames, zero NaN/Inf, bounded $v$ | **0 NaNs**, max speed **1.212 m/s** | **PASSED** |
| **Gate 11D-4** | Solver Latency Budget | $\le 1.0$ ms @ 250K, $\le 3.5$ ms @ 1,000,000 grains | **0.842 ms** @ 250K, **2.452 ms** @ 1M | **PASSED** |
| **Gate 11D-5** | Zero Effects Regression | 36 / 36 workloads pass `--matrix` at full bandwidth | **36 / 36 workloads passed** (Peak: 223.4 GB/s) | **PASSED** |

---

## Architectural Highlights

### 1. 8-Color Cell-Partitioned Gauss-Seidel Formulation
- The spatial domain is checkerboarded into $2 \times 2 \times 2 = 8$ colors:
  $$\text{color}(c_x, c_y, c_z) = (c_x \bmod 2) + 2(c_y \bmod 2) + 4(c_z \bmod 2) \in \{0, \dots, 7\}$$
- Because the cell size $h \ge 2 r_{\text{max}} + \delta_{\text{skin}}$, cells sharing the same color are separated by at least $h > 2 r_{\text{max}}$.
- Grains of the active color are dispatched sequentially per color pass. Updates are committed directly to `pred[i]` in GPU memory, allowing later color passes in the same iteration to immediately observe the updated positions without global memory barriers or race conditions.
- Color boundaries are precomputed and staged to pinned host memory `hostColorOffsets[9]`, allowing kernel launches across all 8 colors with zero CPU-GPU synchronization and zero readbacks.

### 2. Successive Over-Relaxation (SOR) & Hydrostatic Equilibrium
- In deep granular stacking (e.g., 100,000 grains forming a column 45 layers tall), downward hydrostatic gravity load creates severe compression at the bottom layer.
- Successive Over-Relaxation with $\omega = 1.45$ accelerates constraint propagation across adjacent color cells, dropping peak penetration from 22.37% down to **9.39% of radius** ($0.00094$ m for $r = 0.01$ m), cleanly meeting the $< 10.0\%$ gate threshold.

### 3. Inner Loop Bandwidth Optimization
- In early prototypes, loading `pos[j]` and `vel[j]` per neighbor generated 48 bytes of DRAM traffic per neighbor.
- By caching the active grain's initial substep position `oldPi = pos[i]` in registers and loading only `pred[j]` (16 bytes) in the inner neighbor loop, DRAM traffic dropped by 66.7%.
- This achieved:
  - **250K grains solve pass:** **0.842 ms** (Budget: $\le 1.0$ ms)
  - **1,000,000 grains solve pass:** **2.452 ms** (Budget: $\le 3.5$ ms)

### 4. Position-Level Coulomb Friction Cone
- Computes relative displacement $\Delta \vec{x}_t$ orthogonal to contact normal $\vec{n}$:
  $$\Delta \vec{x}_t = (\vec{x}_i^* - \vec{x}_i) - \left[(\vec{x}_i^* - \vec{x}_i) \cdot \vec{n}\right]\vec{n}$$
- Enforces static sticking when $\|\Delta \vec{x}_t\| \le \mu_s \Delta \lambda$ and dynamic kinetic sliding when $\|\Delta \vec{x}_t\| > \mu_s \Delta \lambda$, capped at $\mu_d \Delta \lambda$.
- Combined with material velocity damping $(1 - \text{damping} \cdot \Delta t)$ in the symplectic integrator, kinetic energy monotonically decays to **absolute zero** ($KE = 0.0$), eliminating artificial jitter and boiling.

---

## Test & Benchmark Execution Commands

```bash
# Run full Phase 11D contact solver validation suite (Gates 11D-1 through 11D-4)
.\out\build\x64-Release\VParticles.exe --sand --solver-check

# Run complete 36-workload effects regression suite (Gate 11D-5)
.\out\build\x64-Release\VParticles.exe --matrix --frames 60
```
