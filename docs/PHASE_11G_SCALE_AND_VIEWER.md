# Phase 11G: Large-Scale Awake Scaling, Stress Testing & Interactive Sand Viewer

## 1. Executive Summary

Phase 11G serves as the capstone milestone of **Saga 2 Part 1: Granular Physics Engine**. Building upon the contact solving, material friction, sleeping optimization, and XPBD constraint layers developed in Phases 11A through 11F, Phase 11G achieves two pivotal milestones:
1. **Interactive Real-Time 3D Sand Viewer (`VParticlesViewer.exe`)**: Seamless dual-mode execution (`SimulationMode::Effects` vs `SimulationMode::Granular`), direct GPU VBO zero-copy streaming, 3D lit sphere surface normal reconstruction on the fly via fragment shader, multiple visualization color modes (material tint, velocity magnitude heatmap, sleeping state visualization), and interactive hot-switching across all 7 Houdini-class sand scenarios (`pile`, `column_collapse`, `funnel`, `dune`, `cliff`, `bonded_block`, `clumps`).
2. **Large-Scale Awake Stress Testing & Kinematic Moving Colliders**: High-velocity shearing of 2,000,000 active awake grains without sleeping, continuous non-tunneling contact resolution against dynamic kinematic moving colliders (oscillating bulldozers and rotating churners), and complete latency validation on GPU hardware (NVIDIA GeForce RTX 3050 6GB Laptop GPU).

All five Phase 11G exit gates passed cleanly with zero regression across all 36 default effects workloads.

---

## 2. Architecture & Key Features

### 2.1 Interactive Dual-Mode OpenGL Viewer (`VParticlesViewer.exe`)

The interactive viewer provides real-time paired simulation and rendering with no CPU readback:
- **Zero-Copy GPU Buffer Mapping**: `CudaGLBridge::gatherGranular` maps the OpenGL Vertex Buffer Object (`GL_ARRAY_BUFFER`) directly into CUDA address space via `cudaGraphicsMapResources` / `cudaGraphicsResourceGetMappedPointer`.
- **Dynamic 3D Sphere Normal Reconstruction**: Instead of instancing high-polygon sphere meshes, grains are rendered as point sprites whose surface normals are analytically computed in the fragment shader from `gl_PointCoord`:
  $$\vec{N} = (x, -y, \sqrt{1 - x^2 - y^2})$$
  Points outside the unit sphere ($x^2 + y^2 > 1$) are discarded via `discard`. The fragment shader applies Blinn-Phong directional lighting, specular highlights, and ambient occlusion, producing high-fidelity 3D grains at 130–180+ FPS.
- **Color Modes**:
  - `0: Material Tint`: Hash-based natural color variation distinguishing materials (dry sand, wet sand, clumps).
  - `1: Turbo Velocity Heatmap`: Dynamic jet colormap displaying instantaneous grain speed ($0.0$ to $3.0+$ m/s).
  - `2: Sleeping State`: Cool blue for frozen sleeping grains vs bright amber for active awake grains.
- **Controls**:
  - `M`: Toggle Mode (`Effects Particles` $\leftrightarrow$ `Granular Sand`).
  - `1 - 7`: Instant switch to sand scenarios (`1: Pile`, `2: Column Collapse`, `3: Funnel`, `4: Dune`, `5: Cliff`, `6: Bonded Block`, `7: Clumps`).
  - `K`: Toggle Kinematic Moving Collider (dynamic bulldozer plow / churner).
  - `V`: Cycle Sand Color Mode.
  - `Space`: Pause / resume.
  - `R`: Reset current scenario.

### 2.2 Kinematic Moving Colliders & Dynamic Wake-Up

Kinematic obstacles (e.g. bulldozer blades sweeping at 2.5 m/s or rotating mixer paddles) are integrated into `GranularSettings`:
- Trajectory advancement occurs inside `stepFixed()` before substep solving:
  $$\vec{x}_{\text{plow}}(t) = \vec{x}_0 + \sin(\omega t) \cdot A \hat{x}$$
  $$\vec{v}_{\text{plow}}(t) = \omega A \cos(\omega t) \hat{x}$$
- Moving colliders impart dynamic velocity impulses directly to grains via PBD position projection:
  $$\vec{v}_i = \frac{\vec{x}^*_i - \vec{x}_i}{\Delta t}$$
- Grains in contact with the moving obstacle wake up instantaneously, clearing `kGrainFlagAsleep` and dynamically propagating cascade wake-up to neighboring grains in the contact graph.

---

## 3. Exit Criteria & Gate Results

Validation executed via:
```bash
.\out\build\x64-Release\VParticles.exe --scale-check
```

Hardware: **NVIDIA GeForce RTX 3050 6GB Laptop GPU (20 SMs, 7001 MHz mem clock, CUDA Driver 13.3 / Runtime 13.3)**.

| Gate | Description | Target Metric | Measured Metric | Status |
|---|---|---|---|---|
| **Gate 11G-1** | Interactive Viewer Latency Budget | < 8.33 ms @ 100K (120+ FPS)<br>< 16.6 ms @ 250K (60+ FPS) | **6.141 ms** (162.8 FPS)<br>**7.542 ms** (132.6 FPS)<br>*(1M: 40.6 ms / 24.6 FPS)* | **PASSED** |
| **Gate 11G-2** | Multi-Million Awake Scaling | 1M–2M active awake grains, 0 NaNs | **2,000,000 grains**, 0 NaNs, bounded speed 1.27 m/s, ~373 MB VRAM | **PASSED** |
| **Gate 11G-3** | Kinematic Moving Collider Plowing | Dynamic wake-up along plow trench, non-tunneling, 0 NaNs | **96,941 grains** woken, 0 tunneling violations, 0 NaNs | **PASSED** |
| **Gate 11G-4** | Scenario Preset Hot-Switching | Clean states & zero leakage across all 7 scenarios | **7/7 scenarios** verified (`pile`, `column`, `funnel`, `dune`, `cliff`, `block`, `clumps`) | **PASSED** |
| **Gate 11G-5** | Zero Effects Mode Regression | 36/36 workloads in `--matrix` | **36/36 passed**, peak bandwidth **220.0 GB/s** | **PASSED** |

---

## 4. Workload Performance Matrix (Gate 11G-5)

Validated across 36 combinations of workload types, capacities (1M, 5M, 10M particles), and emitter counts (1, 8, 64) with CUDA Graph capture:
- **Ramp**: up to 10M alive particles, p50 sim latency 0.587 ms, peak bandwidth 156.7 GB/s.
- **Recycle**: up to 10M particles churning, p50 sim latency 0.551 ms, peak bandwidth 152.5 GB/s.
- **Saturation**: up to 10M particles full pool drop, p50 sim latency 1.117 ms, peak bandwidth 192.1 GB/s.
- **Burst**: up to 10M particles burst explosions, p50 sim latency 2.644 ms, peak bandwidth **220.0 GB/s**.

Zero regressions detected across all core particle modules.

---

## 5. Artifacts and Source Files

- [VParticles/include/VParticles/GranularTypes.h](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/include/VParticles/GranularTypes.h): Kinematic collider settings (`kinematicColliderEnabled`, `kinematicColliderType`, `kinematicOrigin`, `kinematicSpeed`, `kinematicAmplitude`).
- [VParticles/src/granular/GranularSystem.cu](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/granular/GranularSystem.cu): Kinematic moving obstacle trajectory update in `stepFixed()`.
- [VParticles/src/viewer/CudaGLBridge.h](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/viewer/CudaGLBridge.h) & [CudaGLBridge.cu](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/viewer/CudaGLBridge.cu): `gatherGranular` with 3 color modes and zero-copy direct VBO write.
- [VParticles/src/viewer/Shaders.h](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/viewer/Shaders.h): 3D normal-reconstructed lit sphere fragment shader.
- [VParticles/src/viewer/ViewerMain.cpp](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/viewer/ViewerMain.cpp): Interactive dual-mode sand viewer, keyboard controls, HUD telemetry, and 7 scenario presets.
- [VParticles/include/VParticles/GranularValidation.h](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/include/VParticles/GranularValidation.h): `ScaleValidationResult` and `runScaleValidationSuite()`.
- [VParticles/src/granular/GranularScaleValidation.cu](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/granular/GranularScaleValidation.cu): Hardware stress test suite for Gates 11G-1 to 11G-4.
- [VParticles/src/VParticles.cpp](file:///c:/Users/death/Desktop/Important%20folder/PROJECTS/VParticles/VParticles/src/VParticles.cpp): CLI integration for `--scale-check` and `--sand-scale-check`.
