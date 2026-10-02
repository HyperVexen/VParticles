# VParticles

VParticles is a high-scale CUDA particle simulation engine for game VFX. It is currently a compute-only project: no renderer, Blender bridge, editor UI, or external DCC dependency is part of the engine core.

The target is 10M+ live particles on NVIDIA GPUs while keeping the simulation bandwidth-aware and measurable.

## Current Status

The current engine provides:

- One shared FP32 structure-of-arrays particle pool with runtime dual-mode support for Packed SoA storage (28 bytes/particle in pool, ~2.3x speedup).
- Multi-tile large-world spatial scale: up to 256 world-space tiles in CUDA constant memory with zero-copy tile migration.
- Deterministic spawn variation without persistent per-particle RNG state.
- Fused gravity, wind, drag, Euler integration, age, lifetime kill, and alpha fade.
- Fused divergence-free 3D curl-noise turbulence.
- Fused analytical collision solvers: ground/arbitrary planes, bounding/obstacle spheres, and boxes with bounce restitution and surface friction.
- Fused size-over-life and color-over-life curves with zero extra DRAM bandwidth overhead.
- Dense active indices plus a persistent GPU free-list for particle lifecycle management.
- GPU-resident active counters and block-scan active-index compaction.
- A batched spawn-command buffer: one GPU spawn launch per frame, independent of emitter count.
- A delayed telemetry ring for per-stage GPU timing, lifecycle statistics, and renderer-ready tile metadata.
- Automated benchmark matrix with four workload types, timing percentiles (p50/p95/p99), and effective memory bandwidth.
- CUDA Graph replay for the steady-state compute sequence, with a safe eager A/B fallback.
- Data-driven effect recipes: up to 64 concurrent `EffectRecipe` profiles in constant memory with per-particle assignment, single-recipe zero-overhead fast path, and zero graph rebuilds on recipe parameter tuning.
- Interactive OpenGL viewer (`VParticlesViewer.exe`): zero-copy CUDA-OpenGL buffer interop via `cudaGraphicsGLRegisterBuffer`, GPU-resident indirect draw dispatch (`glDrawArraysIndirect`), 150 FPS at 1M particles, 8 visual presets, and Arcball camera controls.
- Modern Engine Interop (`VParticlesD3D12InteropTest.exe`): zero-copy GPU memory sharing with DirectX 12 (`ID3D12Resource` NT handles) and Vulkan, lockless hardware timeline synchronization (`cudaWaitExternalSemaphoresAsync` / `cudaSignalExternalSemaphoresAsync`), and GPU-driven `D3D12DrawArguments` emission executing at 3,564 FPS.

The detailed implementation report and latest validation notes are in [docs/PROJECT_STATUS_REPORT.md](docs/PROJECT_STATUS_REPORT.md). The planned work is in [ROADMAP.md](ROADMAP.md).
CUDA Graph design, timing semantics, and measured A/B results are documented in [docs/PHASE_8_CUDA_GRAPHS.md](docs/PHASE_8_CUDA_GRAPHS.md).
Phase 9 data-driven effects architecture and validation are in [docs/PHASE_9_DATA_DRIVEN_EFFECTS.md](docs/PHASE_9_DATA_DRIVEN_EFFECTS.md).
Phase 10A OpenGL interactive viewer architecture and benchmarks are in [docs/PHASE_10A_OPENGL_VIEWER.md](docs/PHASE_10A_OPENGL_VIEWER.md).
Phase 10B Modern Engine Interop architecture and Unreal Engine 5 integration guide are in [docs/PHASE_10B_MODERN_ENGINE_INTEROP.md](docs/PHASE_10B_MODERN_ENGINE_INTEROP.md).

## Benchmarking Note

The runtime keeps frame-control counters on the GPU and publishes host-visible statistics through delayed telemetry. `update()` does not wait for exact per-frame stats. The benchmark executable calls `synchronize()` only at the reporting boundary so the final line is an exact host-visible snapshot.

The `--matrix` mode iterates workload types, capacities, and emitter counts, printing timing percentiles and effective simulation bandwidth for each scenario. GPU model, VRAM, compute capability, and CUDA driver version are printed at startup.

## Architecture

```text
simulate active particles
  -> apply forces (gravity, wind, drag)
  -> compute divergence-free curl noise
  -> integrate positions (dt)
  -> resolve plane, sphere, box collisions
  -> evaluate color and size curves
  -> return dead slots to a GPU free-list
  -> compact active indices on the GPU when deaths occur
  -> upload batched spawn commands (pinned host ring)
  -> reserve and spawn into reclaimed slots
  -> publish delayed telemetry
```

Particle attributes stay in stable SoA slots. The active list is compacted as `uint32_t` indices, avoiding a full gather of position, velocity, color, and lifetime data whenever particles die.

## Prerequisites

- Windows with an NVIDIA GPU and CUDA driver.
- CUDA Toolkit with a CUDA-compatible MSVC toolchain.
- Visual Studio C++ build tools with CMake and Ninja, or CMake and Ninja available on `PATH`.

The project uses C++20, CUDA C++17, and CMake's `native` CUDA architecture setting by default. Override `CMAKE_CUDA_ARCHITECTURES` when building for a specific target GPU.

## Build

From a Visual Studio Developer Command Prompt or Developer PowerShell:

```powershell
.\build.bat
```

Equivalent explicit commands:

```powershell
cmake -S . -B out/build/x64-Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build out/build/x64-Release
```

If CMake cannot locate CUDA automatically, set `CUDAToolkit_ROOT` during configure:

```powershell
cmake -S . -B out/build/x64-Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="D:\CUDA"
```

If the installed CUDA toolkit is newer than the NVIDIA driver's supported PTX version, build SASS for the target GPU rather than relying on runtime PTX JIT. For example, an `sm_86` GPU can use:

```powershell
$env:VPARTICLES_CUDA_ARCHITECTURES = "86-real"
.\build.bat
```

The equivalent direct CMake argument is `-DCMAKE_CUDA_ARCHITECTURES=86-real`.

## Benchmark

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 1000000 --frames 240 --spawn-rate 250000
```

Add `--emitters` to split the total spawn rate among multiple emitters while preserving a single spawn dispatch:

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 10000000 --frames 360 --emitters 64 --spawn-rate 2500000
```

Run with fused simulation modules (turbulence, collisions, curves):

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 1000000 --frames 120 --turbulence 5.0 --ground-plane 0.0 --curves
```

Run in Packed State Mode (28 bytes/particle storage, ~2.3x simulate speedup, scale up to 20M+ particles):

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 10000000 --frames 120 --packed
```

Run multi-tile simulation across large-world origins (e.g. 16 tiles, 64 emitters):

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 10000000 --frames 120 --emitters 64 --tiles 16 --packed
```

Compare CUDA Graph replay with conventional eager submission:

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 10000000 --frames 120 --packed
.\out\build\x64-Release\VParticles.exe --capacity 10000000 --frames 120 --packed --no-graph
```

Launch the interactive OpenGL Viewer:

```powershell
# Interactive viewer at 1M particles (presets, orbit/pan/zoom camera, HUD telemetry)
.\out\build\x64-Release\VParticlesViewer.exe --capacity 1000000

# High-scale uncapped viewer benchmark at 5M particles
.\out\build\x64-Release\VParticlesViewer.exe --capacity 5000000 --no-vsync
```

Available arguments:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--capacity` | 1,000,000 | Maximum particle slots in the global pool. |
| `--frames` | 240 | Number of fixed-timestep updates. |
| `--emitters` | 1 | Number of emitters sharing the total spawn rate. |
| `--spawn-rate` | 250,000 | Total requested particles per second. |
| `--dt` | 1/60 | Fixed timestep in seconds. |
| `--workload` | `ramp` | Workload type: `ramp`, `recycle`, `saturation`, `burst`. |
| `--matrix-workloads` | `ramp,recycle,saturation,burst` | Comma-separated workload types for `--matrix` mode. |
| `--turbulence` | `0.0` | Divergence-free curl noise strength (0 = disabled). |
| `--turb-freq` | `0.2` | Spatial frequency for curl noise. |
| `--ground-plane` | `0.0` | Enable ground collision plane at optional height `y`. |
| `--sphere-collision` | `5.0` | Enable sphere obstacle collision at center `(0,5,0)` with radius `r`. |
| `--box-collision` | `false` | Enable bounding AABB box collision. |
| `--curves` | `false` | Enable color-over-life and size-over-life curves. |
| `--packed` | `false` | Enable 28-byte/particle compressed SoA layout. |
| `--no-graph` | `false` | Disable CUDA Graph replay and submit the update sequence eagerly. |
| `--tiles` | `1` | Number of simulation tiles arranged in world space (up to 256). |
| `--tile-scale` | `100.0` | Quantization scale factor (local int16 units per world unit). |

### Workload Types

The `--workload` argument selects the particle lifecycle profile. In `--matrix` mode all four workloads are iterated by default.

| Workload | Spawn strategy | Lifetime | Behaviour |
| --- | --- | --- | --- |
| `ramp` | `capacity / (frames × dt)` | 4.0 s | Gradual fill over the run |
| `recycle` | `capacity` per second | 1.0 s | Fill in ~1 s, then sustained churn |
| `saturation` | 4× ramp rate | 2× run duration | Fill fast, rest is all drops |
| `burst` | 0 continuous | 2.0 s | Burst capacity/4 every 30 frames |

## Project Layout

```text
VParticles/include/VParticles/  Public engine types and CUDA error helper
VParticles/src/                 CUDA simulation and benchmark executable
docs/                           Architecture, benchmark, and status documents
ROADMAP.md                      Engineering milestones and sequencing
```

## Deliberate Non-Goals Today

- Rendering, renderer interop, and UI.
- Blender or other DCC integrations.
- Per-effect GPU buffers and per-emitter spawn launches.
- Multi-GPU scaling before single-GPU spatial hierarchy is proven.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Performance changes should include a reproducible benchmark command and relevant timing output.
