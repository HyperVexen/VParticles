# VParticles

VParticles is a high-scale CUDA particle simulation engine for game VFX. It is currently a compute-only project: no renderer, Blender bridge, editor UI, or external DCC dependency is part of the engine core.

The target is 10M+ live particles on NVIDIA GPUs while keeping the simulation bandwidth-aware and measurable.

## Current Status

The current engine provides:

- One shared FP32 structure-of-arrays particle pool.
- Deterministic spawn variation without persistent per-particle RNG state.
- Fused gravity, wind, drag, Euler integration, age, lifetime kill, and alpha fade.
- Dense active indices plus a persistent GPU free-list for particle lifecycle management.
- CUB-based active-index compaction and high-occupancy locality repair.
- A batched spawn-command buffer: one GPU spawn launch per frame, independent of emitter count.
- A headless benchmark with per-stage GPU timing and lifecycle statistics.

The detailed implementation report and latest benchmark are in [docs/PROJECT_STATUS_REPORT.md](docs/PROJECT_STATUS_REPORT.md). The planned work is in [ROADMAP.md](ROADMAP.md).

## Latest Measured Result

The latest recorded 1M-capacity benchmark used one emitter at 250,000 particles/second and a 1/60 second timestep.

| Frame | Live particles | Simulate | Lifecycle | Total |
| --- | ---: | ---: | ---: | ---: |
| 239 | 951,364 | 0.861 ms | 0.191 ms | 1.099 ms |

This is a near-capacity ramp result, not a substitute for sustained recycle testing. GPU model, clocks, driver version, and effect workload all affect the result.

## Architecture

```text
simulate active particles
  -> return dead slots to a GPU free-list
  -> compact active indices when deaths occur
  -> repair index locality at high occupancy
  -> upload batched spawn commands
  -> spawn into reclaimed slots
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

Available arguments:

| Argument | Default | Meaning |
| --- | ---: | --- |
| `--capacity` | 1,000,000 | Maximum particle slots in the global pool. |
| `--frames` | 240 | Number of fixed-timestep updates. |
| `--emitters` | 1 | Number of emitters sharing the total spawn rate. |
| `--spawn-rate` | 250,000 | Total requested particles per second. |
| `--dt` | 1/60 | Fixed timestep in seconds. |

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
- Packed state formats before the FP32 lifecycle baseline is profiled.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Performance changes should include a reproducible benchmark command and relevant timing output.
