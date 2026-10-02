# Phase 10A: Interactive OpenGL Viewer

**VParticles · Milestone Report**  
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (GA107, 20 SMs, SM 8.6, 96-bit bus, 168 GB/s physical memory bandwidth)  
**Host:** Windows x64, MSVC 19.51, CUDA Runtime 13.3, OpenGL 4.5.0 NVIDIA 610.88

---

## 1. Overview & Architecture

Phase 10A introduces an interactive, high-performance OpenGL viewer (`VParticlesViewer.exe`) that renders the CUDA particle simulation in real time using **zero-copy GPU buffer sharing**.

### Key Architectural Tenets
1. **Zero CPU Readback for Rendering:** The particle simulation, compaction, buffer gather, and indirect draw command generation execute entirely on the GPU. The CPU never stalls or waits for the GPU to submit draw calls.
2. **Modular Architecture:** The core simulation engine is compiled as `VParticlesCore` (static library) with strictly zero OpenGL dependencies. The headless benchmark `VParticles.exe` remains 100% compute-only.
3. **GPU-Driven Indirect Dispatch:** Particle counts are written directly to an OpenGL `GL_DRAW_INDIRECT_BUFFER` by a 1-thread CUDA bridge kernel.
4. **Point Sprite Rendering:** Antialiased circular particles with perspective size attenuation, smooth Gaussian-like alpha falloff, and additive/alpha blending modes.

```
┌────────────────────────────────────────────────────────────────────────┐
│                              CPU Host                                  │
│  - GLFW Window & Event Loop                                            │
│  - Arcball Orbit Camera (Mouse Drag / Scroll)                          │
│  - Real-Time Preset / Recipe Selector                                  │
│  - Real-Time HUD Telemetry (Window Title @ 10 Hz)                      │
└──────────────────┬─────────────────────────────────┬───────────────────┘
                   │ particleSystem.update(dt)       │ bridge.gather()
                   ▼                                 ▼
┌───────────────────────────────────┐    ┌───────────────────────────────┐
│     CUDA Simulation Pipeline      │    │    CUDA-OpenGL Bridge Kernel  │
│  - Graph Launch (Sim + Compact)   │    │  - Reads activeIndices[0..N]  │
│  - Free-List / Slot Allocation    │───►│  - Flattens pos & color to VBO│
│  - Multi-Recipe Constant Cache    │    │  - Writes count to IndirectCmd│
└───────────────────────────────────┘    └───────────────┬───────────────┘
                                                         │
                                                         ▼
                                         ┌───────────────────────────────┐
                                         │   OpenGL 4.5 Core Renderer    │
                                         │  - glDrawArraysIndirect       │
                                         │  - Perspective Point Sprites  │
                                         │  - Procedural Smoothstep AA   │
                                         │  - Additive / Alpha Blending  │
                                         └───────────────────────────────┘
```

---

## 2. Technical Implementation Details

### A. Zero-Copy CUDA-OpenGL Interop Bridge
In OpenGL 4.5, buffer storage is allocated via `glBufferData` and registered once with CUDA via `cudaGraphicsGLRegisterBuffer`:
```cpp
cudaGraphicsGLRegisterBuffer(&vboRes, vboId, cudaGraphicsRegisterFlagsWriteDiscard);
cudaGraphicsGLRegisterBuffer(&indirectRes, cmdBufferId, cudaGraphicsRegisterFlagsWriteDiscard);
```

Per frame:
1. Map OpenGL buffers: `cudaGraphicsMapResources(2, resources, stream)`.
2. Retrieve mapped GPU pointers:
   ```cpp
   cudaGraphicsResourceGetMappedPointer((void**)&d_outVertices, &vboBytes, vboRes);
   cudaGraphicsResourceGetMappedPointer((void**)&d_outCmd, &cmdBytes, indirectRes);
   ```
3. Launch gather kernel:
   ```cuda
   __global__ void gatherRenderKernel(
       const float4* __restrict__ pos,
       const float4* __restrict__ color,
       uint32_t* const* __restrict__ activeIndicesPtr,
       const uint32_t* __restrict__ aliveCountPtr,
       uint32_t maxCapacity,
       RenderVertex* __restrict__ outVertices,
       DrawArraysIndirectCommand* __restrict__ outCmd)
   {
       if (blockIdx.x == 0 && threadIdx.x == 0) {
           uint32_t alive = *aliveCountPtr;
           if (alive > maxCapacity) alive = maxCapacity;
           outCmd->count = alive;
           outCmd->instanceCount = 1;
           outCmd->first = 0;
           outCmd->baseInstance = 0;
       }

       uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       uint32_t alive = *aliveCountPtr;
       if (idx >= alive || idx >= maxCapacity) return;

       const uint32_t* activeIndices = *activeIndicesPtr;
       uint32_t slot = activeIndices[idx];

       float4 p = pos[slot];
       float4 c = color[slot];
       outVertices[idx] = RenderVertex{ p.x, p.y, p.z, p.w, c.x, c.y, c.z, c.w };
   }
   ```
4. Unmap resources: `cudaGraphicsUnmapResources(2, resources, stream)`.

### B. Shaders & Visual Pipeline
- **Vertex Shader:**
  - Projects particle to clip space: `gl_Position = uViewProj * vec4(in_Position.xyz, 1.0);`
  - Applies perspective distance attenuation:
    ```glsl
    float dist = length(uEyePos - in_Position.xyz);
    float pSize = ((in_Position.w * uPointScale) * uViewportHeight) / (dist * 1.5);
    gl_PointSize = clamp(pSize, 1.0, 256.0);
    ```
- **Fragment Shader:**
  - Evaluates radial distance in `gl_PointCoord`:
    ```glsl
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float distSq = dot(coord, coord);
    if (distSq > 1.0) discard;
    float alpha = smoothstep(1.0, 0.0, distSq) * vColor.a;
    fragColor = vec4(vColor.rgb * alpha, alpha);
    ```
  - Yields antialiased circular particles with zero rasterization artifacts.

---

## 3. Benchmark & Validation Results

Benchmarks run on NVIDIA GeForce RTX 3050 6GB Laptop GPU at 1600×900 resolution (uncapped framerate, `--no-vsync`):

| Live Particles | Capacity | VBO Size | Average FPS | Sim Time | Total Pipeline Time |
|---|---|---|---|---|---|
| **50,000** | 1,000,000 | 30 MB | **761.8 FPS** | 0.66 ms | 1.31 ms |
| **1,000,000** (Full Saturation) | 1,000,000 | 30 MB | **150.0 FPS** | 0.22 ms | 6.66 ms |
| **5,000,000** (Massive Scale) | 5,000,000 | 152 MB | **27.7 FPS** | 0.18 ms | 36.1 ms |

### Key Findings
1. **150 FPS at 1M Particles:** At 1,000,000 simultaneously simulated and rendered particles, the entire pipeline (simulation, compaction, buffer gather, indirect draw submission, rasterization, swap) executes in **6.66 ms**.
2. **Zero-Copy Performance:** The CUDA-GL gather kernel processes 1M particles in **~0.19 ms**, eliminating any host memory bottleneck.
3. **Headless Benchmark Parity:** `VParticles.exe` continues to achieve **0.217 ms** simulation p50 and **138.5 GB/s** peak memory bandwidth with zero regression.

---

## 4. Interactive Controls & Presets

`VParticlesViewer.exe` includes full interactive controls and 8 presets:

| Key / Input | Action |
|---|---|
| **Left Click + Drag** | Orbit camera (azimuth / elevation) |
| **Right Click + Drag** | Pan camera (view plane translation) |
| **Mouse Scroll** | Zoom camera distance |
| **C** | Reset camera view to default perspective |
| **Space** | Pause / resume simulation |
| **R** | Reset particle simulation |
| **B** | Toggle blend mode (Additive vs Alpha) |
| **+ / -** | Increase / decrease particle point size |
| **1** | Preset: Showroom (All 8 distinct effects arranged around a museum showroom ring) |
| **2** | Preset: Smoke Plume (Buoyant ground plume, rolling curl-noise turbulence, expanding ash) |
| **3** | Preset: Sparks & Ricochet (Elevated high-speed downward shower, dynamic floor bounce) |
| **4** | Preset: Bonfire & Deflector Sphere (Obstacle sphere deflector, flame curl wrap) |
| **5** | Preset: Grand Water Fountain (Central towering geyser, 6 arching peripheral jets, basin) |
| **6** | Preset: Plasma Vortex (Zero-G magnetic containment sphere, violent turbulent swirl) |
| **7** | Preset: Shrapnel & Shockwave (Crushing gravity, high-velocity ground ricochet dome) |
| **8** | Preset: Magic Shimmer Galaxy (Stardust double helix, ethereal low-gravity float) |
| **Tab** | Cycle sequentially through all 8 presets |
| **Escape** | Exit viewer |
