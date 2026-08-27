# Contributing To VParticles

## Development Focus

VParticles is compute-first. Keep rendering, Blender, editor, and DCC integration outside the simulation core until the CUDA path has passed the required scale and profiling milestones.

Prefer the existing design rules:

- One global pool rather than one allocation per effect.
- Structure-of-arrays state and fused simulation passes.
- GPU-resident data where possible.
- Simple, measured lifecycle strategies over theoretical optimizations.

## Local Build

Use a Visual Studio Developer Command Prompt or Developer PowerShell:

```powershell
.\build.bat
```

Run the standard baseline before changing performance-sensitive code:

```powershell
.\out\build\x64-Release\VParticles.exe --capacity 1000000 --frames 240 --spawn-rate 250000
```

Use longer sustained-recycle and multi-emitter runs when changing free-list, active-index, compaction, sorting, or spawn-command code.

## Pull Requests

- Keep changes focused on one milestone.
- Explain the data-layout and memory-traffic consequences of simulation changes.
- Include the exact benchmark command, GPU environment, and before/after timing when performance can change.
- Update `ROADMAP.md`, architecture notes, or the status report when behavior or milestones change.
- Do not add renderer, UI, Blender, or DCC dependencies to the compute core.

The PR template provides the expected handoff structure.
