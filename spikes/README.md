# Phase 11A spikes

These targets are calibration tools, not production components. They do not link against `VParticlesCore` and must not alter the effects runtime. They are deliberately excluded from normal builds while Phase 11A is implemented on `master`.

## DEM reference

`Phase11ADemReference.cu` is an intentionally slow, all-pairs, gather-style spring-dashpot DEM reference. It has a ground-plane contact, capped Coulomb tangential force, tangential damping, and explicit linear damping. Its purpose is to generate small-N settling/calibration trajectories; it is not a performance baseline and it is not an XPBD implementation.

Build only this target:

```powershell
cmake --build out/build/x64-Release --target Phase11ADemReference
```

Reproducible initial settling run:

```powershell
.\out\build\x64-Release\Phase11ADemReference.exe --grains 5000 --frames 180 --substeps 8 --radius 0.025 --stiffness 25000 --damping 250 --friction 0.45 --tangential-damping 8 --linear-damping 2 --seed 11011 --csv phase11a_dem_5k.csv
```

The source prints mean/max height, kinetic energy, maximum overlap, and milliseconds per DEM substep. Preserve the CSV and exact command alongside any Phase 11A calibration interpretation.
