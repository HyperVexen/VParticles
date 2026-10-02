#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace vparticles {

// ── Procedural Divergence-Free Curl Noise ─────────────────────────────
// Analytically divergence-free velocity field:
// v(x) = sum_i (k_i x a_i) * cos(k_i . x + omega * t + phi_i)
// Because k_i . (k_i x a_i) == 0, div(v) = 0 exactly.
__device__ __forceinline__ float3 computeCurlNoise(
    float3 pos,
    float time,
    float freq,
    float strength)
{
    if (strength <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float3 p = make_float3(pos.x * freq, pos.y * freq, pos.z * freq);

    // 4 orthogonal/quasi-orthogonal wave directions and basis vectors
    // Wave 1: k1 = (1, 1, 0), a1 = (0, 0, 1) -> w1 = k1 x a1 = (1, -1, 0)
    const float phi1 = p.x + p.y + time;
    const float c1 = cosf(phi1);
    float vx = c1;
    float vy = -c1;
    float vz = 0.0f;

    // Wave 2: k2 = (0, 1, 1), a2 = (1, 0, 0) -> w2 = (0, 1, -1)
    const float phi2 = p.y + p.z + time * 1.3f + 1.57f;
    const float c2 = cosf(phi2);
    vy += c2;
    vz -= c2;

    // Wave 3: k3 = (1, 0, 1), a3 = (0, 1, 0) -> w3 = (-1, 0, 1)
    const float phi3 = p.x + p.z + time * 0.7f + 3.14f;
    const float c3 = cosf(phi3);
    vx -= c3;
    vz += c3;

    // Wave 4: k4 = (1, -1, 1), a4 = (1, 1, 0) -> w4 = (-1, 1, 2)
    const float phi4 = (p.x - p.y + p.z) * 0.707f + time * 1.1f + 0.78f;
    const float c4 = cosf(phi4) * 0.5f;
    vx -= c4;
    vy += c4;
    vz += c4 * 2.0f;

    return make_float3(vx * strength, vy * strength, vz * strength);
}

} // namespace vparticles
