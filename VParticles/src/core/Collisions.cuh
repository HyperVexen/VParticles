#pragma once

#include "VParticles/SimulationTypes.h"

#include <cuda_runtime.h>
#include <cmath>

namespace vparticles {

// ── Analytical Collision Handlers ─────────────────────────────────────

__device__ __forceinline__ void resolvePlaneCollision(
    float4& pos,
    float4& vel,
    const CollisionPlane& plane)
{
    const float radius = pos.w;
    const float dotP = (pos.x - plane.point.x) * plane.normal.x +
                       (pos.y - plane.point.y) * plane.normal.y +
                       (pos.z - plane.point.z) * plane.normal.z;

    if (dotP < radius) {
        // Penetration correction
        const float penetration = radius - dotP;
        pos.x += plane.normal.x * penetration;
        pos.y += plane.normal.y * penetration;
        pos.z += plane.normal.z * penetration;

        // Normal velocity component
        const float vDotN = vel.x * plane.normal.x +
                            vel.y * plane.normal.y +
                            vel.z * plane.normal.z;
        if (vDotN < 0.0f) {
            // Moving into the plane: reflect normal component, damp tangential with friction
            const float vNormX = plane.normal.x * vDotN;
            const float vNormY = plane.normal.y * vDotN;
            const float vNormZ = plane.normal.z * vDotN;

            const float vTanX = vel.x - vNormX;
            const float vTanY = vel.y - vNormY;
            const float vTanZ = vel.z - vNormZ;

            const float frictionFactor = fmaxf(0.0f, 1.0f - plane.friction);
            vel.x = vTanX * frictionFactor - vNormX * plane.bounce;
            vel.y = vTanY * frictionFactor - vNormY * plane.bounce;
            vel.z = vTanZ * frictionFactor - vNormZ * plane.bounce;
        }
    }
}

__device__ __forceinline__ void resolveSphereCollision(
    float4& pos,
    float4& vel,
    const CollisionSphere& sphere)
{
    const float radius = pos.w;
    const float dx = pos.x - sphere.center.x;
    const float dy = pos.y - sphere.center.y;
    const float dz = pos.z - sphere.center.z;
    const float distSqr = dx * dx + dy * dy + dz * dz;
    const float dist = sqrtf(fmaxf(1.0e-7f, distSqr));

    if (!sphere.invert) {
        // Exterior obstacle sphere: particles bounce off the outside
        const float targetDist = sphere.radius + radius;
        if (dist < targetDist) {
            const float invDist = 1.0f / dist;
            const float nx = dx * invDist;
            const float ny = dy * invDist;
            const float nz = dz * invDist;

            pos.x = sphere.center.x + nx * targetDist;
            pos.y = sphere.center.y + ny * targetDist;
            pos.z = sphere.center.z + nz * targetDist;

            const float vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
            if (vDotN < 0.0f) {
                const float vNormX = nx * vDotN;
                const float vNormY = ny * vDotN;
                const float vNormZ = nz * vDotN;

                const float frictionFactor = fmaxf(0.0f, 1.0f - sphere.friction);
                vel.x = (vel.x - vNormX) * frictionFactor - vNormX * sphere.bounce;
                vel.y = (vel.y - vNormY) * frictionFactor - vNormY * sphere.bounce;
                vel.z = (vel.z - vNormZ) * frictionFactor - vNormZ * sphere.bounce;
            }
        }
    } else {
        // Inverted containment sphere: particles are trapped inside
        const float targetDist = fmaxf(0.001f, sphere.radius - radius);
        if (dist > targetDist) {
            const float invDist = 1.0f / dist;
            // Normal points inwards towards center
            const float nx = -dx * invDist;
            const float ny = -dy * invDist;
            const float nz = -dz * invDist;

            pos.x = sphere.center.x - nx * targetDist;
            pos.y = sphere.center.y - ny * targetDist;
            pos.z = sphere.center.z - nz * targetDist;

            const float vDotN = vel.x * nx + vel.y * ny + vel.z * nz;
            if (vDotN < 0.0f) {
                const float vNormX = nx * vDotN;
                const float vNormY = ny * vDotN;
                const float vNormZ = nz * vDotN;

                const float frictionFactor = fmaxf(0.0f, 1.0f - sphere.friction);
                vel.x = (vel.x - vNormX) * frictionFactor - vNormX * sphere.bounce;
                vel.y = (vel.y - vNormY) * frictionFactor - vNormY * sphere.bounce;
                vel.z = (vel.z - vNormZ) * frictionFactor - vNormZ * sphere.bounce;
            }
        }
    }
}

__device__ __forceinline__ void resolveBoxCollision(
    float4& pos,
    float4& vel,
    const CollisionBox& box)
{
    const float radius = pos.w;
    const float frictionFactor = fmaxf(0.0f, 1.0f - box.friction);

    // X axis boundaries
    if (pos.x < box.minBounds.x + radius) {
        pos.x = box.minBounds.x + radius;
        if (vel.x < 0.0f) {
            vel.x = -vel.x * box.bounce;
            vel.y *= frictionFactor;
            vel.z *= frictionFactor;
        }
    } else if (pos.x > box.maxBounds.x - radius) {
        pos.x = box.maxBounds.x - radius;
        if (vel.x > 0.0f) {
            vel.x = -vel.x * box.bounce;
            vel.y *= frictionFactor;
            vel.z *= frictionFactor;
        }
    }

    // Y axis boundaries
    if (pos.y < box.minBounds.y + radius) {
        pos.y = box.minBounds.y + radius;
        if (vel.y < 0.0f) {
            vel.y = -vel.y * box.bounce;
            vel.x *= frictionFactor;
            vel.z *= frictionFactor;
        }
    } else if (pos.y > box.maxBounds.y - radius) {
        pos.y = box.maxBounds.y - radius;
        if (vel.y > 0.0f) {
            vel.y = -vel.y * box.bounce;
            vel.x *= frictionFactor;
            vel.z *= frictionFactor;
        }
    }

    // Z axis boundaries
    if (pos.z < box.minBounds.z + radius) {
        pos.z = box.minBounds.z + radius;
        if (vel.z < 0.0f) {
            vel.z = -vel.z * box.bounce;
            vel.x *= frictionFactor;
            vel.y *= frictionFactor;
        }
    } else if (pos.z > box.maxBounds.z - radius) {
        pos.z = box.maxBounds.z - radius;
        if (vel.z > 0.0f) {
            vel.z = -vel.z * box.bounce;
            vel.x *= frictionFactor;
            vel.y *= frictionFactor;
        }
    }
}

} // namespace vparticles
