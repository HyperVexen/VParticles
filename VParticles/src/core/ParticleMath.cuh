#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace vparticles {

__device__ __forceinline__ uint32_t mixBits(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

__device__ __forceinline__ float random01(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    const uint32_t mixed = mixBits(seed ^ (particle * 0x9E3779B9u) ^ (frame * 0x85EBCA6Bu) ^ lane);
    return static_cast<float>(mixed & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float randomSigned(uint32_t seed, uint32_t particle, uint32_t frame, uint32_t lane)
{
    return random01(seed, particle, frame, lane) * 2.0f - 1.0f;
}

__device__ __forceinline__ int16_t encodePosition(float worldPos, float tileOrigin, float scale)
{
    float local = (worldPos - tileOrigin) * scale;
    local = fminf(32767.0f, fmaxf(-32768.0f, local));
    return static_cast<int16_t>(__float2int_rn(local));
}

__device__ __forceinline__ float decodePosition(int16_t packed, float tileOrigin, float invScale)
{
    return tileOrigin + static_cast<float>(packed) * invScale;
}

__device__ __forceinline__ uint32_t encodeColorRGBA8(float r, float g, float b, float a)
{
    uint32_t ri = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, r)) * 255.0f));
    uint32_t gi = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, g)) * 255.0f));
    uint32_t bi = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, b)) * 255.0f));
    uint32_t ai = static_cast<uint32_t>(__float2uint_rn(fminf(1.0f, fmaxf(0.0f, a)) * 255.0f));
    return ri | (gi << 8) | (bi << 16) | (ai << 24);
}

__device__ __forceinline__ float4 decodeColorRGBA8(uint32_t packed)
{
    float r = static_cast<float>(packed & 0xFFu) * (1.0f / 255.0f);
    float g = static_cast<float>((packed >> 8) & 0xFFu) * (1.0f / 255.0f);
    float b = static_cast<float>((packed >> 16) & 0xFFu) * (1.0f / 255.0f);
    float a = static_cast<float>((packed >> 24) & 0xFFu) * (1.0f / 255.0f);
    return make_float4(r, g, b, a);
}

} // namespace vparticles
