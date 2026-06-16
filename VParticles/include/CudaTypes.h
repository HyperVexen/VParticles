#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

struct CudaColor
{
    std::uint8_t r;
    std::uint8_t g;
    std::uint8_t b;
    std::uint8_t a;

    CUDA_CALLABLE constexpr CudaColor() : r(0), g(0), b(0), a(255) {}
    CUDA_CALLABLE constexpr CudaColor(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a=255) : r(r), g(g), b(b), a(a) {}
};
