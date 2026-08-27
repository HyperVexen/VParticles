#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <sstream>
#include <string>

namespace vparticles {

inline void throwOnCudaError(
    cudaError_t result,
    const char* expression,
    const char* file,
    int line)
{
    if (result == cudaSuccess) {
        return;
    }

    std::ostringstream message;
    message << "CUDA call failed: " << expression << " at " << file << ':' << line
            << " (" << cudaGetErrorString(result) << ')';
    throw std::runtime_error(message.str());
}

} // namespace vparticles

#define VP_CUDA_CHECK(expression) \
    ::vparticles::throwOnCudaError((expression), #expression, __FILE__, __LINE__)
