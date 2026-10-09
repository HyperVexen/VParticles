#pragma once

#include "VParticles/GranularTypes.h"
#include "VParticles/NeighborTypes.h"
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace vparticles {

class IContactSolver {
public:
    virtual ~IContactSolver() = default;
    virtual const char* name() const = 0;
    virtual void solve(
        GranularPool& pool,
        const NeighborData& neighbors,
        const GranularSettings& settings,
        float substepDt,
        cudaStream_t stream) = 0;
};

// Factory functions
std::unique_ptr<IContactSolver> createJacobiContactSolver();
std::unique_ptr<IContactSolver> createGaussSeidelContactSolver();

} // namespace vparticles
