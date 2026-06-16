#pragma once
#include "PerformanceStats.h"

// Polls NVIDIA GPU metrics via NVML.
// Designed to be called once per second from the main loop.
// Gracefully no-ops if NVML is unavailable (non-NVIDIA GPU, driver too old, etc.)
class GpuMonitor
{
public:
    GpuMonitor();
    ~GpuMonitor();

    // Call this once per second; fills the GPU fields in stats.
    void Poll(PerformanceStats& stats);

    bool IsAvailable() const { return m_available; }

private:
    bool     m_available = false;
    void*    m_device    = nullptr; // nvmlDevice_t (opaque handle)
};
