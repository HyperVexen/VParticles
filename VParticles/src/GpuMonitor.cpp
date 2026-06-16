#include "GpuMonitor.h"
#include <nvml.h>
#include <cstdio>

GpuMonitor::GpuMonitor()
{
    nvmlReturn_t ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS)
    {
        fprintf(stderr, "[GpuMonitor] nvmlInit failed: %s\n", nvmlErrorString(ret));
        m_available = false;
        return;
    }

    nvmlDevice_t device;
    ret = nvmlDeviceGetHandleByIndex_v2(0, &device);
    if (ret != NVML_SUCCESS)
    {
        fprintf(stderr, "[GpuMonitor] nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(ret));
        nvmlShutdown();
        m_available = false;
        return;
    }

    m_device    = reinterpret_cast<void*>(device);
    m_available = true;
}

GpuMonitor::~GpuMonitor()
{
    if (m_available)
    {
        nvmlShutdown();
    }
}

void GpuMonitor::Poll(PerformanceStats& stats)
{
    if (!m_available) return;

    nvmlDevice_t device = reinterpret_cast<nvmlDevice_t>(m_device);

    // GPU Utilization
    nvmlUtilization_t util{};
    if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS)
    {
        stats.gpuUtilPercent = util.gpu;
    }

    // VRAM
    nvmlMemory_t mem{};
    if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS)
    {
        stats.gpuMemUsedMb  = static_cast<uint32_t>(mem.used  / (1024 * 1024));
        stats.gpuMemTotalMb = static_cast<uint32_t>(mem.total / (1024 * 1024));
    }

    // Temperature
    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS)
    {
        stats.gpuTempC = static_cast<float>(temp);
    }
}
