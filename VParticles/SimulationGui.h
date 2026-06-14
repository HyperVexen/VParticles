#pragma once

#include "PerformanceStats.h"
#include "SimulationSettings.h"

struct SimulationGuiResult
{
    bool resetRequested = false;
    bool velocityChanged = false;
};

class SimulationGui
{
public:
    SimulationGuiResult Draw(
        SimulationSettings& settings,
        const PerformanceStats& stats) const;
};
