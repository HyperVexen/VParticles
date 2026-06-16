#pragma once

#include "PerformanceStats.h"
#include "SimulationSettings.h"

struct SimulationGuiResult
{
    bool resetRequested = false;
    bool velocityChanged = false;
    bool particleCountChanged = false;
    bool switchToConsoleRequested = false;
};

class SimulationGui
{
public:
    SimulationGuiResult Draw(
        SimulationSettings& settings,
        const PerformanceStats& stats,
        const SimulationStats& simStats) const;
};
