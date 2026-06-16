#pragma once
#include "PerformanceStats.h"
#include "SimulationSettings.h"
#include "UndoSystem.h"

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
        UndoSystem& undoSystem,
        const PerformanceStats& stats,
        const SimulationStats& simStats);

private:
    bool m_wasAnyItemActive = false;
    SimulationSettings m_settingsBeforeEdit;
};
