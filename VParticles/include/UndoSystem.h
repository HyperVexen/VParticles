#pragma once
#include "SimulationSettings.h"
#include <vector>

class UndoSystem
{
public:
    void PushUndo(const SimulationSettings& s)
    {
        // Don't push duplicates
        if (!m_undoStack.empty() && m_undoStack.back() == s)
        {
            return;
        }
        m_undoStack.push_back(s);
        m_redoStack.clear(); // Clear redo stack on new action
        if (m_undoStack.size() > 100)
        {
            m_undoStack.erase(m_undoStack.begin());
        }
    }

    bool Undo(SimulationSettings& currentSettings)
    {
        if (m_undoStack.empty()) return false;
        m_redoStack.push_back(currentSettings);
        currentSettings = m_undoStack.back();
        m_undoStack.pop_back();
        return true;
    }

    bool Redo(SimulationSettings& currentSettings)
    {
        if (m_redoStack.empty()) return false;
        m_undoStack.push_back(currentSettings);
        currentSettings = m_redoStack.back();
        m_redoStack.pop_back();
        return true;
    }

    void Clear()
    {
        m_undoStack.clear();
        m_redoStack.clear();
    }

    bool CanUndo() const { return !m_undoStack.empty(); }
    bool CanRedo() const { return !m_redoStack.empty(); }

private:
    std::vector<SimulationSettings> m_undoStack;
    std::vector<SimulationSettings> m_redoStack;
};
