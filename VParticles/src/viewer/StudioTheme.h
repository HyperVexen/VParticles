#pragma once

#include <imgui.h>

namespace vparticles::viewer {

inline void applyStudioTheme() {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // --- High-End JangaFX / Unreal-Inspired Charcoal Palette ---
    // Deep dark background, subtle cool-slate panels, cyan and amber particle accents
    const ImVec4 kBgDark         = ImVec4(0.075f, 0.078f, 0.090f, 1.000f); // #131417
    const ImVec4 kBgPanel        = ImVec4(0.106f, 0.110f, 0.125f, 1.000f); // #1b1c20
    const ImVec4 kBgWidget       = ImVec4(0.145f, 0.153f, 0.173f, 1.000f); // #25272c
    const ImVec4 kBgWidgetHover  = ImVec4(0.196f, 0.208f, 0.235f, 1.000f); // #32353c
    const ImVec4 kBgWidgetActive = ImVec4(0.247f, 0.263f, 0.298f, 1.000f); // #3f434c

    const ImVec4 kAccentCyan     = ImVec4(0.055f, 0.647f, 0.812f, 1.000f); // #0ea5cf (Awake/Active Cyan)
    const ImVec4 kAccentAmber    = ImVec4(0.961f, 0.620f, 0.043f, 1.000f); // #f59e0b (Sand Gold/Amber)
    const ImVec4 kAccentHover    = ImVec4(0.125f, 0.741f, 0.922f, 1.000f);

    const ImVec4 kTextPrimary    = ImVec4(0.922f, 0.929f, 0.941f, 1.000f);
    const ImVec4 kTextSecondary  = ImVec4(0.550f, 0.565f, 0.600f, 1.000f);
    const ImVec4 kBorderColor    = ImVec4(0.180f, 0.188f, 0.212f, 1.000f);

    colors[ImGuiCol_Text]                  = kTextPrimary;
    colors[ImGuiCol_TextDisabled]          = kTextSecondary;
    colors[ImGuiCol_WindowBg]              = kBgDark;
    colors[ImGuiCol_ChildBg]               = kBgPanel;
    colors[ImGuiCol_PopupBg]               = ImVec4(0.090f, 0.094f, 0.106f, 0.980f);
    colors[ImGuiCol_Border]                = kBorderColor;
    colors[ImGuiCol_BorderShadow]          = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
    colors[ImGuiCol_FrameBg]               = kBgWidget;
    colors[ImGuiCol_FrameBgHovered]        = kBgWidgetHover;
    colors[ImGuiCol_FrameBgActive]         = kBgWidgetActive;
    colors[ImGuiCol_TitleBg]               = kBgPanel;
    colors[ImGuiCol_TitleBgActive]         = ImVec4(0.125f, 0.133f, 0.153f, 1.000f);
    colors[ImGuiCol_TitleBgCollapsed]      = kBgDark;
    colors[ImGuiCol_MenuBarBg]             = kBgPanel;
    colors[ImGuiCol_ScrollbarBg]           = kBgDark;
    colors[ImGuiCol_ScrollbarGrab]         = kBgWidget;
    colors[ImGuiCol_ScrollbarGrabHovered]  = kBgWidgetHover;
    colors[ImGuiCol_ScrollbarGrabActive]   = kBgWidgetActive;
    colors[ImGuiCol_CheckMark]             = kAccentCyan;
    colors[ImGuiCol_SliderGrab]            = kAccentCyan;
    colors[ImGuiCol_SliderGrabActive]      = kAccentHover;
    colors[ImGuiCol_Button]                = kBgWidget;
    colors[ImGuiCol_ButtonHovered]         = kBgWidgetHover;
    colors[ImGuiCol_ButtonActive]          = kBgWidgetActive;
    colors[ImGuiCol_Header]                = ImVec4(0.157f, 0.165f, 0.188f, 1.000f);
    colors[ImGuiCol_HeaderHovered]         = kBgWidgetHover;
    colors[ImGuiCol_HeaderActive]          = kBgWidgetActive;
    colors[ImGuiCol_Separator]             = kBorderColor;
    colors[ImGuiCol_SeparatorHovered]      = kAccentCyan;
    colors[ImGuiCol_SeparatorActive]       = kAccentCyan;
    colors[ImGuiCol_ResizeGrip]            = ImVec4(0.20f, 0.22f, 0.25f, 0.20f);
    colors[ImGuiCol_ResizeGripHovered]     = kAccentCyan;
    colors[ImGuiCol_ResizeGripActive]      = kAccentHover;
    colors[ImGuiCol_Tab]                   = kBgPanel;
    colors[ImGuiCol_TabHovered]            = kBgWidgetHover;
    colors[ImGuiCol_TabSelected]           = ImVec4(0.145f, 0.153f, 0.176f, 1.000f);
    colors[ImGuiCol_TabDimmed]             = kBgDark;
    colors[ImGuiCol_TabDimmedSelected]     = kBgPanel;
    colors[ImGuiCol_DockingPreview]        = ImVec4(0.055f, 0.647f, 0.812f, 0.350f);
    colors[ImGuiCol_DockingEmptyBg]        = kBgDark;
    colors[ImGuiCol_PlotLines]             = kAccentCyan;
    colors[ImGuiCol_PlotLinesHovered]      = kAccentAmber;
    colors[ImGuiCol_PlotHistogram]         = kAccentAmber;
    colors[ImGuiCol_PlotHistogramHovered]  = kAccentCyan;
    colors[ImGuiCol_TableHeaderBg]         = kBgPanel;
    colors[ImGuiCol_TableBorderStrong]     = kBorderColor;
    colors[ImGuiCol_TableBorderLight]      = ImVec4(0.140f, 0.155f, 0.165f, 1.000f);
    colors[ImGuiCol_TableRowBg]            = ImVec4(0.000f, 0.000f, 0.000f, 0.000f);
    colors[ImGuiCol_TableRowBgAlt]         = ImVec4(1.000f, 1.000f, 1.000f, 0.020f);
    colors[ImGuiCol_TextSelectedBg]        = ImVec4(0.055f, 0.647f, 0.812f, 0.350f);
    colors[ImGuiCol_DragDropTarget]        = kAccentAmber;
    colors[ImGuiCol_NavCursor]             = kAccentCyan;

    // --- Crisp, Modern Roundings & Layout Metrics ---
    style.WindowPadding     = ImVec2(10.0f, 10.0f);
    style.FramePadding      = ImVec2(8.0f, 5.0f);
    style.CellPadding       = ImVec2(6.0f, 4.0f);
    style.ItemSpacing       = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing  = ImVec2(6.0f, 4.0f);
    style.TouchExtraPadding = ImVec2(0.0f, 0.0f);
    style.IndentSpacing     = 18.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 10.0f;

    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 0.0f;
    style.TabBorderSize     = 0.0f;

    style.WindowRounding    = 4.0f;
    style.ChildRounding     = 4.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 4.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 3.0f;
    style.TabRounding       = 4.0f;

    style.WindowMenuButtonPosition = ImGuiDir_None;
}

} // namespace vparticles::viewer
