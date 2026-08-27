@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "BUILD_DIR=%PROJECT_ROOT%\out\build\x64-Release"

if defined VSCMD_VER goto find_cmake

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto find_cmake

for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_ROOT=%%I"
if not defined VS_ROOT goto find_cmake

if exist "%VS_ROOT%\Common7\Tools\VsDevCmd.bat" call "%VS_ROOT%\Common7\Tools\VsDevCmd.bat" -arch=x64

:find_cmake
where cmake >nul 2>nul
if not errorlevel 1 set "CMAKE_EXE=cmake"

if defined CMAKE_EXE goto configure
if not defined VS_ROOT goto cmake_missing

set "CMAKE_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if exist "%CMAKE_EXE%" goto configure

:cmake_missing
echo CMake was not found. Run this from a Visual Studio Developer Command Prompt or add CMake to PATH.
exit /b 1

:configure
set "CUDA_ARCH_ARGUMENT="
if defined VPARTICLES_CUDA_ARCHITECTURES set "CUDA_ARCH_ARGUMENT=-DCMAKE_CUDA_ARCHITECTURES=%VPARTICLES_CUDA_ARCHITECTURES%"

"%CMAKE_EXE%" -S "%PROJECT_ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release %CUDA_ARCH_ARGUMENT%
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" --build "%BUILD_DIR%"
exit /b %errorlevel%
