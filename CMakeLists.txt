cmake_minimum_required(VERSION 3.18)
project(VParticles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

set(VCPKG_ROOT "C:/Users/death/Desktop/Important folder/LIBRARIES/VCPKG/installed/x64-windows")

include_directories(
    "${VCPKG_ROOT}/include"
    "D:/CUDA/include"   # nvml.h, cuda_gl_interop.h for non-.cu files
)

# Gather source files
file(GLOB SOURCES "VParticles/*.cpp" "VParticles/*.cu")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/ParticleSystem.cpp")

add_executable(VParticles ${SOURCES})

if(MSVC)
    target_compile_options(VParticles PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

# Link correct lib dirs and names per configuration
target_link_directories(VParticles PRIVATE
    $<$<CONFIG:Debug>:${VCPKG_ROOT}/debug/lib>
    $<$<NOT:$<CONFIG:Debug>>:${VCPKG_ROOT}/lib>
    D:/CUDA/lib/x64
)

target_link_libraries(VParticles PRIVATE
    # SFML
    $<$<CONFIG:Debug>:sfml-graphics-d>
    $<$<NOT:$<CONFIG:Debug>>:sfml-graphics>
    $<$<CONFIG:Debug>:sfml-window-d>
    $<$<NOT:$<CONFIG:Debug>>:sfml-window>
    $<$<CONFIG:Debug>:sfml-system-d>
    $<$<NOT:$<CONFIG:Debug>>:sfml-system>
    # ImGui-SFML / imgui
    ImGui-SFML
    $<$<CONFIG:Debug>:imguid>
    $<$<NOT:$<CONFIG:Debug>>:imgui>
    # GLEW (OpenGL extension loading)
    $<$<CONFIG:Debug>:glew32d>
    $<$<NOT:$<CONFIG:Debug>>:glew32>
    OpenGL32
    # CUDA
    cudadevrt
    cudart
    curand
    nvml
)
