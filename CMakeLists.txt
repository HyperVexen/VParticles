cmake_minimum_required(VERSION 3.18)
project(VParticles LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ----------------------------------------------------------------------
# 1. Core Simulation Library (Static)
# ----------------------------------------------------------------------
add_library(VParticlesCore STATIC
    "VParticles/src/ParticleSystem.cu"
    "VParticles/src/interop/ExternalInterop.cu"
)

target_include_directories(VParticlesCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/include"
)

target_link_libraries(VParticlesCore PUBLIC
    CUDA::cudart
)

if(MSVC)
    target_compile_options(VParticlesCore PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

set_target_properties(VParticlesCore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ----------------------------------------------------------------------
# 2. Headless Benchmark Executable
# ----------------------------------------------------------------------
add_executable(VParticles
    "VParticles/src/VParticles.cpp"
)

target_link_libraries(VParticles PRIVATE
    VParticlesCore
    CUDA::cudart
)

if(MSVC)
    target_compile_options(VParticles PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

set_target_properties(VParticles PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ----------------------------------------------------------------------
# 3. Interactive OpenGL Viewer Executable
# ----------------------------------------------------------------------
add_executable(VParticlesViewer
    "VParticles/src/viewer/ViewerMain.cpp"
    "VParticles/src/viewer/CudaGLBridge.cu"
    "third_party/glad/src/glad.c"
)

target_include_directories(VParticlesViewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/src/viewer"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad/include"
)

target_link_libraries(VParticlesViewer PRIVATE
    VParticlesCore
    glfw
    OpenGL::GL
    CUDA::cudart
)

if(MSVC)
    target_compile_options(VParticlesViewer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

set_target_properties(VParticlesViewer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ----------------------------------------------------------------------
# 4. DirectX 12 Interop Test Executable (Windows)
# ----------------------------------------------------------------------
if(WIN32)
    add_executable(VParticlesD3D12InteropTest
        "VParticles/src/interop/D3D12InteropTest.cpp"
    )

    target_include_directories(VParticlesD3D12InteropTest PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/include"
    )

    target_link_libraries(VParticlesD3D12InteropTest PRIVATE
        VParticlesCore
        CUDA::cudart
        d3d12
        dxgi
    )

    if(MSVC)
        target_compile_options(VParticlesD3D12InteropTest PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        )
    endif()
endif()

