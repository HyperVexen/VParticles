cmake_minimum_required(VERSION 3.18)
project(VParticles LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE VPARTICLES_SOURCES CONFIGURE_DEPENDS
    "VParticles/src/*.cpp"
    "VParticles/src/*.cu"
)

add_executable(VParticles ${VPARTICLES_SOURCES})

target_include_directories(VParticles PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/VParticles/include"
)

target_link_libraries(VParticles PRIVATE
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
