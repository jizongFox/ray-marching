cmake_minimum_required(VERSION 3.18)
project(ray-marching LANGUAGES CXX CUDA)

set(Torch_DIR /opt/miniconda3/envs/nerfstudio/lib/python3.10/site-packages/torch/share/cmake/Torch)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3 -std=c++14 ")

set(HEADERS include)

include_directories(${HEADERS})
include_directories(/opt/miniconda3/envs/nerfstudio/include/python3.10)
include_directories(/opt/miniconda3/envs/nerfstudio/lib/python3.10/site-packages/torch/include)
include_directories(/opt/miniconda3/envs/nerfstudio/lib/python3.10/site-packages/torch/include/torch/csrc/api/include)
include_directories(/usr/local/cuda/include)

set(PYTHON_EXECUTABLE /opt/miniconda3/envs/nerfstudio/bin/python)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif ()

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)


#find_package(Torch REQUIRED PATHS ${Torch_DIR})
#find_package(Torch REQUIRED)

#include(FetchContent)
#FetchContent_Declare(LoguruGitRepo
#        GIT_REPOSITORY "https://github.com/emilk/loguru" # can be a filesystem path
#        GIT_TAG        "master"
#        )
## set any loguru compile-time flags before calling MakeAvailable()
#set(LOGURU_WITH_STREAMS TRUE)
#FetchContent_MakeAvailable(LoguruGitRepo)


add_library(raymarchinglib STATIC ${HEADERS} src/raymarching.cu src/bindings.cpp)

target_link_libraries(raymarchinglib PUBLIC Python::Python "${TORCH_LIBRARIES}")

find_library(RAYMARCHING raymarchinglib cmake-build-debug)
add_executable(test test.cpp)

target_link_libraries(test PUBLIC Python::Python "${TORCH_LIBRARIES}" raymarchinglib)
