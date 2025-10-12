cmake_minimum_required(VERSION 3.18)
project(mlmodel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CPU_SOURCES "src/cpu/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/gpu/*.cu")
file(GLOB MAIN_SOURCE "src/main.cpp")

add_executable(mlmodel 
    ${MAIN_SOURCE}
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(mlmodel PRIVATE
    CUDA::cudart
    CUDA::curand
)

target_compile_options(mlmodel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        /W4           
        /permissive-  
    >
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:
        /Zi          
        /Od           
        /RTC1        
    >
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:
        /O2           
        /Ob2         
        /DNDEBUG     
    >
)

target_compile_options(mlmodel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math              
        -Xcompiler=/W4               
        $<$<CONFIG:Debug>:-G -g>     g
        $<$<CONFIG:Release>:-O3>     
    >
)

set_target_properties(mlmodel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")