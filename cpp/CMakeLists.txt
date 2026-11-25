cmake_minimum_required(VERSION 3.20)
project(voxt_polish_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LLAMA_PREFIX "/opt/homebrew/opt/llama.cpp" CACHE PATH "Path to llama.cpp install prefix")

find_path(LLAMA_INCLUDE_DIR llama.h PATHS "${LLAMA_PREFIX}/include" REQUIRED)
find_library(LLAMA_LIB llama PATHS "${LLAMA_PREFIX}/lib" REQUIRED)
find_library(GGML_LIB ggml PATHS "${LLAMA_PREFIX}/lib" REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_PREFIX}/lib" REQUIRED)

add_executable(voxt_polish_cpp
    polish_engine.cpp
    main.cpp
)

target_include_directories(voxt_polish_cpp PRIVATE "${LLAMA_INCLUDE_DIR}")
target_link_libraries(voxt_polish_cpp PRIVATE "${LLAMA_LIB}" "${GGML_LIB}" "${GGML_BASE_LIB}")
target_compile_options(voxt_polish_cpp PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)

set_target_properties(voxt_polish_cpp PROPERTIES
    BUILD_RPATH "${LLAMA_PREFIX}/lib"
    INSTALL_RPATH "${LLAMA_PREFIX}/lib"
)
