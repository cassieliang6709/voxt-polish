cmake_minimum_required(VERSION 3.20)
project(voxt_polish_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Two ways to get llama.cpp:
# - LLAMA_SOURCE_DIR (build-static.sh): build it in, static, Metal shaders
#   embedded: one self-contained binary that Utter.app ships.
# - otherwise the Homebrew install (brew install llama.cpp).
set(LLAMA_SOURCE_DIR "" CACHE PATH "llama.cpp source tree to build in statically")
set(LLAMA_PREFIX "/opt/homebrew/opt/llama.cpp" CACHE PATH "Path to llama.cpp install prefix")

add_executable(voxt_polish_cpp
    polish_engine.cpp
    main.cpp
)
target_compile_options(voxt_polish_cpp PRIVATE -O3 -Wall -Wextra -Wno-unused-parameter)

if(LLAMA_SOURCE_DIR)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LLAMA_SOURCE_DIR}" llama.cpp EXCLUDE_FROM_ALL)
    target_link_libraries(voxt_polish_cpp PRIVATE llama)
else()
    find_path(LLAMA_INCLUDE_DIR llama.h PATHS "${LLAMA_PREFIX}/include" REQUIRED)
    find_library(LLAMA_LIB llama PATHS "${LLAMA_PREFIX}/lib" REQUIRED)
    find_library(GGML_LIB ggml PATHS "${LLAMA_PREFIX}/lib" REQUIRED)
    find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_PREFIX}/lib" REQUIRED)
    target_include_directories(voxt_polish_cpp PRIVATE "${LLAMA_INCLUDE_DIR}")
    target_link_libraries(voxt_polish_cpp PRIVATE "${LLAMA_LIB}" "${GGML_LIB}" "${GGML_BASE_LIB}")
    set_target_properties(voxt_polish_cpp PROPERTIES
        BUILD_RPATH "${LLAMA_PREFIX}/lib"
        INSTALL_RPATH "${LLAMA_PREFIX}/lib"
    )
endif()
