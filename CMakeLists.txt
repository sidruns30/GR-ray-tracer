cmake_minimum_required(VERSION 3.16)

project(GR-ray-tracer
    VERSION 0.1.0
    DESCRIPTION "GR ray tracer"
    LANGUAGES CXX)

# Defaults
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Helpful flags for editor integrations
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_TESTING "Enable building tests" ON)
option(ENABLE_WARNINGS "Enable extra compiler warnings" ON)

if(ENABLE_WARNINGS)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-Wall HAS_WALL)
    if(HAS_WALL)
        add_compile_options(-Wall -Wextra -Wpedantic)
    endif()
endif()

# Add subdirectory containing the project's CMakeLists.txt
add_subdirectory(src)

# Testing and packaging helpers
include(CTest)
include(CPack)