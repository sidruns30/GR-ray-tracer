find_package(MPI REQUIRED COMPONENTS CXX)

if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix kokkos
            OUTPUT_VARIABLE KOKKOS_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(KOKKOS_PREFIX)
            list(APPEND CMAKE_PREFIX_PATH "${KOKKOS_PREFIX}")
        endif()
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(LIBOMP_PREFIX)
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "" FORCE)
        endif()
    endif()
endif()

find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED CONFIG)

add_library(grraytracer_core STATIC utils.cpp)
target_include_directories(grraytracer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(grraytracer_core PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX Kokkos::kokkos)

add_executable(gr-ray-trace main.cpp)
target_link_libraries(gr-ray-trace PRIVATE grraytracer_core)

if(BUILD_TESTING)
    add_executable(test_npy_loader tests/test_npy_loader.cpp)
    target_link_libraries(test_npy_loader PRIVATE grraytracer_core)
    add_test(NAME npy_loader COMMAND test_npy_loader)

    add_executable(test_geodesic_observables tests/test_geodesic_observables.cpp)
    target_link_libraries(test_geodesic_observables PRIVATE grraytracer_core)
    add_test(NAME geodesic_observables COMMAND test_geodesic_observables)

    add_executable(test_rk_integrators tests/test_rk_integrators.cpp)
    target_link_libraries(test_rk_integrators PRIVATE grraytracer_core)
    add_test(NAME rk_integrators COMMAND test_rk_integrators)
endif()
