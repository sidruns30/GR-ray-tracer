cmake_minimum_required(VERSION 3.16)
project(gr-ray-trace LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Kokkos REQUIRED)

# cnpy is typically found via find_package or as a subdirectory
# If cnpy is installed system-wide, you can use find_package
# Otherwise, you may need to add it as a subdirectory or specify include/lib paths
# For this example, we assume it's installed or available via pkg-config
find_library(CNPY_LIBRARY NAMES cnpy)
find_path(CNPY_INCLUDE_DIR NAMES cnpy.h)

# Create executable
add_executable(gr-ray-trace main.cpp)

# Link libraries
target_link_libraries(gr-ray-trace
    PRIVATE
        MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        Kokkos::kokkos
)

# If cnpy was found, link it
if(CNPY_LIBRARY AND CNPY_INCLUDE_DIR)
    target_include_directories(gr-ray-trace PRIVATE ${CNPY_INCLUDE_DIR})
    target_link_libraries(gr-ray-trace PRIVATE ${CNPY_LIBRARY})
    target_link_libraries(gr-ray-trace PRIVATE z)  # cnpy requires zlib
else()
    message(WARNING "cnpy library not found. Please install cnpy or specify CNPY_LIBRARY and CNPY_INCLUDE_DIR")
    # You can uncomment the following lines to specify custom paths
    # set(CNPY_INCLUDE_DIR "/path/to/cnpy/include")
    # set(CNPY_LIBRARY "/path/to/cnpy/lib/libcnpy.a")
    # target_include_directories(gr-ray-trace PRIVATE ${CNPY_INCLUDE_DIR})
    # target_link_libraries(gr-ray-trace PRIVATE ${CNPY_LIBRARY} z)
endif()

# Print build configuration
message(STATUS "=== Build Configuration ===")
message(STATUS "MPI Found: ${MPI_FOUND}")
message(STATUS "OpenMP Found: ${OpenMP_FOUND}")
message(STATUS "Kokkos Found: ${Kokkos_FOUND}")
message(STATUS "cnpy Library: ${CNPY_LIBRARY}")
message(STATUS "cnpy Include: ${CNPY_INCLUDE_DIR}")
