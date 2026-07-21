// Load NumPy grid fields into rank-local execution-space Kokkos Views.
#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../utils.hpp"
#include "domain_decomposition.hpp"

struct NumpyFieldPaths {
    std::string r;
    std::string theta;
    std::string phi;
    std::string density;
    std::string temperature;
    std::string velocity;
    std::string magnetic;
};

inline NumpyFieldPaths discover_numpy_field_paths(const std::string& directory) {
    namespace fs = std::filesystem;
    const fs::path root(directory);
    if (!fs::exists(root)) {
        throw std::runtime_error("NumPy directory does not exist: " + root.string());
    }
    if (!fs::is_directory(root)) {
        throw std::runtime_error("NumPy input path is not a directory: " + root.string());
    }

    NumpyFieldPaths paths{
        (root / "r.npy").string(),
        (root / "theta.npy").string(),
        (root / "phi.npy").string(),
        (root / "rho.npy").string(),
        (root / "Tgas.npy").string(),
        (root / "vel.npy").string(),
        (root / "mag.npy").string()};
    const std::array<std::pair<const char*, const std::string*>, 7> required{{
        {"r.npy", &paths.r}, {"theta.npy", &paths.theta}, {"phi.npy", &paths.phi},
        {"rho.npy", &paths.density}, {"Tgas.npy", &paths.temperature},
        {"vel.npy", &paths.velocity}, {"mag.npy", &paths.magnetic}}};

    std::string missing;
    for (const auto& [name, path] : required) {
        if (!fs::is_regular_file(*path)) {
            if (!missing.empty()) missing += ", ";
            missing += name;
        }
    }
    if (!missing.empty()) {
        throw std::runtime_error(
            "NumPy directory '" + root.string() + "' is missing required file(s): " + missing);
    }
    return paths;
}

struct NumpyFieldViews {
    Kokkos::View<real*> r;
    Kokkos::View<real*> theta;
    Kokkos::View<real*> phi;
    Kokkos::View<real***> density;
    Kokkos::View<real***> temperature;
    Kokkos::View<real****> velocity;
    Kokkos::View<real****> magnetic;
    DomainDecomposition decomposition;
    std::array<std::size_t, 3> global_extents{0, 0, 0};
    // Logarithmic r spacing followed by linear theta and phi spacing.
    std::array<real, 3> coordinate_spacing{0.0, 0.0, 0.0};
};

inline std::vector<real> load_coordinate_values(const std::string& filename) {
    const cnpy::NpyArray array = cnpy::npy_load(filename);
    if (array.shape.size() != 1 || array.shape[0] < 2) {
        throw std::runtime_error("Coordinate array must be one-dimensional with at least two values: " + filename);
    }
    if (array.descr != "<f8" && array.descr != "=f8") {
        throw std::runtime_error("Grid arrays must use native/little-endian float64 values: " + filename);
    }
    const real* data = array.data<real>();
    return std::vector<real>(data, data + array.shape[0]);
}

inline void validate_coordinate_spacing(const std::vector<real>& values,
                                        const std::string& name,
                                        bool logarithmic) {
    const real spacing = logarithmic
        ? Kokkos::log(values[1] / values[0])
        : values[1] - values[0];
    for (std::size_t i = 1; i + 1 < values.size(); ++i) {
        const real current = logarithmic
            ? Kokkos::log(values[i + 1] / values[i])
            : values[i + 1] - values[i];
        if (Kokkos::abs(current - spacing) > 1.0e-2) {
            throw std::runtime_error(name + " coordinate array has non-uniform spacing");
        }
    }
}

inline Kokkos::View<real*> make_coordinate_view(const std::vector<real>& values,
                                                IndexRange range,
                                                const std::string& label) {
    Kokkos::View<real*> view(label, range.size());
    auto host = Kokkos::create_mirror_view(view);
    for (std::size_t local = 0; local < range.size(); ++local) {
        host(local) = values[range.begin + local];
    }
    Kokkos::deep_copy(view, host);
    return view;
}

inline void validate_field_shape(const std::vector<std::size_t>& shape,
                                 const std::array<std::size_t, 3>& expected,
                                 std::size_t dimensions,
                                 const std::string& filename) {
    if (shape.size() != dimensions ||
        shape[0] != expected[0] ||
        shape[1] != expected[1] ||
        shape[2] != expected[2]) {
        throw std::runtime_error("Grid field has incompatible dimensions: " + filename);
    }
}

inline cnpy::NpyHeader open_real_numpy_file(const std::string& filename,
                                             std::ifstream& input) {
    input.open(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open numpy file: " + filename);
    cnpy::NpyHeader header = cnpy::read_npy_header(input, filename);
    if (header.descr != "<f8" && header.descr != "=f8") {
        throw std::runtime_error("Grid arrays must use native/little-endian float64 values: " + filename);
    }
    return header;
}

inline void read_numpy_values(std::ifstream& input,
                              const cnpy::NpyHeader& header,
                              std::size_t value_offset,
                              real* destination,
                              std::size_t value_count,
                              const std::string& filename) {
    const auto byte_offset = header.data_offset +
        static_cast<std::streamoff>(value_offset * sizeof(real));
    input.seekg(byte_offset);
    input.read(reinterpret_cast<char*>(destination),
               static_cast<std::streamsize>(value_count * sizeof(real)));
    if (!input) throw std::runtime_error("Unexpected EOF while reading numpy data: " + filename);
}

inline Kokkos::View<real***> load_3d_numpy_array_slice(
    const std::string& filename,
    const std::array<std::size_t, 3>& global_extents,
    const std::array<IndexRange, 3>& ranges) {
    std::ifstream input;
    const cnpy::NpyHeader header = open_real_numpy_file(filename, input);
    validate_field_shape(header.shape, global_extents, 3, filename);

    Kokkos::View<real***> view(
        "loaded_3d_field", ranges[0].size(), ranges[1].size(), ranges[2].size());
    auto host = Kokkos::create_mirror_view(view);
    std::vector<real> row(ranges[2].size());
    for (std::size_t i = 0; i < ranges[0].size(); ++i) {
        const std::size_t gi = ranges[0].begin + i;
        for (std::size_t j = 0; j < ranges[1].size(); ++j) {
            const std::size_t gj = ranges[1].begin + j;
            const std::size_t row_offset =
                (gi * global_extents[1] + gj) * global_extents[2] + ranges[2].begin;
            read_numpy_values(input, header, row_offset, row.data(), row.size(), filename);
            for (std::size_t k = 0; k < row.size(); ++k) host(i, j, k) = row[k];
        }
    }
    Kokkos::deep_copy(view, host);
    return view;
}

inline Kokkos::View<real****> load_4d_numpy_array_slice(
    const std::string& filename,
    const std::array<std::size_t, 3>& global_extents,
    const std::array<IndexRange, 3>& ranges) {
    std::ifstream input;
    const cnpy::NpyHeader header = open_real_numpy_file(filename, input);
    validate_field_shape(header.shape, global_extents, 4, filename);
    const std::size_t components = header.shape[3];
    if (components != 4) {
        throw std::runtime_error("Four-vector field must have four components: " + filename);
    }

    Kokkos::View<real****> view(
        "loaded_4d_field", ranges[0].size(), ranges[1].size(), ranges[2].size(), components);
    auto host = Kokkos::create_mirror_view(view);
    std::vector<real> row(ranges[2].size() * components);
    for (std::size_t i = 0; i < ranges[0].size(); ++i) {
        const std::size_t gi = ranges[0].begin + i;
        for (std::size_t j = 0; j < ranges[1].size(); ++j) {
            const std::size_t gj = ranges[1].begin + j;
            const std::size_t row_offset =
                ((gi * global_extents[1] + gj) * global_extents[2] + ranges[2].begin) * components;
            read_numpy_values(input, header, row_offset, row.data(), row.size(), filename);
            for (std::size_t k = 0; k < ranges[2].size(); ++k) {
                for (std::size_t component = 0; component < components; ++component) {
                    host(i, j, k, component) = row[k * components + component];
                }
            }
        }
    }
    Kokkos::deep_copy(view, host);
    return view;
}

inline NumpyFieldViews load_numpy_field_bundle(
    const NumpyFieldPaths& paths,
    const DomainDecompositionSpec& decomposition_spec = DomainDecompositionSpec{},
    int mpi_rank = 0,
    int mpi_size = 1) {
    const std::vector<real> r_values = load_coordinate_values(paths.r);
    const std::vector<real> theta_values = load_coordinate_values(paths.theta);
    const std::vector<real> phi_values = load_coordinate_values(paths.phi);
    validate_coordinate_spacing(r_values, "r", true);
    validate_coordinate_spacing(theta_values, "theta", false);
    validate_coordinate_spacing(phi_values, "phi", false);

    NumpyFieldViews views;
    views.global_extents = {r_values.size(), theta_values.size(), phi_values.size()};
    views.coordinate_spacing = {
        Kokkos::log(r_values[1] / r_values[0]),
        theta_values[1] - theta_values[0],
        phi_values[1] - phi_values[0]};
    views.decomposition = decomposition_spec.resolve(mpi_size, mpi_rank, views.global_extents);
    views.r = make_coordinate_view(r_values, views.decomposition.ranges[0], "r");
    views.theta = make_coordinate_view(theta_values, views.decomposition.ranges[1], "theta");
    views.phi = make_coordinate_view(phi_values, views.decomposition.ranges[2], "phi");
    views.density = load_3d_numpy_array_slice(paths.density, views.global_extents, views.decomposition.ranges);
    views.temperature = load_3d_numpy_array_slice(paths.temperature, views.global_extents, views.decomposition.ranges);
    views.velocity = load_4d_numpy_array_slice(paths.velocity, views.global_extents, views.decomposition.ranges);
    views.magnetic = load_4d_numpy_array_slice(paths.magnetic, views.global_extents, views.decomposition.ranges);

    nr = views.r.extent(0);
    ntheta = views.theta.extent(0);
    nphi = views.phi.extent(0);
    r_min = r_values.front();
    r_max = r_values.back();
    theta_min = theta_values.front();
    theta_max = theta_values.back();
    phi_min = phi_values.front();
    phi_max = phi_values.back();
    dlog_r = views.coordinate_spacing[0];
    return views;
}
