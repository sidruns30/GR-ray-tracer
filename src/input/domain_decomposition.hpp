#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

struct IndexRange {
    std::size_t begin = 0;
    std::size_t end = 0;

    std::size_t size() const { return end - begin; }
};

inline IndexRange split_index_range(std::size_t extent, int partitions, int coordinate) {
    const std::size_t base = extent / static_cast<std::size_t>(partitions);
    const std::size_t remainder = extent % static_cast<std::size_t>(partitions);
    const std::size_t coord = static_cast<std::size_t>(coordinate);
    const std::size_t begin = coord * base + (coord < remainder ? coord : remainder);
    return {begin, begin + base + (coord < remainder ? 1u : 0u)};
}

struct DomainDecomposition {
    std::array<int, 3> partitions{1, 1, 1};
    std::array<int, 3> coordinates{0, 0, 0};
    std::array<IndexRange, 3> ranges{};
};

struct DomainDecompositionSpec {
    // Ordering is [r, theta, phi]. Phi-first automatic decomposition is the default.
    std::array<int, 3> partitions{1, 1, -1};

    static DomainDecompositionSpec parse(std::string value) {
        const auto open = value.find('[');
        const auto close = value.rfind(']');
        if (open != std::string::npos && close != std::string::npos && close > open) {
            value = value.substr(open + 1, close - open - 1);
        }

        DomainDecompositionSpec spec;
        std::size_t begin = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const std::size_t comma = value.find(',', begin);
            if (axis < 2 && comma == std::string::npos) {
                throw std::runtime_error("parallel.decomposition must contain [r, theta, phi]");
            }
            const std::string token = value.substr(begin, comma - begin);
            try {
                spec.partitions[axis] = std::stoi(token);
            } catch (...) {
                throw std::runtime_error("parallel.decomposition contains a non-integer value");
            }
            begin = comma == std::string::npos ? value.size() : comma + 1;
        }
        if (value.find(',', begin) != std::string::npos || value.find_first_not_of(" \t", begin) != std::string::npos) {
            throw std::runtime_error("parallel.decomposition must contain exactly three integers");
        }
        return spec;
    }

    std::array<int, 3> resolve_partitions(int mpi_size) const {
        std::array<int, 3> resolved = partitions;
        int inferred_axis = -1;
        int specified_product = 1;
        for (int axis = 0; axis < 3; ++axis) {
            if (resolved[axis] == -1) {
                if (inferred_axis >= 0) {
                    throw std::runtime_error("parallel.decomposition may contain only one -1");
                }
                inferred_axis = axis;
            } else if (resolved[axis] <= 0) {
                throw std::runtime_error("parallel.decomposition entries must be positive or -1");
            } else {
                specified_product *= resolved[axis];
            }
        }

        if (inferred_axis >= 0) {
            if (mpi_size % specified_product != 0) {
                throw std::runtime_error("MPI task count is not divisible by the specified decomposition factors");
            }
            resolved[inferred_axis] = mpi_size / specified_product;
        } else if (specified_product != mpi_size) {
            throw std::runtime_error("parallel.decomposition product must equal the MPI task count");
        }
        return resolved;
    }

    DomainDecomposition resolve(int mpi_size, int mpi_rank,
                                const std::array<std::size_t, 3>& global_extents) const {
        DomainDecomposition result;
        result.partitions = resolve_partitions(mpi_size);
        for (int axis = 0; axis < 3; ++axis) {
            if (static_cast<std::size_t>(result.partitions[axis]) > global_extents[axis]) {
                throw std::runtime_error("domain decomposition has more partitions than grid cells on an axis");
            }
        }

        // Phi varies fastest, followed by theta and radius.
        int remaining = mpi_rank;
        result.coordinates[2] = remaining % result.partitions[2];
        remaining /= result.partitions[2];
        result.coordinates[1] = remaining % result.partitions[1];
        remaining /= result.partitions[1];
        result.coordinates[0] = remaining;

        for (int axis = 0; axis < 3; ++axis) {
            result.ranges[axis] = split_index_range(
                global_extents[axis], result.partitions[axis], result.coordinates[axis]);
        }
        return result;
    }
};
