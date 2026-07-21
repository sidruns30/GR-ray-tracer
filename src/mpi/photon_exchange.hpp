// Batched photon ownership transfer between MPI domains.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "../input/load_python_arrays.hpp"
#include "../metrics/kerr_schild_core.hpp"

struct PhotonDomainBounds {
    std::array<real, 3> minimum{};
    std::array<real, 3> maximum{};
};

inline MPI_Datatype create_photon_record_type();

// Every rank keeps the small rank-bound table. Classification is therefore
// local, and one collective sends one contiguous bundle to each destination.
class PhotonDomainMap {
public:
    PhotonDomainMap(const NumpyFieldViews& fields, int mpi_size) {
        domains_.resize(static_cast<std::size_t>(mpi_size));
        std::array<real, 6> local{
            fields.coordinate_min[0], fields.coordinate_min[1], fields.coordinate_min[2],
            fields.coordinate_max[0], fields.coordinate_max[1], fields.coordinate_max[2]};
        std::vector<real> gathered(static_cast<std::size_t>(mpi_size) * local.size());
        MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                      gathered.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        for (int rank = 0; rank < mpi_size; ++rank) {
            const std::size_t offset = static_cast<std::size_t>(rank) * local.size();
            domains_[rank].minimum = {
                gathered[offset], gathered[offset + 1], gathered[offset + 2]};
            domains_[rank].maximum = {
                gathered[offset + 3], gathered[offset + 4], gathered[offset + 5]};
        }
        device_bounds_ = Kokkos::View<real**>("photon_domain_bounds", mpi_size, 6);
        auto host_bounds = Kokkos::create_mirror_view(device_bounds_);
        for (int rank = 0; rank < mpi_size; ++rank) {
            for (int axis = 0; axis < 3; ++axis) {
                host_bounds(rank, axis) = domains_[rank].minimum[axis];
                host_bounds(rank, axis + 3) = domains_[rank].maximum[axis];
            }
        }
        Kokkos::deep_copy(device_bounds_, host_bounds);
        record_type_ = create_photon_record_type();
    }

    PhotonDomainMap(const PhotonDomainMap&) = delete;
    PhotonDomainMap& operator=(const PhotonDomainMap&) = delete;

    ~PhotonDomainMap() {
        if (record_type_ != MPI_DATATYPE_NULL) MPI_Type_free(&record_type_);
    }

    const Kokkos::View<real**>& device_bounds() const { return device_bounds_; }
    MPI_Datatype record_type() const { return record_type_; }

    int owner(real x, real y, real z, real spin, int current_rank) const {
        const real radius = kerr_schild::compute_r(x, y, z, spin);
        if (!(radius > 0.0) || !Kokkos::isfinite(radius)) return current_rank;
        const real polar = Kokkos::acos(Kokkos::fmax(
            real(-1.0), Kokkos::fmin(real(1.0), z / radius)));
        real azimuth = Kokkos::atan2(radius * y - spin * x, radius * x + spin * y);
        if (azimuth < 0.0) azimuth += 2.0 * PI;

        int best_rank = current_rank;
        real best_score = score(domains_[current_rank], radius, polar, azimuth);
        for (std::size_t rank = 0; rank < domains_.size(); ++rank) {
            const real candidate = score(domains_[rank], radius, polar, azimuth);
            if (candidate < best_score) {
                best_score = candidate;
                best_rank = static_cast<int>(rank);
            }
        }
        return best_rank;
    }

private:
    static real periodic_distance(real lhs, real rhs) {
        real distance = Kokkos::fmod(Kokkos::abs(lhs - rhs), 2.0 * PI);
        return Kokkos::fmin(distance, 2.0 * PI - distance);
    }

    static real linear_distance(real value, real minimum, real maximum) {
        if (value < minimum) return minimum - value;
        if (value > maximum) return value - maximum;
        return 0.0;
    }

    static real angular_distance(real value, real minimum, real maximum) {
        if (value >= minimum && value <= maximum) return 0.0;
        return Kokkos::fmin(
            periodic_distance(value, minimum),
            periodic_distance(value, maximum));
    }

    static real score(const PhotonDomainBounds& bounds,
                      real radius, real polar, real azimuth) {
        const real radial_scale = Kokkos::fmax(
            real(1.0e-12), bounds.maximum[0] - bounds.minimum[0]);
        const real polar_scale = Kokkos::fmax(
            real(1.0e-12), bounds.maximum[1] - bounds.minimum[1]);
        const real azimuth_scale = Kokkos::fmax(
            real(1.0e-12), bounds.maximum[2] - bounds.minimum[2]);
        const real dr = linear_distance(radius, bounds.minimum[0], bounds.maximum[0]) / radial_scale;
        const real dt = linear_distance(polar, bounds.minimum[1], bounds.maximum[1]) / polar_scale;
        const real dp = angular_distance(azimuth, bounds.minimum[2], bounds.maximum[2]) / azimuth_scale;
        return dr * dr + dt * dt + dp * dp;
    }

    std::vector<PhotonDomainBounds> domains_;
    Kokkos::View<real**> device_bounds_;
    MPI_Datatype record_type_ = MPI_DATATYPE_NULL;
};

KOKKOS_INLINE_FUNCTION
real device_periodic_distance(real lhs, real rhs) {
    real distance = Kokkos::fmod(Kokkos::abs(lhs - rhs), 2.0 * PI);
    return Kokkos::fmin(distance, 2.0 * PI - distance);
}

KOKKOS_INLINE_FUNCTION
real device_interval_distance(real value, real minimum, real maximum) {
    if (value < minimum) return minimum - value;
    if (value > maximum) return value - maximum;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
int device_photon_owner(const Kokkos::View<real**>& bounds, int mpi_size,
                        real x, real y, real z, real spin, int current_rank) {
    const real radius = kerr_schild::compute_r(x, y, z, spin);
    if (!(radius > 0.0) || !Kokkos::isfinite(radius)) return current_rank;
    const real polar = Kokkos::acos(Kokkos::fmax(
        real(-1.0), Kokkos::fmin(real(1.0), z / radius)));
    real azimuth = Kokkos::atan2(radius * y - spin * x, radius * x + spin * y);
    if (azimuth < 0.0) azimuth += 2.0 * PI;

    int best_rank = current_rank;
    real best_score = 1.0e300;
    for (int rank = 0; rank < mpi_size; ++rank) {
        const real radial_scale = Kokkos::fmax(real(1.0e-12), bounds(rank, 3) - bounds(rank, 0));
        const real polar_scale = Kokkos::fmax(real(1.0e-12), bounds(rank, 4) - bounds(rank, 1));
        const real azimuth_scale = Kokkos::fmax(real(1.0e-12), bounds(rank, 5) - bounds(rank, 2));
        const real dr = device_interval_distance(radius, bounds(rank, 0), bounds(rank, 3)) / radial_scale;
        const real dt = device_interval_distance(polar, bounds(rank, 1), bounds(rank, 4)) / polar_scale;
        real dp = 0.0;
        if (azimuth < bounds(rank, 2) || azimuth > bounds(rank, 5)) {
            dp = Kokkos::fmin(
                device_periodic_distance(azimuth, bounds(rank, 2)),
                device_periodic_distance(azimuth, bounds(rank, 5))) / azimuth_scale;
        }
        const real candidate = dr * dr + dt * dt + dp * dp;
        if (candidate < best_score || (candidate == best_score && rank == current_rank)) {
            best_score = candidate;
            best_rank = rank;
        }
    }
    return best_rank;
}

// Wire record is intentionally a simple, fixed-size value type. Keeping every
// photon field together makes the destination bundles contiguous and avoids a
// separate MPI collective for each structure-of-arrays field.
struct PhotonWireRecord {
    std::uint64_t id;
    real frequency;
    real emission_frame_energy;
    real x0, x1, x2, x3;
    real k0, k1, k2, k3;
    real I, Q, U, V;
    real dlambda;
    real theta_disp, phi_disp;
    std::uint8_t terminate;
    std::uint8_t phase;
};

inline MPI_Datatype create_photon_record_type() {
    MPI_Datatype type;
    MPI_Type_contiguous(static_cast<int>(sizeof(PhotonWireRecord)), MPI_BYTE, &type);
    MPI_Type_commit(&type);
    return type;
}

KOKKOS_INLINE_FUNCTION
PhotonWireRecord pack_photon_record(const Photons& photons, int i) {
    return {
        photons.id(i), photons.frequency(i), photons.emission_frame_energy(i),
        photons.x0(i), photons.x1(i), photons.x2(i), photons.x3(i),
        photons.k0(i), photons.k1(i), photons.k2(i), photons.k3(i),
        photons.I(i), photons.Q(i), photons.U(i), photons.V(i),
        photons.dlambda(i), photons.theta_disp(i), photons.phi_disp(i),
        static_cast<std::uint8_t>(photons.terminate(i) ? 1 : 0), photons.phase(i)};
}

KOKKOS_INLINE_FUNCTION
void unpack_photon_record(const Photons& photons, int i, const PhotonWireRecord& record) {
    photons.id(i) = record.id;
    photons.frequency(i) = record.frequency;
    photons.emission_frame_energy(i) = record.emission_frame_energy;
    photons.x0(i) = record.x0; photons.x1(i) = record.x1;
    photons.x2(i) = record.x2; photons.x3(i) = record.x3;
    photons.k0(i) = record.k0; photons.k1(i) = record.k1;
    photons.k2(i) = record.k2; photons.k3(i) = record.k3;
    photons.I(i) = record.I; photons.Q(i) = record.Q;
    photons.U(i) = record.U; photons.V(i) = record.V;
    photons.dlambda(i) = record.dlambda;
    photons.theta_disp(i) = record.theta_disp;
    photons.phi_disp(i) = record.phi_disp;
    photons.terminate(i) = record.terminate != 0;
    photons.phase(i) = record.phase;
}

inline std::vector<PhotonWireRecord> photons_to_host_records(const Photons& photons) {
    Kokkos::View<PhotonWireRecord*> packed("packed_photon_exchange", photons.id.extent(0));
    Kokkos::parallel_for(
        "PackPhotonExchange",
        Kokkos::RangePolicy<>(0, photons.id.extent(0)),
        KOKKOS_LAMBDA(const int i) {
            packed(i) = pack_photon_record(photons, i);
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, packed);
    std::vector<PhotonWireRecord> records(host.extent(0));
    for (std::size_t i = 0; i < records.size(); ++i) records[i] = host(i);
    return records;
}

inline Photons host_records_to_photons(const std::vector<PhotonWireRecord>& records) {
    if (records.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Photon exchange assigned more than INT_MAX packets to one MPI rank; use more ranks");
    }
    Photons result(static_cast<int>(records.size()), "exchanged_photons");
    Kokkos::View<PhotonWireRecord*, Kokkos::HostSpace> host("received_photons_host", records.size());
    for (std::size_t i = 0; i < records.size(); ++i) host(i) = records[i];
    Kokkos::View<PhotonWireRecord*> packed("received_photons_device", records.size());
    Kokkos::deep_copy(packed, host);
    Kokkos::parallel_for(
        "UnpackPhotonExchange",
        Kokkos::RangePolicy<>(0, records.size()),
        KOKKOS_LAMBDA(const int i) {
            unpack_photon_record(result, i, packed(i));
        });
    return result;
}

inline void exchange_photon_ownership(Photons& photons,
                                      const PhotonDomainMap& domains,
                                      int mpi_rank,
                                      int mpi_size,
                                      real spin) {
    const int local_count = static_cast<int>(photons.id.extent(0));
    Kokkos::View<int*> destinations("photon_destinations", local_count);
    Kokkos::View<int*> device_send_counts("photon_send_counts", mpi_size);
    Kokkos::deep_copy(device_send_counts, 0);
    const auto bounds = domains.device_bounds();
    Kokkos::parallel_for(
        "ClassifyPhotonDomains", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(const int i) {
            const int destination = photons.terminate(i)
                ? mpi_rank
                : device_photon_owner(
                    bounds, mpi_size, photons.x1(i), photons.x2(i), photons.x3(i),
                    spin, mpi_rank);
            destinations(i) = destination;
            if (destination != mpi_rank) {
                Kokkos::atomic_fetch_add(&device_send_counts(destination), 1);
            }
        });
    const auto host_send_counts = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, device_send_counts);
    std::vector<int> send_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_counts(static_cast<std::size_t>(mpi_size));
    std::vector<int> send_offsets(static_cast<std::size_t>(mpi_size));
    std::vector<int> receive_offsets(static_cast<std::size_t>(mpi_size));
    std::size_t send_total = 0;
    for (int rank = 0; rank < mpi_size; ++rank) {
        send_counts[rank] = host_send_counts(rank);
        send_offsets[rank] = static_cast<int>(send_total);
        send_total += static_cast<std::size_t>(send_counts[rank]);
    }
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 receive_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::size_t receive_total = 0;
    for (int rank = 0; rank < mpi_size; ++rank) {
        receive_offsets[rank] = static_cast<int>(receive_total);
        receive_total += static_cast<std::size_t>(receive_counts[rank]);
    }
    if (receive_total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Photon exchange assigned more than INT_MAX packets to one MPI rank; use more ranks");
    }

    std::uint64_t local_outgoing = static_cast<std::uint64_t>(send_total);
    std::uint64_t global_outgoing = 0;
    MPI_Allreduce(&local_outgoing, &global_outgoing, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (global_outgoing == 0) return;

    Kokkos::View<int*> device_cursors("photon_send_cursors", mpi_size);
    auto host_cursors = Kokkos::create_mirror_view(device_cursors);
    for (int rank = 0; rank < mpi_size; ++rank) host_cursors(rank) = send_offsets[rank];
    Kokkos::deep_copy(device_cursors, host_cursors);
    Kokkos::View<PhotonWireRecord*> device_send_buffer(
        "photon_send_buffer", send_total);
    Kokkos::parallel_for(
        "PackOutgoingPhotons", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(const int i) {
            const int destination = destinations(i);
            if (destination != mpi_rank) {
                const int offset = Kokkos::atomic_fetch_add(&device_cursors(destination), 1);
                device_send_buffer(offset) = pack_photon_record(photons, i);
            }
        });
    const auto host_send_buffer = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, device_send_buffer);
    Kokkos::View<PhotonWireRecord*, Kokkos::HostSpace> host_receive_buffer(
        "photon_receive_buffer_host", receive_total);

    const MPI_Datatype record_type = domains.record_type();
    MPI_Alltoallv(host_send_buffer.data(), send_counts.data(), send_offsets.data(), record_type,
                  host_receive_buffer.data(), receive_counts.data(), receive_offsets.data(), record_type,
                  MPI_COMM_WORLD);

    if (send_total == 0 && receive_total == 0) return;
    const std::size_t retained_count = static_cast<std::size_t>(local_count) - send_total;
    const std::size_t new_count = retained_count + receive_total;
    if (new_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Photon exchange assigned more than INT_MAX packets to one MPI rank; use more ranks");
    }
    Photons updated(static_cast<int>(new_count), "migrated_photons");
    Kokkos::parallel_scan(
        "CompactRetainedPhotons", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(const int i, int& retained_offset, const bool final) {
            if (destinations(i) == mpi_rank) {
                if (final) {
                    unpack_photon_record(updated, retained_offset, pack_photon_record(photons, i));
                }
                ++retained_offset;
            }
        });

    Kokkos::View<PhotonWireRecord*> device_receive_buffer(
        "photon_receive_buffer_device", receive_total);
    Kokkos::deep_copy(device_receive_buffer, host_receive_buffer);
    Kokkos::parallel_for(
        "UnpackIncomingPhotons", Kokkos::RangePolicy<>(0, receive_total),
        KOKKOS_LAMBDA(const int i) {
            unpack_photon_record(
                updated, static_cast<int>(retained_count) + i, device_receive_buffer(i));
        });
    Kokkos::fence();
    photons = std::move(updated);
}
