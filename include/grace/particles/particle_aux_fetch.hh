/**
 * @file particle_aux_fetch.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Per-substep aux-fetch protocol via Cabana::Distributor.
 *
 * Mechanism: forward Distributor ships request tuples (orig_rank, orig_idx,
 * owner_local_quad, position) to fluid owners; owners run trilinear interp on
 * the requested cell-centered fields; reverse Distributor ships responses
 * back. See doc/design/particles.md.
 *
 * This header exposes the protocol as a templated function parametrized by
 * the number of fetched fields. The v1 tracer-fetch field count is fixed and
 * declared in particle_storage.hh (sample block of tracer_member_types); for
 * MC/PIC species, instantiate with a different N_FIELDS.
 */
#ifndef GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH
#define GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_CABANA

#include <grace/data_structures/memory_defaults.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstdint>
#include <mpi.h>

namespace grace {
namespace particles {

/// Source array for a fetched scalar cell-centered field.
enum class field_source : uint8_t {
    STATE = 0,
    AUX   = 1,
};

struct fetch_field_spec_t {
    field_source source;
    int          var_idx;
};

/// Fetch N_FIELDS scalar cell-centered fields at each particle's position via
/// trilinear interpolation on the owning fluid rank.
///
/// All input/output views must be allocated on grace::default_space. The
/// caller is responsible for setting `owner_rank` and `owner_local_quad` to
/// values consistent with the current fluid topology shadow (see
/// particle_owner_search.hh). Particles with `owner_rank < 0` are skipped
/// silently — `out` rows are left untouched (caller should pre-zero if
/// desired).
///
/// `fields[].source` selects the source array on the owner side (STATE or AUX);
/// `fields[].var_idx` is the variable index within that array.
///
/// Output `out` has shape (n_particles, N_FIELDS); row `i` receives the
/// trilinearly-interpolated values for particle `i` in the same order as
/// `fields`.
template <int N_FIELDS>
void fetch_at_positions(
    MPI_Comm comm,
    std::size_t n_particles,
    const std::array<fetch_field_spec_t, N_FIELDS>& fields,
    Kokkos::View<int*,         grace::default_space> owner_rank,
    Kokkos::View<int*,         grace::default_space> owner_local_quad,
    Kokkos::View<double*[3],   grace::default_space> positions,
    Kokkos::View<double**,     grace::default_space> out);

} // namespace particles
} // namespace grace

#include "particle_aux_fetch.tpp"

#endif // GRACE_ENABLE_CABANA

#endif // GRACE_PARTICLES_PARTICLE_AUX_FETCH_HH
