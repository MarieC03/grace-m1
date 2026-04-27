/**
 * @file particle_storage.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief AoSoA layout and field-index conventions for GRACE particles.
 *
 * The storage type is a Cabana::AoSoA. Field indices are exposed as a strong
 * enum so that callers do not depend on the order of MemberTypes.
 *
 * Sort key: ascending (owner_rank, owner_local_quad, intra_quad_idx). Because
 * p4est SFC traversal is z-order, this is particle Morton order at the quad's
 * level — see doc/design/particles.md.
 */
#ifndef GRACE_PARTICLES_PARTICLE_STORAGE_HH
#define GRACE_PARTICLES_PARTICLE_STORAGE_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_CABANA

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <cstdint>

namespace grace {
namespace particles {

// Tracer AoSoA field indices. The advection block is what the per-substep
// fetch needs; the sample block is what the tracer actually records for
// scientific output. Both are populated in the same fetch round-trip — see
// doc/design/particles.md (sampling section).
enum class particle_field : int {
    // Identity / topology
    pos              = 0,  // double[3]   physical position
    id               = 1,  // uint64      immortal global id
    status           = 2,  // uint8       particle_status_flag_t
    owner_rank       = 3,  // int32       cached owning fluid rank
    owner_local_quad = 4,  // int32       cached local quad index on owner
    // Advection inputs (sampled at src_pos every substep)
    sample_alpha     = 5,  // double      lapse
    sample_beta      = 6,  // double[3]   shift
    sample_v         = 7,  // double[3]   3-velocity
    sample_W         = 8,  // double      Lorentz factor
    // Hydro samples (output)
    sample_rho       = 9,  // double      rest-mass density
    sample_temp      = 10, // double      temperature
    sample_ye        = 11, // double      electron fraction
    sample_entropy   = 12, // double      specific entropy
    sample_press     = 13, // double      pressure
    sample_eps       = 14, // double      specific internal energy
    // Magnetic field samples
    sample_B         = 15  // double[3]   3-magnetic-field
};

using tracer_member_types = Cabana::MemberTypes<
    double[3],   // pos
    uint64_t,    // id
    uint8_t,     // status
    int32_t,     // owner_rank
    int32_t,     // owner_local_quad
    double,      // sample_alpha
    double[3],   // sample_beta
    double[3],   // sample_v
    double,      // sample_W
    double,      // sample_rho
    double,      // sample_temp
    double,      // sample_ye
    double,      // sample_entropy
    double,      // sample_press
    double,      // sample_eps
    double[3]    // sample_B
>;

constexpr int particle_vector_length = 16;

using default_memory_space = Kokkos::DefaultExecutionSpace::memory_space;

template <class MemorySpace = default_memory_space>
using tracer_aosoa_t = Cabana::AoSoA<tracer_member_types, MemorySpace,
                                     particle_vector_length>;

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_CABANA

#endif // GRACE_PARTICLES_PARTICLE_STORAGE_HH
