/**
 * @file particle_aux_fetch.tpp
 * @brief Templated implementation of fetch_at_positions().
 *
 * Included from particle_aux_fetch.hh — do not include directly.
 */
#ifndef GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP
#define GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP

#ifdef GRACE_ENABLE_CABANA

#include <grace/particles/particle_owner_search.hh>
#include <grace/data_structures/variables.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/grace_utils.hh>

#include <Cabana_Core.hpp>

namespace grace {
namespace particles {

namespace detail {

// AoSoA payload types. Vector length kept small; will be tuned later.
constexpr int fetch_vector_length = 16;

// Request: orig_rank, orig_idx, owner_local_quad, position[3].
using request_member_types = Cabana::MemberTypes<
    int,         // orig_rank
    int,         // orig_idx
    int,         // owner_local_quad
    double[3]    // position
>;

template <int N_FIELDS>
using response_member_types = Cabana::MemberTypes<
    int,                // orig_rank
    int,                // orig_idx
    double[N_FIELDS]    // interpolated values
>;

// Pack-into-device-view helper for the per-quad geometry table.
struct quad_geom_dev_t {
    Kokkos::View<double*[6], grace::default_space> bbox;   // (q, {xlo,ylo,zlo,xhi,yhi,zhi})
    Kokkos::View<double*,    grace::default_space> dx;     // dx_cell per quad
};

inline quad_geom_dev_t copy_geom_to_device()
{
    auto& sh = fluid_topology_shadow_t::get();
    const auto& geom = sh.local_geometry();
    int n_local = static_cast<int>(geom.size());

    quad_geom_dev_t out;
    out.bbox = Kokkos::View<double*[6], grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_bbox"), n_local);
    out.dx = Kokkos::View<double*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_dx"), n_local);

    auto h_bbox = Kokkos::create_mirror_view(out.bbox);
    auto h_dx   = Kokkos::create_mirror_view(out.dx);
    for (int q = 0; q < n_local; ++q) {
        h_bbox(q, 0) = geom[q].bbox.xlo;
        h_bbox(q, 1) = geom[q].bbox.ylo;
        h_bbox(q, 2) = geom[q].bbox.zlo;
        h_bbox(q, 3) = geom[q].bbox.xhi;
        h_bbox(q, 4) = geom[q].bbox.yhi;
        h_bbox(q, 5) = geom[q].bbox.zhi;
        h_dx(q)      = geom[q].dx_cell;
    }
    Kokkos::deep_copy(out.bbox, h_bbox);
    Kokkos::deep_copy(out.dx,   h_dx);
    return out;
}

template <int N_FIELDS>
struct field_specs_dev_t {
    Kokkos::View<int*, grace::default_space> source; // STATE=0, AUX=1
    Kokkos::View<int*, grace::default_space> idx;
};

template <int N_FIELDS>
inline field_specs_dev_t<N_FIELDS> copy_field_specs_to_device(
    const std::array<fetch_field_spec_t, N_FIELDS>& fields)
{
    field_specs_dev_t<N_FIELDS> out;
    out.source = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_field_source"),
        N_FIELDS);
    out.idx = Kokkos::View<int*, grace::default_space>(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_field_idx"),
        N_FIELDS);
    auto h_source = Kokkos::create_mirror_view(out.source);
    auto h_idx    = Kokkos::create_mirror_view(out.idx);
    for (int f = 0; f < N_FIELDS; ++f) {
        h_source(f) = static_cast<int>(fields[f].source);
        h_idx(f)    = fields[f].var_idx;
    }
    Kokkos::deep_copy(out.source, h_source);
    Kokkos::deep_copy(out.idx,    h_idx);
    return out;
}

} // namespace detail

template <int N_FIELDS>
void fetch_at_positions(
    MPI_Comm comm,
    std::size_t n_particles,
    const std::array<fetch_field_spec_t, N_FIELDS>& fields,
    Kokkos::View<int*,         grace::default_space> owner_rank,
    Kokkos::View<int*,         grace::default_space> owner_local_quad,
    Kokkos::View<double*[3],   grace::default_space> positions,
    Kokkos::View<double**,     grace::default_space> out)
{
    using exec_space = typename grace::default_space::execution_space;
    using request_t  = detail::request_member_types;
    using response_t = detail::response_member_types<N_FIELDS>;
    constexpr int vlen = detail::fetch_vector_length;

    int my_rank = -1;
    MPI_Comm_rank(comm, &my_rank);

    // -------------------------------------------------------------------
    // 1. Build forward request AoSoA. Particles with owner_rank<0 are routed
    //    to my_rank with quad=-1 so the owner-side kernel can no-op them
    //    cheaply rather than special-casing the export plan.
    // -------------------------------------------------------------------
    Cabana::AoSoA<request_t, grace::default_space, vlen>
        req("particle_fetch_req", n_particles);
    auto req_orig_rank = Cabana::slice<0>(req);
    auto req_orig_idx  = Cabana::slice<1>(req);
    auto req_quad      = Cabana::slice<2>(req);
    auto req_pos       = Cabana::slice<3>(req);

    Kokkos::View<int*, grace::default_space> export_ranks(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_export"),
        n_particles);

    Kokkos::parallel_for("particle_fetch_pack_request",
        Kokkos::RangePolicy<exec_space>(0, n_particles),
        KOKKOS_LAMBDA(const int i) {
            int dst = owner_rank(i);
            if (dst < 0) {
                req_orig_rank(i) = my_rank;
                req_orig_idx(i)  = i;
                req_quad(i)      = -1;
                req_pos(i, 0)    = 0.0;
                req_pos(i, 1)    = 0.0;
                req_pos(i, 2)    = 0.0;
                export_ranks(i)  = my_rank;
            } else {
                req_orig_rank(i) = my_rank;
                req_orig_idx(i)  = i;
                req_quad(i)      = owner_local_quad(i);
                req_pos(i, 0)    = positions(i, 0);
                req_pos(i, 1)    = positions(i, 1);
                req_pos(i, 2)    = positions(i, 2);
                export_ranks(i)  = dst;
            }
        });
    Kokkos::fence();

    // -------------------------------------------------------------------
    // 2. Forward Distributor: ship requests to fluid owners.
    // -------------------------------------------------------------------
    Cabana::Distributor<grace::default_space> fwd(comm, export_ranks);
    Cabana::AoSoA<request_t, grace::default_space, vlen>
        req_recv("particle_fetch_req_recv", fwd.totalNumImport());
    Cabana::migrate(fwd, req, req_recv);

    // -------------------------------------------------------------------
    // 3. Owner-side fulfillment: trilinear interp per requested field.
    // -------------------------------------------------------------------
    auto rcv_orig_rank = Cabana::slice<0>(req_recv);
    auto rcv_orig_idx  = Cabana::slice<1>(req_recv);
    auto rcv_quad      = Cabana::slice<2>(req_recv);
    auto rcv_pos       = Cabana::slice<3>(req_recv);

    const std::size_t n_recv = req_recv.size();
    Cabana::AoSoA<response_t, grace::default_space, vlen>
        resp("particle_fetch_resp", n_recv);
    auto resp_orig_rank = Cabana::slice<0>(resp);
    auto resp_orig_idx  = Cabana::slice<1>(resp);
    auto resp_values    = Cabana::slice<2>(resp);

    auto& vars = grace::variables::variable_list::get();
    auto state_view = vars.getstate();
    auto aux_view   = vars.getaux();
    auto extents = grace::amr::get_quadrant_extents();
    const int nx_  = static_cast<int>(std::get<0>(extents));
    const int ngz_ = grace::amr::get_n_ghosts();

    auto geom_dev   = detail::copy_geom_to_device();
    auto specs_dev  = detail::copy_field_specs_to_device<N_FIELDS>(fields);

    auto bbox_v   = geom_dev.bbox;
    auto dx_v     = geom_dev.dx;
    auto src_v    = specs_dev.source;
    auto idx_v    = specs_dev.idx;

    Kokkos::parallel_for("particle_fetch_fulfill",
        Kokkos::RangePolicy<exec_space>(0, n_recv),
        KOKKOS_LAMBDA(const int r) {
            resp_orig_rank(r) = rcv_orig_rank(r);
            resp_orig_idx(r)  = rcv_orig_idx(r);

            const int q = rcv_quad(r);
            if (q < 0) {
                for (int f = 0; f < N_FIELDS; ++f) resp_values(r, f) = 0.0;
                return;
            }

            const double x = rcv_pos(r, 0);
            const double y = rcv_pos(r, 1);
            const double z = rcv_pos(r, 2);
            const double xlo = bbox_v(q, 0);
            const double ylo = bbox_v(q, 1);
            const double zlo = bbox_v(q, 2);
            const double inv_dx = 1.0 / dx_v(q);

            const double cx = (x - xlo) * inv_dx - 0.5;
            const double cy = (y - ylo) * inv_dx - 0.5;
            const double cz = (z - zlo) * inv_dx - 0.5;
            const int bi = static_cast<int>(Kokkos::floor(cx));
            const int bj = static_cast<int>(Kokkos::floor(cy));
            const int bk = static_cast<int>(Kokkos::floor(cz));
            const double fx = cx - static_cast<double>(bi);
            const double fy = cy - static_cast<double>(bj);
            const double fz = cz - static_cast<double>(bk);

            // Interior coords [0, nx) → view coords [ngz, ngz+nx).
            const int vi = bi + ngz_;
            const int vj = bj + ngz_;
            const int vk = bk + ngz_;

            const double w000 = (1.0-fx)*(1.0-fy)*(1.0-fz);
            const double w100 =      fx *(1.0-fy)*(1.0-fz);
            const double w010 = (1.0-fx)*     fy *(1.0-fz);
            const double w110 =      fx *     fy *(1.0-fz);
            const double w001 = (1.0-fx)*(1.0-fy)*     fz ;
            const double w101 =      fx *(1.0-fy)*     fz ;
            const double w011 = (1.0-fx)*     fy *     fz ;
            const double w111 =      fx *     fy *     fz ;

            for (int f = 0; f < N_FIELDS; ++f) {
                const int src = src_v(f);
                const int vidx = idx_v(f);
                double v;
                if (src == 0) {
                    v =   w000 * state_view(vi  , vj  , vk  , vidx, q)
                        + w100 * state_view(vi+1, vj  , vk  , vidx, q)
                        + w010 * state_view(vi  , vj+1, vk  , vidx, q)
                        + w110 * state_view(vi+1, vj+1, vk  , vidx, q)
                        + w001 * state_view(vi  , vj  , vk+1, vidx, q)
                        + w101 * state_view(vi+1, vj  , vk+1, vidx, q)
                        + w011 * state_view(vi  , vj+1, vk+1, vidx, q)
                        + w111 * state_view(vi+1, vj+1, vk+1, vidx, q);
                } else {
                    v =   w000 * aux_view(vi  , vj  , vk  , vidx, q)
                        + w100 * aux_view(vi+1, vj  , vk  , vidx, q)
                        + w010 * aux_view(vi  , vj+1, vk  , vidx, q)
                        + w110 * aux_view(vi+1, vj+1, vk  , vidx, q)
                        + w001 * aux_view(vi  , vj  , vk+1, vidx, q)
                        + w101 * aux_view(vi+1, vj  , vk+1, vidx, q)
                        + w011 * aux_view(vi  , vj+1, vk+1, vidx, q)
                        + w111 * aux_view(vi+1, vj+1, vk+1, vidx, q);
                }
                resp_values(r, f) = v;
            }
        });
    Kokkos::fence();
    (void)nx_; // (currently unused; reserved for bounds debug asserts)

    // -------------------------------------------------------------------
    // 4. Reverse Distributor: ship responses back to originating ranks.
    // -------------------------------------------------------------------
    Kokkos::View<int*, grace::default_space> resp_export(
        Kokkos::ViewAllocateWithoutInitializing("particle_fetch_resp_export"),
        n_recv);
    Kokkos::parallel_for("particle_fetch_build_resp_export",
        Kokkos::RangePolicy<exec_space>(0, n_recv),
        KOKKOS_LAMBDA(const int r) {
            resp_export(r) = resp_orig_rank(r);
        });
    Kokkos::fence();

    Cabana::Distributor<grace::default_space> rev(comm, resp_export);
    Cabana::AoSoA<response_t, grace::default_space, vlen>
        resp_recv("particle_fetch_resp_recv", rev.totalNumImport());
    Cabana::migrate(rev, resp, resp_recv);

    // -------------------------------------------------------------------
    // 5. Write back into the caller's output view, indexed by orig_idx.
    // -------------------------------------------------------------------
    auto recv_orig_idx = Cabana::slice<1>(resp_recv);
    auto recv_values   = Cabana::slice<2>(resp_recv);
    const std::size_t n_back = resp_recv.size();

    Kokkos::parallel_for("particle_fetch_writeback",
        Kokkos::RangePolicy<exec_space>(0, n_back),
        KOKKOS_LAMBDA(const int r) {
            const int dst = recv_orig_idx(r);
            for (int f = 0; f < N_FIELDS; ++f) {
                out(dst, f) = recv_values(r, f);
            }
        });
    Kokkos::fence();
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_CABANA

#endif // GRACE_PARTICLES_PARTICLE_AUX_FETCH_TPP
