/**
 * @file lbm_velocity_mesh.hh
 * @brief Velocity-space interpolation for the fixed LBM stencil: the
 *        spherical Delaunay triangulation of the directions (host, once), a
 *        (theta, phi) lookup table and the device barycentric locator.
 * @date 2026-09-05
 *
 * @copyright This file is part of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef GRACE_PHYSICS_LBM_VELOCITY_MESH_HH
#define GRACE_PHYSICS_LBM_VELOCITY_MESH_HH

#include <grace_config.h>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/errors/error.hh>
#include <grace/physics/lbm_stencil.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace grace { namespace lbm {

/**
 * @brief Barycentric interpolation of the populations at an arbitrary
 *        direction, on the convex hull (= spherical Delaunay) triangulation of
 *        the stencil.  Exact at the stencil directions, so the curved kernel
 *        reproduces the flat one when the rays are straight.
 */
struct velocity_interp_t {
    int ntri = 0, nth = 0, nph = 0 ;
    Kokkos::View<int**>    tri ;   //!< (ntri,3) vertex (direction) indices, outward oriented
    Kokkos::View<int**>    adj ;   //!< (ntri,3) neighbour across the edge opposite vertex k
    Kokkos::View<int*>     lut ;   //!< (nth*nph) triangle containing the node direction
    Kokkos::View<double**> c ;     //!< (ndir,3) the stencil directions

    static double GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    det3(double const a[3], double const b[3], double const d[3])
    {
        return a[0]*(b[1]*d[2] - b[2]*d[1]) - a[1]*(b[0]*d[2] - b[2]*d[0]) + a[2]*(b[0]*d[1] - b[1]*d[0]) ;
    }

    /**
     * @brief Locate the unit direction u: the three stencil directions of the
     *        containing triangle and their barycentric weights (sum 1, >= 0).
     *
     * Lookup-table guess, then a walk across the edge opposite the most
     * negative weight (planar barycentric coordinates by Cramer's rule are
     * proportional to the gnomonic ones), then clamp and renormalise.
     */
    void GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
    locate(double const u[3], int idx[3], double lam[3]) const
    {
        constexpr double pi = 3.14159265358979323846 ;
        double const th = Kokkos::acos(Kokkos::fmin(1.0, Kokkos::fmax(-1.0, u[2]))) ;
        double ph = Kokkos::atan2(u[1], u[0]) ; if ( ph < 0. ) ph += 2.0*pi ;
        int it = static_cast<int>(th/pi*(nth-1) + 0.5) ; it = it < 0 ? 0 : (it > nth-1 ? nth-1 : it) ;
        int ip = static_cast<int>(ph/(2.0*pi)*nph + 0.5) % nph ;
        int t = lut(it*nph + ip) ;
        double p[3][3] ;
        for ( int hop = 0; hop < 8; ++hop ) {
            for ( int k = 0; k < 3; ++k ) { idx[k] = tri(t,k) ; for ( int a = 0; a < 3; ++a ) p[k][a] = c(idx[k],a) ; }
            double const l0 = det3(u, p[1], p[2]), l1 = det3(p[0], u, p[2]), l2 = det3(p[0], p[1], u) ;
            double const sum = l0 + l1 + l2 ;
            // sum <= 0: the triangle faces away from u; the raw determinants still
            // rank the vertices, so hop away from the most negative one
            double const n = sum > 0. ? sum : 1.0 ;
            lam[0] = l0/n ; lam[1] = l1/n ; lam[2] = l2/n ;
            int kmin = 0 ;
            if ( lam[1] < lam[kmin] ) kmin = 1 ;
            if ( lam[2] < lam[kmin] ) kmin = 2 ;
            if ( sum > 0. && lam[kmin] >= -1e-12 ) break ;
            int const nt = adj(t,kmin) ;
            if ( nt < 0 || nt == t ) break ;
            t = nt ;
        }
        double s = 0. ;
        for ( int k = 0; k < 3; ++k ) { lam[k] = Kokkos::fmax(lam[k], 0.0) ; s += lam[k] ; }
        for ( int k = 0; k < 3; ++k ) lam[k] /= s ;
    }
} ;

namespace detail {

struct hull_face_t { int v[3] ; } ;

/// Incremental convex hull of unit vectors -> outward-oriented triangles (host).
inline std::vector<hull_face_t> convex_hull_triangles(std::vector<std::array<double,3>> const& P)
{
    int const N = static_cast<int>(P.size()) ;
    auto sub  = [](std::array<double,3> const& a, std::array<double,3> const& b) { return std::array<double,3>{a[0]-b[0], a[1]-b[1], a[2]-b[2]} ; } ;
    auto det  = [](std::array<double,3> const& a, std::array<double,3> const& b, std::array<double,3> const& c) {
        return a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]) ; } ;
    auto norm2 = [](std::array<double,3> const& a) { return a[0]*a[0]+a[1]*a[1]+a[2]*a[2] ; } ;
    auto cross = [](std::array<double,3> const& a, std::array<double,3> const& b) {
        return std::array<double,3>{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]} ; } ;
    if ( N < 4 ) ERROR("LBM velocity mesh: need at least 4 directions.") ;

    // initial tetrahedron: farthest point, farthest from the line, farthest from the plane
    int p0 = 0, p1 = 1, p2 = -1, p3 = -1 ; double best = -1. ;
    for ( int n = 1; n < N; ++n ) { double d = norm2(sub(P[n],P[p0])) ; if ( d > best ) { best = d ; p1 = n ; } }
    best = -1. ;
    for ( int n = 0; n < N; ++n ) { double d = norm2(cross(sub(P[p1],P[p0]), sub(P[n],P[p0]))) ; if ( d > best ) { best = d ; p2 = n ; } }
    best = -1. ;
    for ( int n = 0; n < N; ++n ) { double d = std::fabs(det(sub(P[p1],P[p0]), sub(P[p2],P[p0]), sub(P[n],P[p0]))) ; if ( d > best ) { best = d ; p3 = n ; } }
    if ( best < 1e-12 ) ERROR("LBM velocity mesh: directions are coplanar.") ;
    // Consistently oriented tetrahedron: outward means away from its own
    // centroid (the origin need not lie inside this first tetrahedron).
    std::vector<hull_face_t> faces { {{p0,p1,p2}}, {{p0,p2,p3}}, {{p0,p3,p1}}, {{p1,p3,p2}} } ;
    std::array<double,3> const cen { 0.25*(P[p0][0]+P[p1][0]+P[p2][0]+P[p3][0]),
                                     0.25*(P[p0][1]+P[p1][1]+P[p2][1]+P[p3][1]),
                                     0.25*(P[p0][2]+P[p1][2]+P[p2][2]+P[p3][2]) } ;
    for ( auto& f : faces ) {
        auto const& a = P[f.v[0]] ;
        if ( det(sub(P[f.v[1]],a), sub(P[f.v[2]],a), sub(cen,a)) > 0 ) std::swap(f.v[1], f.v[2]) ;
    }

    std::vector<char> used(N, 0) ; used[p0] = used[p1] = used[p2] = used[p3] = 1 ;
    constexpr double eps = 1e-12 ;
    for ( int n = 0; n < N; ++n ) {
        if ( used[n] ) continue ;
        std::vector<char> vis(faces.size(), 0) ; int nvis = 0 ;
        for ( size_t f = 0; f < faces.size(); ++f ) {
            auto const& a = P[faces[f].v[0]] ;
            if ( det(sub(P[faces[f].v[1]],a), sub(P[faces[f].v[2]],a), sub(P[n],a)) > eps ) { vis[f] = 1 ; ++nvis ; }
        }
        if ( nvis == 0 ) ERROR("LBM velocity mesh: direction " << n << " is not outside any hull face (duplicate direction?).") ;
        // directed edges of visible faces whose twin lies in a non-visible face form the horizon
        std::unordered_map<long, size_t> edge_face ;
        for ( size_t f = 0; f < faces.size(); ++f ) for ( int e = 0; e < 3; ++e )
            edge_face[static_cast<long>(faces[f].v[e])*N + faces[f].v[(e+1)%3]] = f ;
        std::vector<hull_face_t> next ;
        for ( size_t f = 0; f < faces.size(); ++f ) if ( !vis[f] ) next.push_back(faces[f]) ;
        for ( size_t f = 0; f < faces.size(); ++f ) {
            if ( !vis[f] ) continue ;
            for ( int e = 0; e < 3; ++e ) {
                int const a = faces[f].v[e], b = faces[f].v[(e+1)%3] ;
                auto it = edge_face.find(static_cast<long>(b)*N + a) ;
                if ( it == edge_face.end() ) ERROR("LBM velocity mesh: open edge in the hull.") ;
                if ( !vis[it->second] ) next.push_back({{a, b, n}}) ;
            }
        }
        faces.swap(next) ;
        used[n] = 1 ;
    }
    if ( static_cast<int>(faces.size()) != 2*N - 4 )
        ERROR("LBM velocity mesh: hull has " << faces.size() << " triangles, expected 2N-4 = " << 2*N-4 << ".") ;
    for ( auto const& f : faces )
        if ( det(P[f.v[0]], P[f.v[1]], P[f.v[2]]) <= 0 ) ERROR("LBM velocity mesh: inward-oriented hull face.") ;
    return faces ;
}

} // namespace detail

/**
 * @brief Build the velocity-space interpolation tables for a stencil (host, once).
 */
inline velocity_interp_t build_velocity_interp(stencil_t const& st, int const nth, int const nph)
{
    auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, st.c) ;
    int const N = st.ndir ;
    std::vector<std::array<double,3>> P(N) ;
    for ( int d = 0; d < N; ++d ) P[d] = { hc(d,0), hc(d,1), hc(d,2) } ;
    auto const faces = detail::convex_hull_triangles(P) ;
    int const T = static_cast<int>(faces.size()) ;

    // adjacency: across the edge opposite vertex k lies the face owning the reversed edge
    std::unordered_map<long, int> edge_face ;
    for ( int f = 0; f < T; ++f ) for ( int e = 0; e < 3; ++e )
        edge_face[static_cast<long>(faces[f].v[e])*N + faces[f].v[(e+1)%3]] = f ;
    std::vector<int> adj(3*T, -1) ;
    for ( int f = 0; f < T; ++f ) for ( int k = 0; k < 3; ++k ) {
        int const a = faces[f].v[(k+1)%3], b = faces[f].v[(k+2)%3] ;
        auto it = edge_face.find(static_cast<long>(b)*N + a) ;
        if ( it == edge_face.end() ) ERROR("LBM velocity mesh: non-manifold edge.") ;
        adj[3*f+k] = it->second ;
    }

    // lookup table: containing triangle of every (theta, phi) node
    auto lam_of = [&](int f, std::array<double,3> const& u, double l[3]) {
        auto const& a = P[faces[f].v[0]] ; auto const& b = P[faces[f].v[1]] ; auto const& d = P[faces[f].v[2]] ;
        double const l0 = velocity_interp_t::det3(u.data(), b.data(), d.data()) ;
        double const l1 = velocity_interp_t::det3(a.data(), u.data(), d.data()) ;
        double const l2 = velocity_interp_t::det3(a.data(), b.data(), u.data()) ;
        double const s = l0 + l1 + l2 ;
        if ( s <= 0. ) { l[0] = l[1] = l[2] = -1. ; return -1e300 ; }   // triangle faces away from u
        l[0] = l0/s ; l[1] = l1/s ; l[2] = l2/s ;
        return std::fmin(l[0], std::fmin(l[1], l[2])) ;
    } ;
    std::vector<int> lut(static_cast<size_t>(nth)*nph, -1) ;
    for ( int it = 0; it < nth; ++it )
    for ( int ip = 0; ip < nph; ++ip ) {
        double const th = M_PI*it/(nth-1.0), ph = 2.0*M_PI*ip/nph ;
        std::array<double,3> const u { std::sin(th)*std::cos(ph), std::sin(th)*std::sin(ph), std::cos(th) } ;
        int best = -1 ; double bestmin = -1e300 ; double l[3] ;
        for ( int f = 0; f < T; ++f ) {
            double const m = lam_of(f, u, l) ;
            if ( m > bestmin ) { bestmin = m ; best = f ; }
            if ( m >= -1e-10 ) break ;
        }
        lut[static_cast<size_t>(it)*nph + ip] = best ;
    }

    velocity_interp_t vi ;
    vi.ntri = T ; vi.nth = nth ; vi.nph = nph ;
    vi.tri = Kokkos::View<int**>("lbm_vmesh_tri", T, 3) ;
    vi.adj = Kokkos::View<int**>("lbm_vmesh_adj", T, 3) ;
    vi.lut = Kokkos::View<int*> ("lbm_vmesh_lut", static_cast<size_t>(nth)*nph) ;
    vi.c   = st.c ;
    auto ht = Kokkos::create_mirror_view(vi.tri) ; auto ha = Kokkos::create_mirror_view(vi.adj) ; auto hl = Kokkos::create_mirror_view(vi.lut) ;
    for ( int f = 0; f < T; ++f ) for ( int k = 0; k < 3; ++k ) { ht(f,k) = faces[f].v[k] ; ha(f,k) = adj[3*f+k] ; }
    for ( size_t n = 0; n < lut.size(); ++n ) hl(n) = lut[n] ;
    Kokkos::deep_copy(vi.tri, ht) ; Kokkos::deep_copy(vi.adj, ha) ; Kokkos::deep_copy(vi.lut, hl) ;
    return vi ;
}

}} // namespace grace::lbm

#endif // GRACE_PHYSICS_LBM_VELOCITY_MESH_HH
