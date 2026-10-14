// Regression tests for the M1 transport fixes that arrived with the pair-process merge
// (zero-radiation initial-data floors, curved number flux, FOFC conserved input).  They
// use no pair kernel, so they are built in every M1 configuration with >= 3 species.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/m1.hh>
#include <grace/physics/id/m1_initial_data.hh>
#include <cmath>
using namespace grace;
using Catch::Matchers::WithinRel;

TEST_CASE("Zero radiation ID matches the independent evolution floors", "[m1][initial_data]") {
    m1_atmo_params_t atmo{};
    atmo.E_fl=1e-16; atmo.N_fl=1e-14; atmo.eps_fl=1; atmo.r_damping=50;
    m1_excision_params_t excision{};
    coord_array_t<GRACE_NSPACEDIM> coords("id_floor_coords",VEC(1,1,1),3,1);
    auto host=Kokkos::create_mirror_view(coords);
    for(double r:{1.,100.,1e6}) {
        host(VEC(0,0,0),0,0)=r;
        Kokkos::deep_copy(coords,host);
        zero_m1_id_t id(atmo,excision,coords);
        Kokkos::View<m1_id_t> result("id_floor_result");
        Kokkos::parallel_for("id_floor_check",1,KOKKOS_LAMBDA(int) {
            result()=id(VEC(0,0,0),0);
        });
        auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
        REQUIRE(got().erad1==atmo.E_floor(r));
        REQUIRE(got().nrad1==atmo.N_floor(r));
        REQUIRE(got().nrad3==atmo.N_floor(r));
        #if GRACE_M1_NU_SPECIES >= 5
        REQUIRE(got().nrad4==atmo.N_floor(r));
        REQUIRE(got().nrad5==atmo.N_floor(r));
        #endif
    }
}

TEST_CASE("Number and energy face fluxes preserve a common spectrum in a static curved metric", "[m1][transport]") {
    var_array_t state("number_flux_state",4,4,4,N_EVOL_VARS,1);
    var_array_t aux("number_flux_aux",4,4,4,N_AUX_VARS,1);
    flux_array_t flux("number_flux",4,4,4,N_EVOL_VARS,3,1);
    scalar_array_t<GRACE_NSPACEDIM> dx("number_flux_dx",3,1);
    Kokkos::deep_copy(aux,0.0); Kokkos::deep_copy(dx,1.0);
    auto st=Kokkos::create_mirror_view(state);
    // Fluid at rest, zero shift. E=2, N=3, F_x=1; g_xx=4 implies
    // sqrt(gamma)=2 and F^x=1/4. Thus flux_E=0.35 and flux_N=0.525.
    for(int i=0;i<4;++i) for(int j=0;j<4;++j) for(int l=0;l<4;++l) {
        #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
        st(i,j,l,GXX_,0)=4; st(i,j,l,GYY_,0)=st(i,j,l,GZZ_,0)=1;
        #else
        st(i,j,l,GTXX_,0)=4; st(i,j,l,GTYY_,0)=st(i,j,l,GTZZ_,0)=1;
        st(i,j,l,CHI_,0)=1;
        #endif
        st(i,j,l,ALP_,0)=0.7;
        // getflux consumes the normalized reconstruction state: E, N/E, F_i/E.
        st(i,j,l,ERAD3_,0)=2;
        st(i,j,l,NRAD3_,0)=1.5;
        st(i,j,l,FRADX3_,0)=0.5;
    }
    Kokkos::deep_copy(state,st);
    m1_equations_system_t system(state,staggered_variable_arrays_t{},aux);
    Kokkos::parallel_for("curved_number_flux",1,KOKKOS_LAMBDA(int) {
        system.compute_x_flux<godunov_reconstructor_t,2>(0,VEC(2,2,2),flux,flux,dx,0.01,1);
    });
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),flux);
    REQUIRE_THAT(got(2,2,2,ERAD3_,0,0),WithinRel(0.35,1e-12));
    REQUIRE_THAT(got(2,2,2,NRAD3_,0,0),WithinRel(0.525,1e-12));
    REQUIRE_THAT(got(2,2,2,NRAD3_,0,0)/got(2,2,2,ERAD3_,0,0),WithinRel(1.5,1e-12));
}

TEST_CASE("FOFC consumes conserved radiation and preserves donor fluxes", "[m1][transport][fofc]") {
    var_array_t conserved("fofc_conserved",4,4,4,N_EVOL_VARS,1);
    var_array_t normalized("fofc_normalized",4,4,4,N_EVOL_VARS,1);
    var_array_t aux("fofc_aux",4,4,4,N_AUX_VARS,1);
    flux_array_t actual("fofc_flux",4,4,4,N_EVOL_VARS,3,1);
    flux_array_t reference("donor_flux",4,4,4,N_EVOL_VARS,3,1);
    scalar_array_t<GRACE_NSPACEDIM> dx("fofc_dx",3,1);
    Kokkos::deep_copy(aux,0.0); Kokkos::deep_copy(dx,1.0);
    for(bool curved:{false,true}) {
        auto st=Kokkos::create_mirror(conserved);
        auto norm=Kokkos::create_mirror(normalized);
        Kokkos::deep_copy(st,0.0); Kokkos::deep_copy(norm,0.0);
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) for(int k=0;k<4;++k) {
            // A hot donor next to an almost empty cell, as after the first
            // implicit emission stage. Different cell metrics also test that
            // each donor is undensitized with its own volume factor.
            double const sg=curved ? 1.0+0.1*(i+j+k) : 1.0;
            #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
            st(i,j,k,GXX_,0)=sg*sg; st(i,j,k,GYY_,0)=st(i,j,k,GZZ_,0)=1;
            #else
            st(i,j,k,GTXX_,0)=sg*sg; st(i,j,k,GTYY_,0)=st(i,j,k,GTZZ_,0)=1;
            st(i,j,k,CHI_,0)=1;
            #endif
            st(i,j,k,ALP_,0)=1;
            for(int v=0;v<N_EVOL_VARS;++v) norm(i,j,k,v,0)=st(i,j,k,v,0);
            for(int s=0;s<GRACE_M1_NU_SPECIES;++s) {
                int const off=s*GRACE_N_M1_VARS;
                bool const hot=(i+j+k<6);
                double const E=hot ? (s+1)*1e-8 : 1e-16;
                double const N=hot ? 10*E : 1e-14;
                st(i,j,k,ERAD1_+off,0)=sg*E;
                st(i,j,k,NRAD1_+off,0)=sg*N;
                norm(i,j,k,ERAD1_+off,0)=E;
                norm(i,j,k,NRAD1_+off,0)=N/E;
                for(int d=0;d<3;++d) {
                    double const f=curved ? 0.1*(d+1) : 0.0;
                    st(i,j,k,FRADX1_+off+d,0)=sg*f*E;
                    norm(i,j,k,FRADX1_+off+d,0)=f;
                }
            }
        }
        Kokkos::deep_copy(conserved,st); Kokkos::deep_copy(normalized,norm);
        m1_equations_system_t raw(conserved,staggered_variable_arrays_t{},aux);
        m1_equations_system_t prepared(normalized,staggered_variable_arrays_t{},aux);
        auto check_species=[&](auto species) {
            constexpr int sp=decltype(species)::value;
            Kokkos::parallel_for("fofc_donor_regression",1,KOKKOS_LAMBDA(int) {
                raw.compute_fofc_flux<0,sp>(0,VEC(2,2,2),actual,dx,0.01,1);
                raw.compute_fofc_flux<1,sp>(0,VEC(2,2,2),actual,dx,0.01,1);
                raw.compute_fofc_flux<2,sp>(0,VEC(2,2,2),actual,dx,0.01,1);
                prepared.compute_x_flux<donor_cell_reconstructor_t,sp>(0,VEC(2,2,2),reference,reference,dx,0.01,1);
                prepared.compute_y_flux<donor_cell_reconstructor_t,sp>(0,VEC(2,2,2),reference,reference,dx,0.01,1);
                prepared.compute_z_flux<donor_cell_reconstructor_t,sp>(0,VEC(2,2,2),reference,reference,dx,0.01,1);
            });
            auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),actual);
            auto ref=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),reference);
            int const off=sp*GRACE_N_M1_VARS;
            for(int d=0;d<3;++d) {
                for(int v=0;v<GRACE_N_M1_VARS;++v) {
                    double const expected=ref(2,2,2,ERAD1_+off+v,d,0);
                    REQUIRE_THAT(got(2,2,2,ERAD1_+off+v,d,0),
                        Catch::Matchers::WithinAbs(expected,std::max(1e-28,1e-11*std::abs(expected))));
                }
                if(!curved) {
                    // Isotropic flat-space HLLE: diffusion carries the same
                    // donor E and N jumps, including into a floor-level cell.
                    double const expected_ratio=(10*(sp+1)*1e-8-1e-14)/((sp+1)*1e-8-1e-16);
                    REQUIRE_THAT(got(2,2,2,NRAD1_+off,d,0)/got(2,2,2,ERAD1_+off,d,0),
                                 WithinRel(expected_ratio,1e-10));
                }
            }
        };
        check_species(std::integral_constant<int,0>{});
        #if GRACE_M1_NU_SPECIES >= 3
        check_species(std::integral_constant<int,1>{});
        check_species(std::integral_constant<int,2>{});
        #endif
        #if GRACE_M1_NU_SPECIES >= 5
        check_species(std::integral_constant<int,3>{});
        check_species(std::integral_constant<int,4>{});
        #endif
        auto unchanged=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),conserved);
        for(int s=0;s<GRACE_M1_NU_SPECIES;++s) for(int v=0;v<GRACE_N_M1_VARS;++v)
            REQUIRE(unchanged(1,2,2,ERAD1_+s*GRACE_N_M1_VARS+v,0)==st(1,2,2,ERAD1_+s*GRACE_N_M1_VARS+v,0));
    }
}
