#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/neutrino_pair_update.hh>
#include <grace/physics/m1.hh>
#include <grace/physics/id/m1_initial_data.hh>
#include <grace/physics/eas_neutrino_rates_analytic.hh>
#include <cmath>
using namespace grace;
using namespace grace::pairs;
using Catch::Matchers::WithinRel;

namespace {
void fd(grid const& g,double T,double eta,double* f,double& n,double& J) {
    n=J=0;
    for(int i=0;i<g.n;++i) {
        f[i]=occupation(g.e[i]/T-eta);
        n+=g.w[i]*f[i]; J+=g.w[i]*g.e[i]*f[i];
    }
}
material const matter{10,20,6e37,0.8,0.2};
}

TEST_CASE("Pair failures retain reconstruction context and bounded device samples", "[pairs][diagnostics]") {
    kernel k; diagnostic d;
    auto bad=matter; bad.T=-1;
    REQUIRE_FALSE(k.init(bad,{true,false,false},16,10,false,&d));
    REQUIRE(d.code==failure_code::kernel_input);
    REQUIRE(k.init(matter,{true,false,false},16,10,false,&d));
    REQUIRE(d.code==failure_code::none);
    double f[max_order];
    REQUIRE_FALSE(reconstruct(k.g,-1,1,f,&d));
    REQUIRE(d.code==failure_code::moment_input);
    REQUIRE(d.n==-1);
    REQUIRE(d.J==1);
    double n[4]={0,0,-1,0},J[4]={0,0,1,0},opacity[4];
    REQUIRE_FALSE(leptonic_transport_opacities(k,nullptr,true,n,J,opacity,&d));
    REQUIRE(d.species==2);
    REQUIRE(d.code==failure_code::moment_input);
    // Positive moments whose mean lies below every quadrature node cannot
    // be represented. This is a reconstruction failure, not a kernel error.
    REQUIRE_FALSE(reconstruct(k.g,1e20,1e20*k.g.e[0]*0.5,f,&d));
    REQUIRE(d.code==failure_code::moment_grid_range);
    REQUIRE(d.iteration==-1);
    REQUIRE(d.emin==k.g.e[0]);
    REQUIRE(d.emax==k.g.e[k.g.n-1]);
    REQUIRE(reconstruct(k.g,0,0,f,&d));
    REQUIRE(d.code==failure_code::none);

    failure_buffer buffer; buffer.allocate();
    Kokkos::parallel_for("pair_failure_capture_test",19,KOKKOS_LAMBDA(int cell) {
        failure_record r;
        r.i=cell; r.detail.code=failure_code::moment_input; r.detail.n=-cell-1;
        buffer.save(r);
    });
    auto count=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),buffer.count);
    auto records=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),buffer.records);
    REQUIRE(count()==19);
    for(int slot=0;slot<failure_buffer::capacity;++slot) {
        REQUIRE(records(slot).i>=0);
        REQUIRE(records(slot).i<19);
        REQUIRE(records(slot).detail.n==-records(slot).i-1);
        REQUIRE(records(slot).detail.code==failure_code::moment_input);
        for(int other=0;other<slot;++other) REQUIRE(records(other).i!=records(slot).i);
    }
}

TEST_CASE("Zero radiation ID matches the independent evolution floors", "[pairs][initial_data]") {
    m1_atmo_params_t atmo{};
    atmo.E_fl=1e-16; atmo.N_fl=1e-14; atmo.eps_fl=1; atmo.r_damping=50;
    m1_excision_params_t excision{};
    coord_array_t<GRACE_NSPACEDIM> coords("pair_id_coords",VEC(1,1,1),3,1);
    auto host=Kokkos::create_mirror_view(coords);
    for(double r:{1.,100.,1e6}) {
        host(VEC(0,0,0),0,0)=r;
        Kokkos::deep_copy(coords,host);
        zero_m1_id_t id(atmo,excision,coords);
        Kokkos::View<m1_id_t> result("pair_id_result");
        Kokkos::parallel_for("pair_id_check",1,KOKKOS_LAMBDA(int) {
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

TEST_CASE("Pair units match existing EAS", "[pairs]") {
    REQUIRE_THAT(energy_unit/time_unit,WithinRel(Q_mev_to_code(1,1),1e-14));
    REQUIRE_THAT(number_unit/time_unit,WithinRel(R_to_code(1,1),1e-14));
}

TEST_CASE("Number and energy face fluxes preserve a common spectrum in a static curved metric", "[pairs][transport]") {
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

TEST_CASE("Hunter moments exceed the pair quadrature support", "[pairs][diagnostics]") {
    // rank 2, iteration 0, q=80, (28,23,15), from the 2026-10-01 crash.
    double const n=3.71514358939047201e27,J=4.07411727181158377e34;
    for(int order:{16,24,32}) {
        grid g; g.init(order,2.79366770784541222e3);
        REQUIRE(J/n>g.e[order-1]);
        diagnostic d; double f[max_order];
        REQUIRE_FALSE(reconstruct(g,n,J,f,&d));
        REQUIRE(d.code==failure_code::moment_grid_range);
        REQUIRE(d.iteration==-1);
        // The logged antineutrino moments are representable on the same grid.
        REQUIRE(reconstruct(g,2.12570665646473441e31,2.87870240689971931e33,f,&d));
    }
}

TEST_CASE("Pair quadrature and moment reconstruction", "[pairs]") {
    grid g; g.init(32,10);
    double f[max_order],r[max_order],n,J;
    fd(g,10,0,f,n,J);
    REQUIRE_THAT(n,WithinRel(phase*1000*1.8030853547393914,2e-5));
    REQUIRE_THAT(J,WithinRel(phase*10000*5.682196976983475,3e-5));
    for(double eta: {-8.,0.,4.,12.}) {
        fd(g,10,eta,f,n,J);
        REQUIRE(reconstruct(g,n,J,r));
        double nr=0,jr=0;
        for(int i=0;i<g.n;++i) { nr+=g.w[i]*r[i]; jr+=g.w[i]*g.e[i]*r[i]; }
        REQUIRE_THAT(nr,WithinRel(n,3e-10)); REQUIRE_THAT(jr,WithinRel(J,3e-10));
    }
    REQUIRE(reconstruct(g,0,0,r));
    REQUIRE_FALSE(reconstruct(g,-1,1,r));
}

TEST_CASE("Blocked kernels obey detailed balance and kernel orientation", "[pairs]") {
    for(channels c: {channels{true,false,false},channels{false,true,false},channels{false,false,true}}) {
        kernel k; REQUIRE(k.init(matter,c,24,10));
        double f[max_order],b[max_order],n,J,nb,Jb;
        fd(k.g,10,2,f,n,J); fd(k.g,10,-2,b,nb,Jb);
        auto const s=evaluate(k,f,b);
        REQUIRE(s.emission[0]>0);
        REQUIRE(std::abs(s.energy[0])<3e-13*s.emission[0]);
        REQUIRE(std::abs(s.energy[1])<3e-13*s.emission[1]);
        REQUIRE(std::abs(s.number)<3e-13*s.emission_n[0]);
        REQUIRE_THAT(s.emission[0],WithinRel(s.loss[0],3e-13));
        REQUIRE_THAT(s.emission_n[1],WithinRel(s.loss_n[1],3e-13));
        for(int i=0;i<k.g.n;++i) f[i]=b[i]=0;
        auto const empty=evaluate(k,f,b);
        REQUIRE(empty.number>0); REQUIRE(empty.loss[0]==0);
        // One occupied partner is not enough for inverse annihilation.
        fd(k.g,16,1,f,n,J);
        auto const single=evaluate(k,f,b);
        REQUIRE(single.number>=0);
        REQUIRE(single.number<empty.number);
    }
    MyEOSParams eos{}; eos.temp=matter.T; eos.mu_e=matter.mu_e;
    PairKernelParams p{}; p.omega=12; p.omega_prime=27;
    auto const ref=PairKernels(&eos,&p);
    REQUIRE_THAT(absorption_kernel(matter,{true,false,false},12,27),WithinRel(ref.abs[id_nux]*1e-21,1e-12));
    REQUIRE_THAT(absorption_kernel(matter,{true,false,false},27,12),WithinRel(ref.abs[id_anux]*1e-21,1e-12));
}

TEST_CASE("Evolved pairs conserve number difference in an implicit M1 stage", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,false},16,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},1);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double f[max_order],n,J,old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        fd(k.g,12-s*3,0.3-s,f,n,J);
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
    }
    for(double dt: {1e-8,1e-6,1e-4,1e-2}) {
        INFO("dt [s] = " << dt);
        REQUIRE(implicit_update(k,metric,base,eas,old,out,dt*time_unit,res));
        REQUIRE(res<2e-8);
        REQUIRE_THAT(out[4]-out[9],WithinRel(old[4]-old[9],5e-10));
        REQUIRE(out[0]>0); REQUIRE(out[5]>0); REQUIRE(out[4]>0); REQUIRE(out[9]>0);
    }
}

TEST_CASE("Transverse plasmon event rate agrees with thermal photon decay", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{false,false,true},32,10));
    double f[max_order]={}; auto const s=evaluate(k,f,f);
    constexpr double alpha=1/137.035999084,gf=1.1663787e-11,hbar=6.582119569e-22;
    constexpr double cv=-0.5+2*0.2325;
    double const m2=4*alpha/(3*pairs::pi)*(400+pairs::pi*pairs::pi*100/3);
    double count=0,energy=0;
    // Independent one-dimensional integral over thermal photon momentum.
    for(int i=0;i<k.g.n;++i) {
        double const omega=std::sqrt(k.g.e[i]*k.g.e[i]+m2);
        double const gamma=gf*gf*cv*cv*m2*m2*m2/(48*pairs::pi*pairs::pi*alpha*hbar*omega);
        double const r=2*k.g.w[i]*gamma/std::expm1(omega/10);
        count+=r; energy+=r*omega/2;
    }
    REQUIRE_THAT(s.number,WithinRel(count,0.03));
    REQUIRE_THAT(s.energy[0],WithinRel(energy,0.03));
}

TEST_CASE("Pair fixed point and nonzero ordinary EAS share one stage", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,true},16,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double f[max_order],n,J,old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        fd(k.g,10,s==0?1:-1,f,n,J);
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
        eas[s][KAL]=2; eas[s][KANL]=3;
        eas[s][ETAL]=2*old[5*s]; eas[s][ETANL]=3*old[5*s+4];
    }
    REQUIRE(implicit_update(k,metric,base,eas,old,out,100,res));
    for(int s=0;s<2;++s) {
        REQUIRE_THAT(out[5*s],WithinRel(old[5*s],2e-9));
        REQUIRE_THAT(out[5*s+4],WithinRel(old[5*s+4],2e-9));
    }
}

TEST_CASE("Pairs emit into vacuum and preserve moving-fluid LTE", "[pairs]") {
    kernel k; REQUIRE(k.init(matter,{true,true,false},24,10));
    metric_array_t metric({1,0,0,1,0,1},{0,0,0},0.7);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double old[10]={},out[10],res;
    REQUIRE(implicit_update(k,metric,base,eas,old,out,0.1,res));
    REQUIRE(out[0]>0); REQUIRE(out[5]>0);
    REQUIRE(out[4]==out[9]);
    for(double v: {0.1,0.5}) {
        double const W=1/std::sqrt(1-v*v);
        for(int s=0;s<2;++s) {
            double f[max_order],n,J;
            fd(k.g,10,s==0?0.4:-0.4,f,n,J);
            old[5*s]=base[s][ERADL]=J*energy_unit*(4*W*W-1)/3;
            old[5*s+1]=base[s][FXL]=J*energy_unit*4*W*W*v/3;
            old[5*s+4]=base[s][NRADL]=n*number_unit*W;
            base[s][grace::ZXL]=W*v;
        }
        REQUIRE(implicit_update(k,metric,base,eas,old,out,10,res));
        for(int s=0;s<2;++s) {
            REQUIRE_THAT(out[5*s],WithinRel(old[5*s],3e-8));
            REQUIRE_THAT(out[5*s+1],WithinRel(old[5*s+1],3e-8));
            REQUIRE_THAT(out[5*s+4],WithinRel(old[5*s+4],3e-8));
        }
    }
}

TEST_CASE("Pair kernels execute on the selected Kokkos backend", "[pairs]") {
    Kokkos::View<double*> result("pair_backend_result",3);
    Kokkos::parallel_for("pair_backend_check",1,KOKKOS_LAMBDA(int) {
        kernel k;
        material m{8,12,1e37,1,0}; // also exercise pure-neutron composition
        result(0)=k.init(m,{true,true,true},16,8);
        if(result(0)!=1) return;
        double f[max_order]={},b[max_order]={};
        auto const s=evaluate(k,f,b);
        result(1)=s.number; result(2)=s.energy[0];
    });
    auto const r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
    REQUIRE(r(0)==1); REQUIRE(r(1)>0); REQUIRE(std::isfinite(r(2)));
}

TEST_CASE("GRACE pair source wiring preserves densitization and aggregate weights", "[pairs]") {
    var_array_t state("pair_state",1,1,1,N_EVOL_VARS,1);
    var_array_t next("pair_next",1,1,1,N_EVOL_VARS,1);
    var_array_t aux("pair_aux",1,1,1,N_AUX_VARS,1);
    auto st=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),state);
    auto ax=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),aux);
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_COWLING
    st(0,0,0,GXX_,0)=4; st(0,0,0,GYY_,0)=st(0,0,0,GZZ_,0)=1;
    #else
    st(0,0,0,GTXX_,0)=4; st(0,0,0,GTYY_,0)=st(0,0,0,GTZZ_,0)=1;
    st(0,0,0,CHI_,0)=1;
    #endif
    st(0,0,0,ALP_,0)=0.7;
    ax(0,0,0,PAIR_T_,0)=matter.T; ax(0,0,0,PAIR_MUE_,0)=matter.mu_e;
    ax(0,0,0,PAIR_NB_,0)=matter.nb; ax(0,0,0,PAIR_YN_,0)=matter.yn;
    ax(0,0,0,PAIR_YP_,0)=matter.yp; ax(0,0,0,PAIR_ACTIVE_,0)=3;
    kernel k; REQUIRE(k.init(matter,{true,true,false},16,10,true));
    double f[max_order],n,J; fd(k.g,7,0,f,n,J);
    #if GRACE_M1_NU_SPECIES >= 5
    constexpr int x=4,g=2;
    st(0,0,0,ERAD3_,0)=st(0,0,0,ERAD4_,0)=2*J*energy_unit;
    st(0,0,0,NRAD3_,0)=st(0,0,0,NRAD4_,0)=2*n*number_unit;
    #else
    constexpr int x=2,g=4;
    #endif
    st(0,0,0,m1_erad_idx<x>(),0)=2*g*J*energy_unit;
    st(0,0,0,m1_nrad_idx<x>(),0)=2*g*n*number_unit;
    Kokkos::deep_copy(state,st); Kokkos::deep_copy(next,state); Kokkos::deep_copy(aux,ax);
    m1_equations_system_t system(state,staggered_variable_arrays_t{},aux);
    scalar_array_t<GRACE_NSPACEDIM> idx;
    Kokkos::parallel_for("pair_wiring",1,KOKKOS_LAMBDA(int) {
        #if GRACE_M1_NU_SPECIES >= 5
        system.compute_pair_implicit_update<2,3>(0,VEC(0,0,0),idx,next,0.1,1,16);
        #endif
        system.compute_pair_implicit_update<x,x,g>(0,VEC(0,0,0),idx,next,0.1,1,16);
    });
    auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),next);
    metric_array_t metric({4,0,0,1,0,1},{0,0,0},0.7);
    REQUIRE(metric.sqrtg()==2);
    m1_prims_array_t base[2]{}; m1_eas_array_t eas[2]{};
    double old[10]={},out[10],res;
    for(int s=0;s<2;++s) {
        old[5*s]=base[s][ERADL]=J*energy_unit;
        old[5*s+4]=base[s][NRADL]=n*number_unit;
    }
    REQUIRE(implicit_update(k,metric,base,eas,old,out,0.1,res));
    REQUIRE_THAT(got(0,0,0,m1_erad_idx<x>(),0),WithinRel(2*g*out[0],2e-10));
    REQUIRE_THAT(got(0,0,0,m1_nrad_idx<x>(),0),WithinRel(2*g*out[4],2e-10));
    #if GRACE_M1_NU_SPECIES >= 5
    REQUIRE_THAT(got(0,0,0,NRAD3_,0),WithinRel(got(0,0,0,NRAD4_,0),2e-10));
    #endif
}
