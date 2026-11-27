#include <grace/physics/guo_nnbrem.hh>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cstdlib>

using namespace grace::guo_nnbrem;
TEST_CASE("Guo response has explicit domain status and detailed balance", "[guo][pairs]") {
    device_handle empty;
    REQUIRE(empty.local(10,.01,.1).state==status::uninitialized);
    local_response l;l.state=status::ok;
    for(int i=0;i<nw;++i)l.log_s[i]=std::log(2.);
    auto k=device_handle::kernel(l,20,10,.01);
    REQUIRE(k.state==status::ok);
    // Constant response S=2, n_free=0.01 fm^-3, |g_A|=1.27641.
    REQUIRE(k.annihilation0/5.9639482239343745e-28==Catch::Approx(1.).epsilon(1e-12));
    REQUIRE(k.production0/k.annihilation0==Catch::Approx(std::exp(-2.)).epsilon(1e-13));
    REQUIRE(k.annihilation1/k.annihilation0==Catch::Approx(-1./9).epsilon(1e-13));
    REQUIRE(device_handle::kernel(l,800,10,.01).state==status::energy_outside);
    REQUIRE(device_handle::kernel(l,.01,10,.01).annihilation0==0);
    REQUIRE(device_handle::kernel(l,20,0,.01).state==status::thermo_outside);
}
TEST_CASE("Published Guo response table values and interpolation", "[guo][pairs]") {
    char const* filename=std::getenv("GRACE_GUO_TABLE");
    if(!filename){SKIP("Set GRACE_GUO_TABLE to run the archived author-data integration test");}
    auto h=load(filename);
    // Execute the lookup on the configured Kokkos backend, including GPU builds.
    Kokkos::View<double*> results("Guo test results",8);
    Kokkos::parallel_for("Guo published values",1,KOKKOS_LAMBDA(int){
        status s;
        auto l=h.local(2,1e-4,0);
        results(0)=device_handle::response(l,.05011872336272722,s);
        results(1)=device_handle::response(l,.06309573444801933,s);
        results(2)=device_handle::response(l,.05623413251903491,s);
        auto mid=h.local(2,1e-4,.01);
        results(3)=device_handle::response(mid,.05011872336272722,s);
        results(4)=h.local(1,1e-4,0).state==status::thermo_outside;
        results(5)=h.local(20,1e-5,0).state==status::thermo_outside;
        results(6)=h.local(50,1,.5).state==status::ok;
        results(7)=h.local(20,.01,.51).state==status::thermo_outside;
    });
    auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),results);
    REQUIRE(r(0)==Catch::Approx(1.1002).epsilon(1e-12));
    REQUIRE(r(1)==Catch::Approx(.69636).epsilon(1e-12));
    REQUIRE(r(2)==Catch::Approx(std::sqrt(1.1002*.69636)).epsilon(1e-12));
    REQUIRE(r(3)==Catch::Approx(std::sqrt(1.1002*7.0237)).epsilon(1e-12));
    for(int i=4;i<8;++i)REQUIRE(r(i)==1);
}
