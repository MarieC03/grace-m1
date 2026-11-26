#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <grace/physics/harry_spectral_table.hh>
#include <hdf5.h>
#include <filesystem>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <array>
using namespace grace::harry;
namespace {
void put(hid_t f,char const* name,std::vector<hsize_t> dims,std::vector<double> values) {
 auto s=H5Screate_simple(dims.size(),dims.data(),nullptr);auto d=H5Dcreate2(f,name,H5T_NATIVE_DOUBLE,s,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
 REQUIRE(d>=0);REQUIRE(H5Dwrite(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,values.data())>=0);H5Dclose(d);H5Sclose(s);
}
void field(hid_t f,char const* name,int rank,int channels) {
 std::vector<hsize_t> dims(rank+1,2);dims[0]=channels;std::vector<double> v(channels*(1<<rank));
 for(int c=0;c<channels;++c)for(int i=0;i<(1<<rank);++i)v[c*(1<<rank)+i]=100*c+i;
 put(f,name,dims,v);
}
void fixture(char const* path,bool valid) {
 auto f=H5Fcreate(path,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);REQUIRE(f>=0);
 for(auto name:{"store_lnvalues","IIIeta_e","IIIeta_mu","IIItemp_mu","n_groups","n_species","weakhub_grey_method"})
 put(f,name,{1},{std::string(name)=="store_lnvalues"?1.:std::string(name)=="n_species"?6.:std::string(name)=="weakhub_grey_method"?-1.:2.});
 for(auto name:{"logrho_IVetable","logtemp_IVetable","ye_IVetable","logymu_IVetable","logrho_IVmutable","logtemp_IVmutable","ye_IVmutable","logymu_IVmutable","logrho_IIItable","logtemp_IIItable","yp_IIItable","logtemp_IItable","logeta_e_IItable","logtemp_IIItable_mu","logeta_e_IIItable","logeta_mu_IIItable"}) put(f,name,{2},{0.,std::string(name)=="yp_IIItable"?4.:1.});
 put(f,"energies_table",{2},{1.,3.});put(f,"energies_deps",{4},{99.,2.,4.,99.});
 for(auto name:{"kappa_a_st_table_nue","kappa_a_st_table_nue_bar","kappa_a_st_table_numu","kappa_a_st_table_numu_bar"})field(f,name,4,2);
 field(f,"kappa_s_table_nu",3,12);field(f,"PP_Phi0_ann_table",2,24);field(f,"PP_Phi1_ann_table",2,24);
 if(valid){field(f,"ILD_Phi0_out_table",3,6);field(f,"ILD_Phi1_out_table",3,6);}
 else{put(f,"ILD_Phi0_out_table",{2},{0.,1.});put(f,"ILD_Phi1_out_table",{2},{0.,1.});}
 H5Fclose(f);
}
}
TEST_CASE("Harry reader preserves axes, channel ordering and logarithmic storage","[harry][hdf5]") {
 char const* p="harry-reader-fixture.h5";fixture(p,true);REQUIRE(is_harry_table(p));initialize_harry_table(p,true);
 auto h=get_harry_device_handle();REQUIRE(h.n_groups==2);REQUIRE(h.ild_available);
 Kokkos::View<double*> result("reader results",8);
 Kokkos::parallel_for("check reader",1,KOKKOS_LAMBDA(int){
   double x[4]={.5,.5,.5,.5};result(0)=h.cc[0].interpolate(1,x);result(1)=h.pair0.interpolate(23,x);
   result(2)=h.ild_log_out(Kokkos::exp(.5),Kokkos::exp(.5),Kokkos::exp(.5),1,0,1);
   result(3)=h.ild_log_out(Kokkos::exp(.5),Kokkos::exp(.5),Kokkos::exp(.5),1,1,0);
   result(4)=h.widths(0);result(5)=h.widths(1);x[0]=-10;result(6)=h.cc[0].contains(x)?1:0;
   x[0]=.2;x[1]=.3;x[2]=.4;x[3]=.7;result(7)=h.cc[0].interpolate(1,x);
 });
 auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
 REQUIRE(r(0)==107.5);REQUIRE(r(1)==2301.5);REQUIRE(r(2)==403.5);REQUIRE(r(2)==r(3));REQUIRE(r(4)==2);REQUIRE(r(5)==4);REQUIRE(r(6)==0);REQUIRE_THAT(r(7),Catch::Matchers::WithinAbs(108.,1e-12));
 finalize_harry_table();std::filesystem::remove(p);
}
TEST_CASE("Malformed ILD cannot be enabled","[harry][hdf5]") {
 char const* p="harry-reader-invalid.h5";fixture(p,false);initialize_harry_table(p,false);
 REQUIRE_FALSE(get_harry_device_handle().ild_available);finalize_harry_table();
 REQUIRE_THROWS_WITH(initialize_harry_table(p,true),"Harry ILD table is malformed: expected full energy-pair and three thermodynamic dimensions; cannot use tabulated muon decay");
 std::filesystem::remove(p);
}
TEST_CASE("Unused pair moment still requires the full energy and matter axes","[harry][hdf5]") {
 const char* path="harry-reader-invalid-pair.h5";
 fixture(path,true);
 auto file=H5Fopen(path,H5F_ACC_RDWR,H5P_DEFAULT);
 REQUIRE(file>=0);
 REQUIRE(H5Ldelete(file,"PP_Phi1_ann_table",H5P_DEFAULT)>=0);
 put(file,"PP_Phi1_ann_table",{2},{0.,1.});
 H5Fclose(file);
 REQUIRE_THROWS_WITH(initialize_harry_table(path,true),"Harry table: invalid shape for PP_Phi1_ann_table");
 std::filesystem::remove(path);
}
TEST_CASE("Finite-grid FOFC bounds use number capacity and positive-temperature spectra","[harry][transport]") {
 char const* p="harry-moment-fixture.h5";fixture(p,true);initialize_harry_table(p,true);
 auto bound=get_harry_device_handle().envelope;
 // E=(1,3), weights=(2,36). Filling three particles costs at least 5.
 Kokkos::View<int*> result("transport bounds",6);
 Kokkos::parallel_for("finite-grid envelope",1,KOKKOS_LAMBDA(int){
    result(0)=bound.contains(3.,5.); result(1)=bound.contains(3.,4.99);
    result(2)=bound.contains(1.,2.); result(3)=bound.contains(1.,2.95);
    result(4)=bound.contains(38.,110.); result(5)=bound.contains(38.1,110.);
 });
 auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
 for(int i=0;i<6;++i) REQUIRE(r(i)==(i%2==0));
 finalize_harry_table();std::filesystem::remove(p);
}
TEST_CASE("Harry domain checks allow endpoint roundoff but reject extrapolation","[harry][hdf5]") {
 char const* p="harry-boundary-fixture.h5";fixture(p,true);initialize_harry_table(p,true);
 auto h=get_harry_device_handle();Kokkos::View<int*> result("bounds",3);
 Kokkos::parallel_for("bounds",1,KOKKOS_LAMBDA(int){
   double x[4]={-2e-16,.5,.5,1+2e-16};result(0)=h.cc[0].contains(x);
   x[0]=-1e-10;result(1)=h.cc[0].contains(x);
   x[0]=.5;x[3]=1+1e-10;result(2)=h.cc[0].contains(x);
 });
 auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
 REQUIRE(r(0)==1);REQUIRE(r(1)==0);REQUIRE(r(2)==0);
 finalize_harry_table();std::filesystem::remove(p);
}
TEST_CASE("Optional real Harry file passes schema validation","[harry][external]") {
 auto path=std::getenv("GRACE_HARRY_TEST_TABLE");if(!path){SUCCEED("Set GRACE_HARRY_TEST_TABLE for external data audit");return;}
 initialize_harry_table(path,false);auto h=get_harry_device_handle();REQUIRE(h.n_groups==16);
 auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),h.energies);
 auto w=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),h.widths);
 // Diagnostic only: finite-bin coverage is not an accuracy assertion.
 for(double T:{.1,.5,1.,3.,10.,30.,100.}) {
    double n=0,u=0;for(int i=0;i<h.n_groups;++i){double f=1/(1+std::exp(std::min(e(i)/T,700.)));n+=w(i)*e(i)*e(i)*f;u+=w(i)*e(i)*e(i)*e(i)*f;}
    std::cout<<"HARRY_GRID_AUDIT T="<<T<<" eta=0 number_ratio="<<n/(1.8030853547393914*T*T*T)<<" energy_ratio="<<u/(5.682196976983475*T*T*T*T)<<"\n";
 }
 finalize_harry_table();
}

#include <grace/physics/eas_harry_rates.hh>
#include "harry_fd_reference.hh"
TEST_CASE("Released Harry ILD and ES agree with raw HDF5 cells","[harry][external]") {
    auto path=std::getenv("GRACE_HARRY_ILD_TEST_TABLE");
    if(!path) { SKIP("Set GRACE_HARRY_ILD_TEST_TABLE to a complete spectral release"); }
    initialize_harry_table(path,true);
    auto h=get_harry_device_handle();
    REQUIRE(h.n_groups==16);
    auto file=H5Fopen(path,H5F_ACC_RDONLY,H5P_DEFAULT);
    REQUIRE(file>=0);
    // HDF5 axes after the channel run in reverse Fortran order. The arithmetic
    // mean of eight on-disk log values is the independent midpoint reference.
    for(bool ild:{false,true}) {
        auto const tensor=ild?h.ild0:h.scattering;
        std::array<hsize_t,3> index{};
        std::array<double,3> midpoint{};
        for(int d=0;d<3;++d) {
            auto axis=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),tensor.axes[d]);
            index[d]=axis.size()/2;
            midpoint[d]=.5*(axis(index[d])+axis(index[d]+1));
        }
        auto ds=H5Dopen2(file,ild?"ILD_Phi0_out_table":"kappa_s_table_nu",H5P_DEFAULT);
        REQUIRE(ds>=0);
        auto space=H5Dget_space(ds);
        hsize_t count[4]={1,2,2,2};
        auto mem=H5Screate_simple(4,count,nullptr);
        for(int s=0;s<(ild?2:6);++s) {
            int const i=10,j=6;
            hsize_t channel=ild?s*136+i*(i+1)/2+j:s*16+i;
            hsize_t start[4]={channel,index[2],index[1],index[0]};
            REQUIRE(H5Sselect_hyperslab(space,H5S_SELECT_SET,start,nullptr,count,nullptr)>=0);
            double raw[8];
            REQUIRE(H5Dread(ds,H5T_NATIVE_DOUBLE,mem,space,H5P_DEFAULT,raw)>=0);
            double expected=0; for(double v:raw) expected+=v/8;
            double a=std::exp(midpoint[0]),b=std::exp(midpoint[1]);
            double c=ild?std::exp(midpoint[2]):midpoint[2];
            Kokkos::View<double*> result("real table lookup",1);
            Kokkos::parallel_for("interpolate HDF5 midpoint",1,KOKKOS_LAMBDA(int){
                result(0)=ild?h.ild_log_out(a,b,c,s,j,i):h.scatter_log(s,i,a,b,c);
            });
            auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
            REQUIRE_THAT(host(0),Catch::Matchers::WithinAbs(expected,2e-12));
        }
        if(!ild) {
            // Integrate all six stored ES spectra independently at zero
            // degeneracy. GRACE's fifth species evolves the two tau flavours.
            auto e=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),h.energies);
            auto w=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),h.widths);
            grace::fugacity_state matter{};
            matter.rho_code=std::exp(midpoint[0]); matter.temp_mev=std::exp(midpoint[1]);
            matter.ymu=.001; matter.ye=midpoint[2]-matter.ymu;
            matter.mu_e=grace::nu_constants::Qnp; matter.dilute_cc_target=true;
            matter.eta_cc_mu=0.; matter.mass_scale=1.;
            double direct[6]{};
            for(int s=0;s<6;++s) for(int i=0;i<16;++i) {
                hsize_t start[4]={hsize_t(s*16+i),index[2],index[1],index[0]};
                REQUIRE(H5Sselect_hyperslab(space,H5S_SELECT_SET,start,nullptr,count,nullptr)>=0);
                double raw[8]; REQUIRE(H5Dread(ds,H5T_NATIVE_DOUBLE,mem,space,H5P_DEFAULT,raw)>=0);
                double ln_kappa=0.; for(double v:raw) ln_kappa+=v/8;
                direct[s]+=w(i)*std::pow(e(i),3)*std::exp(ln_kappa)/(1.+std::exp(e(i)/matter.temp_mev));
            }
            direct[4]=.5*(direct[4]+direct[5]);
            Kokkos::View<double*> result("grey ES and tau emission",8);
            Kokkos::parallel_for("grey ES",1,KOKKOS_LAMBDA(int) {
                auto rates=grace::compute_all_species_harry(h,matter,nullptr);
                for(int s=0;s<5;++s) result(s)=rates.out[s].kappa_s;
                result(5)=rates.out[4].eta_E;result(6)=rates.out[4].eta_N;result(7)=rates.out[4].kappa_a;
            });
            auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
            // Integral of x^3/(exp(x)+1) from zero to infinity.
            double const norm=std::pow(matter.temp_mev,4)*7.*std::pow(std::acos(-1.),4)/120.;
            for(int s=0;s<5;++s)
                REQUIRE_THAT(host(s),Catch::Matchers::WithinRel(grace::kappa_to_code(direct[s]/norm,1.),2e-12));
            REQUIRE(host(5)==0.); REQUIRE(host(6)==0.); REQUIRE(host(7)==0.);
        }
        H5Sclose(mem);H5Sclose(space);H5Dclose(ds);
    }
    H5Fclose(file);finalize_harry_table();
}
TEST_CASE("Harry FD moments match independent high-precision integrals","[harry][cc]") {
 Kokkos::View<double**> result("FD reference",13,2);
 for(int j=0;j<13;++j) {
   double const eta=fd_reference_eta[j];
   Kokkos::parallel_for("FD moments",1,KOKKOS_LAMBDA(int){
      auto m=grace::harry_fd_moments(eta);result(j,0)=m.f2;result(j,1)=m.f3;
   });
 }
 auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
 for(int j=0;j<13;++j)for(int k=0;k<2;++k)
   REQUIRE_THAT(r(j,k),Catch::Matchers::WithinRel(fd_reference_moments[j][k],5e-14));
}
TEST_CASE("Harry CC has the continuum LTE target and remains finite in dilute tails","[harry][cc]") {
 char const* p="harry-cc-fixture.h5";fixture(p,true);initialize_harry_table(p,true);
 auto h=get_harry_device_handle();for(int s=0;s<4;++s)Kokkos::deep_copy(h.cc[s].values,-500.);
 Kokkos::deep_copy(h.scattering.values,-5.);
 for(double eta:{-50.,0.,50.,-800.,800.}) {
   grace::fugacity_state F{};F.temp_mev=std::exp(.5);F.rho_code=std::exp(.5);F.ye=.5;F.ymu=std::exp(.5);
   F.mu_e=eta*F.temp_mev+grace::nu_constants::Qnp;F.mu_mu=grace::nu_constants::Qnp;F.dilute_cc_target=true;F.eta_cc_mu=0.;
   Kokkos::View<double*> out("CC rates",5);
   Kokkos::parallel_for("CC LTE",1,KOKKOS_LAMBDA(int){
      auto r=grace::compute_all_species_harry(h,F,nullptr).out[0];
      out(0)=r.eta_E;out(1)=r.eta_N;out(2)=r.kappa_a;out(3)=r.kappa_n;out(4)=r.kappa_s;
   });
   auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
   for(int k=0;k<5;++k)REQUIRE(std::isfinite(r(k)));
   REQUIRE(r(2)>0);REQUIRE(r(3)>0);REQUIRE(r(4)>0);
   if(std::abs(eta)<=50) {
      // Independent high-precision integral values, not the historical FD fit.
      double fd2=eta<0?3.8574996959278356e-22:eta==0?1.8030853547393915:41831.160073351486;
      double fd3=eta<0?1.1572499087783506e-21:eta==0?5.682196976983476:1574848.3698953157;
      double const pref=4*grace::nu_constants::pi*grace::nu_constants::clight/std::pow(grace::nu_constants::hc_mevcm,3);
      double targetE=grace::Q_mev_to_code(pref*std::pow(F.temp_mev,4)*fd3,F.mass_scale)/grace::kappa_to_code(1,F.mass_scale);
      double targetN=grace::R_to_code(pref*std::pow(F.temp_mev,3)*fd2,F.mass_scale)/grace::kappa_to_code(1,F.mass_scale);
      REQUIRE_THAT(r(0)/r(2),Catch::Matchers::WithinRel(targetE,1e-12));
      REQUIRE_THAT(r(1)/r(3),Catch::Matchers::WithinRel(targetN,1e-12));
   }
 }
 finalize_harry_table();std::filesystem::remove(p);
}

TEST_CASE("Harry CC and scattering switch off below their temperature or density domain","[harry][cc]") {
 char const* p="harry-cc-domain.h5";fixture(p,true);initialize_harry_table(p,true);auto h=get_harry_device_handle();
 for(int s=0;s<4;++s)Kokkos::deep_copy(h.cc[s].values,-5.);Kokkos::deep_copy(h.scattering.values,-5.);
 for(int low=0;low<2;++low) {
   grace::fugacity_state F{};F.temp_mev=low==0?.5:std::exp(.5);F.rho_code=low==1?.5:std::exp(.5);F.ye=.5;F.ymu=std::exp(.5);
   F.mu_e=grace::nu_constants::Qnp;F.mu_mu=grace::nu_constants::Qnp;F.dilute_cc_target=true;F.eta_cc_mu=0.;
   Kokkos::View<double*> out("inactive rates",1);
   Kokkos::parallel_for("inactive",1,KOKKOS_LAMBDA(int){auto rates=grace::compute_all_species_harry(h,F,nullptr);double sum=0;for(int s=0;s<5;++s){auto r=rates.out[s];sum+=r.eta_E+r.eta_N+r.kappa_a+r.kappa_n+r.kappa_s;}out(0)=sum;});
   auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);REQUIRE(r(0)==0.);
 }
 finalize_harry_table();std::filesystem::remove(p);
}

TEST_CASE("Harry CC uses the enabled dilute-muon target instead of the raw EOS target","[harry][cc]") {
 char const* p="harry-cc-target.h5";fixture(p,true);initialize_harry_table(p,true);auto h=get_harry_device_handle();
 for(int s=0;s<4;++s)Kokkos::deep_copy(h.cc[s].values,-5.);Kokkos::deep_copy(h.scattering.values,-5.);
 for(bool enabled:{false,true}) {
   grace::fugacity_state F{};F.temp_mev=std::exp(.5);F.rho_code=std::exp(.5);F.ye=.5;F.ymu=std::exp(.5);
   F.mu_e=grace::nu_constants::Qnp;F.mu_mu=grace::nu_constants::Qnp+7*F.temp_mev;F.dilute_cc_target=enabled;F.eta_cc_mu=-3.;
   Kokkos::View<double*> out("muon target",4);
   Kokkos::parallel_for("muon target",1,KOKKOS_LAMBDA(int){auto rates=grace::compute_all_species_harry(h,F,nullptr);for(int s=2;s<4;++s){auto r=rates.out[s];out(2*(s-2))=r.eta_E/r.kappa_a;out(2*(s-2)+1)=r.eta_N/r.kappa_n;}});
   auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
   for(int s=2;s<4;++s) {
      double eta=(enabled?-3.:7.)*(s==2?1:-1);
      int j=0;while(j<13 && fd_reference_eta[j]!=eta)++j;REQUIRE(j<13);
      double const fd2=fd_reference_moments[j][0]*(eta<0?std::exp(eta):1);
      double const fd3=fd_reference_moments[j][1]*(eta<0?std::exp(eta):1);
      double const pref=4*grace::nu_constants::pi*grace::nu_constants::clight/std::pow(grace::nu_constants::hc_mevcm,3);
      double targetE=grace::Q_mev_to_code(pref*std::pow(F.temp_mev,4)*fd3,F.mass_scale)/grace::kappa_to_code(1,F.mass_scale);
      double targetN=grace::R_to_code(pref*std::pow(F.temp_mev,3)*fd2,F.mass_scale)/grace::kappa_to_code(1,F.mass_scale);
      REQUIRE_THAT(r(2*(s-2)),Catch::Matchers::WithinRel(targetE,1e-12));
      REQUIRE_THAT(r(2*(s-2)+1),Catch::Matchers::WithinRel(targetN,1e-12));
   }
 }
 finalize_harry_table();std::filesystem::remove(p);
}

TEST_CASE("Harry grey absorption follows radiation energy while matter emission is unchanged","[harry][cc]") {
 char const* p="harry-cc-spectrum.h5";fixture(p,true);initialize_harry_table(p,true);auto h=get_harry_device_handle();
 for(int s=0;s<4;++s)Kokkos::deep_copy(h.cc[s].values,-5.);Kokkos::deep_copy(h.scattering.values,-5.);
 grace::fugacity_state F{};F.temp_mev=std::exp(.5);F.rho_code=std::exp(.5);F.ye=.5;F.ymu=std::exp(.5);
 F.mu_e=grace::nu_constants::Qnp;F.mu_mu=grace::nu_constants::Qnp;F.dilute_cc_target=true;F.eta_cc_mu=0.;
 Kokkos::View<double*> out("nonLTE CC",8);
 Kokkos::parallel_for("nonLTE CC",1,KOKKOS_LAMBDA(int){
   double eps[5];for(int s=0;s<5;++s)eps[s]=2*F.temp_mev/grace::fermi::FDR<2,3>::get(0.);
   auto eq=grace::compute_all_species_harry(h,F,nullptr).out[0];
   auto hot=grace::compute_all_species_harry(h,F,eps).out[0];
   out(0)=eq.eta_E;out(1)=hot.eta_E;out(2)=eq.eta_N;out(3)=hot.eta_N;
   out(4)=eq.kappa_a;out(5)=hot.kappa_a;out(6)=eq.kappa_n;out(7)=hot.kappa_n;
 });
 auto r=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);
 REQUIRE(r(0)==r(1));REQUIRE(r(2)==r(3));
 REQUIRE(r(4)>0);REQUIRE(r(5)>0);REQUIRE(r(6)>0);REQUIRE(r(7)>0);
 REQUIRE(std::abs(r(4)-r(5))>1e-6*r(4));REQUIRE(std::abs(r(6)-r(7))>1e-6*r(6));
 finalize_harry_table();std::filesystem::remove(p);
}
