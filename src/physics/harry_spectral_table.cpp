#include <grace/physics/harry_spectral_table.hh>
#include <hdf5.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <initializer_list>

namespace grace::harry {
namespace {
device_handle handle;
bool initialized=false;
struct file_guard {hid_t id;~file_guard(){if(id>=0)H5Fclose(id);}};
struct data {std::vector<hsize_t> dims;std::vector<double> values;};
data read(hid_t f,char const* name) {
    hid_t d=H5Dopen2(f,name,H5P_DEFAULT);
    if(d<0)throw std::runtime_error(std::string("Harry table: missing dataset ")+name);
    hid_t s=H5Dget_space(d);int n=H5Sget_simple_extent_ndims(s);
    data r;r.dims.resize(n);H5Sget_simple_extent_dims(s,r.dims.data(),nullptr);
    auto count=H5Sget_simple_extent_npoints(s);H5Sclose(s);
    r.values.resize(count);auto status=H5Dread(d,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,r.values.data());H5Dclose(d);
    if(status<0)throw std::runtime_error(std::string("Harry table: cannot read ")+name);
    for(double v:r.values)if(!std::isfinite(v))throw std::runtime_error(std::string("Harry table: nonfinite ")+name);
    return r;
}
int scalar(hid_t f,char const* n) {auto d=read(f,n);if(d.values.size()!=1)throw std::runtime_error("Harry scalar has wrong shape");return int(d.values[0]);}
Kokkos::View<double*> copy(std::vector<double> const& v,char const* name) {
    Kokkos::View<double*> result(name,v.size());auto h=Kokkos::create_mirror_view(result);
    for(std::size_t i=0;i<v.size();++i)h(i)=v[i];Kokkos::deep_copy(result,h);return result;
}
tensor load(hid_t f,char const* name,int channels,std::initializer_list<char const*> axes) {
    tensor t;t.rank=axes.size();t.channels=channels;
    std::vector<hsize_t> expected(t.rank+1);expected[0]=channels;int d=0;
    for(auto axis:axes) {
        auto a=read(f,axis);if(a.dims.size()!=1||a.values.size()<2)throw std::runtime_error("Harry table axis shape invalid");
        for(std::size_t i=1;i<a.values.size();++i)if(!(a.values[i]>a.values[i-1]))throw std::runtime_error("Harry table axis not strictly increasing");
        t.size[d]=a.values.size();expected[t.rank-d]=a.values.size();t.axes[d]=copy(a.values,axis);++d;
    }
    auto values=read(f,name);
    if(values.dims!=expected)throw std::runtime_error(std::string("Harry table: invalid shape for ")+name+" (expected channel plus all thermodynamic axes)");
    t.values=copy(values.values,name);return t;
}
bool shape(hid_t f,char const* name,std::vector<hsize_t> const& expected) {
    if(H5Lexists(f,name,H5P_DEFAULT)<=0)return false;
    hid_t d=H5Dopen2(f,name,H5P_DEFAULT),s=H5Dget_space(d);int n=H5Sget_simple_extent_ndims(s);
    std::vector<hsize_t> dims(n);H5Sget_simple_extent_dims(s,dims.data(),nullptr);H5Sclose(s);H5Dclose(d);return dims==expected;
}
}
bool is_harry_table(std::string const& path) {
    file_guard f{H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT)};
    if(f.id<0)throw std::runtime_error("Cannot open rate table: "+path);
    return H5Lexists(f.id,"kappa_a_st_table_nue",H5P_DEFAULT)>0 && H5Lexists(f.id,"energies_table",H5P_DEFAULT)>0;
}
void initialize_harry_table(std::string const& path,bool require_ild) {
    file_guard f{H5Fopen(path.c_str(),H5F_ACC_RDONLY,H5P_DEFAULT)};
    if(f.id<0)throw std::runtime_error("Cannot open Harry table: "+path);
    if(scalar(f.id,"store_lnvalues")!=1||scalar(f.id,"n_species")!=6||scalar(f.id,"weakhub_grey_method")!=-1)
        throw std::runtime_error("Harry reader requires six-species log-stored spectral table");
    device_handle h;h.n_groups=scalar(f.id,"n_groups");
    if(h.n_groups<2||h.n_groups>128)throw std::runtime_error("Invalid Harry energy count");
    auto e=read(f.id,"energies_table"),w=read(f.id,"energies_deps");
    if(e.values.size()!=std::size_t(h.n_groups)||w.values.size()!=std::size_t(h.n_groups+2))throw std::runtime_error("Harry energy arrays have wrong shape");
    std::vector<double> widths;
    for(int i=0;i<h.n_groups;++i) {
        if(!(e.values[i]>0)||!(w.values[i+1]>0)||(i&&e.values[i]<=e.values[i-1]))throw std::runtime_error("Invalid Harry energy grid");
        widths.push_back(w.values[i+1]);
    }
    h.energies=copy(e.values,"Harry energies");h.widths=copy(widths,"Harry widths");
    h.envelope.bins=decltype(h.envelope.bins)("Harry moment bounds",h.n_groups);
    auto bounds=Kokkos::create_mirror_view(h.envelope.bins);
    double sum_n=0,sum_J=0;
    for(int i=0;i<h.n_groups;++i) {
        double const energy=e.values[i],weight=energy*energy*widths[i];
        sum_n+=weight; sum_J+=weight*energy;
        bounds(i,0)=energy; bounds(i,1)=sum_n; bounds(i,2)=sum_J;
    }
    Kokkos::deep_copy(h.envelope.bins,bounds);
    h.envelope.flat_mean=sum_J/sum_n;
    char const* names[]={"kappa_a_st_table_nue","kappa_a_st_table_nue_bar","kappa_a_st_table_numu","kappa_a_st_table_numu_bar"};
    for(int s=0;s<4;++s)h.cc[s]=s<2?load(f.id,names[s],h.n_groups,{"logrho_IVetable","logtemp_IVetable","ye_IVetable","logymu_IVetable"}):load(f.id,names[s],h.n_groups,{"logrho_IVmutable","logtemp_IVmutable","ye_IVmutable","logymu_IVmutable"});
    h.scattering=load(f.id,"kappa_s_table_nu",6*h.n_groups,{"logrho_IIItable","logtemp_IIItable","yp_IIItable"});
    h.pair0=load(f.id,"PP_Phi0_ann_table",6*h.n_groups*h.n_groups,{"logtemp_IItable","logeta_e_IItable"});
    h.pair1=load(f.id,"PP_Phi1_ann_table",6*h.n_groups*h.n_groups,{"logtemp_IItable","logeta_e_IItable"});
    int n=h.n_groups*(h.n_groups+1);
    std::vector<hsize_t> dims={hsize_t(n),hsize_t(scalar(f.id,"IIIeta_mu")),hsize_t(scalar(f.id,"IIIeta_e")),hsize_t(scalar(f.id,"IIItemp_mu"))};
    h.ild_available=shape(f.id,"ILD_Phi0_out_table",dims)&&shape(f.id,"ILD_Phi1_out_table",dims);
    if(require_ild&&!h.ild_available)throw std::runtime_error("Harry ILD table is malformed: expected full energy-pair and three thermodynamic dimensions; cannot use tabulated muon decay");
    if(h.ild_available) {
        h.ild0=load(f.id,"ILD_Phi0_out_table",n,{"logtemp_IIItable_mu","logeta_e_IIItable","logeta_mu_IIItable"});
        h.ild1=load(f.id,"ILD_Phi1_out_table",n,{"logtemp_IIItable_mu","logeta_e_IIItable","logeta_mu_IIItable"});
    }
    handle=h;initialized=true;
}
device_handle const& get_harry_device_handle(){if(!initialized)throw std::runtime_error("Harry table not initialized");return handle;}
void finalize_harry_table(){handle={};initialized=false;}
}
