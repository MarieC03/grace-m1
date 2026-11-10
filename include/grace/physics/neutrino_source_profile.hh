#ifndef GRACE_NEUTRINO_SOURCE_PROFILE_HH
#define GRACE_NEUTRINO_SOURCE_PROFILE_HH
// Optional CPU diagnostic build. Inclusive/exclusive per-thread elapsed times,
// NOT application wall time; only exclusive categories may be added. Timers compile out
// in production builds.
#ifdef GRACE_LEPTON_PROFILE
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
#error "GRACE_LEPTON_PROFILE is a CPU-only diagnostic"
#endif
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace grace::source_profile {
enum category { decay_kernel, coupled, lagged, singlepass, thermal_solve, transport, thermal_kernel, grey_rates, count };
inline std::atomic<double> seconds[count]{},exclusive[count]{};
inline std::atomic<unsigned long long> calls[count]{}, lagged_fallbacks{}, lagged_iterations{};
inline std::atomic<unsigned long long> decay_cache_hits{},decay_cache_misses{};
inline std::atomic<unsigned long long> thermal_cache_hits{},thermal_cache_misses{};
inline std::atomic<unsigned long long> block_solves{},block_builds{},block_fallbacks{};
inline std::atomic<unsigned long long> grey_builds{},grey_refinements{};
struct budget {
    double v[10]{};
    KOKKOS_INLINE_FUNCTION budget& operator+=(budget const& b) { for(int i=0;i<10;++i) v[i]+=b.v[i]; return *this; }
};
inline void write_budget(int rank,double t,double h,budget const& b) {
    // A rank-local file avoids interleaved MPI stdout corrupting small errors.
    static std::FILE* file=nullptr;
    if(!file) { char name[80]; std::snprintf(name,sizeof(name),"source_budget_rank%d.dat",rank); file=std::fopen(name,"w"); }
    if(!file) { std::perror("source budget file"); std::abort(); }
    std::fprintf(file,"%.17g %.17g",t,h);
    for(int a=0;a<10;++a) std::fprintf(file," %.17g",b.v[a]);
    std::fprintf(file,"\n"); std::fflush(file);
}
struct scope {
    category id;
    inline static thread_local scope* top=nullptr;
    scope* parent;
    double child=0;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    explicit scope(category c):id(c),parent(top) { top=this; }
    ~scope() {
        double const elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        seconds[id].fetch_add(elapsed,std::memory_order_relaxed);
        exclusive[id].fetch_add(elapsed-child,std::memory_order_relaxed);
        calls[id].fetch_add(1,std::memory_order_relaxed);
        if(parent) parent->child+=elapsed;
        top=parent;
    }
};
inline void dump(int rank) {
    std::printf("GREY_KERNEL rank=%d builds=%llu refinements=%llu\n",rank,grey_builds.load(),grey_refinements.load());
    std::printf("BLOCK_JACOBIAN rank=%d solves=%llu builds=%llu fallbacks=%llu\n",rank,block_solves.load(),block_builds.load(),block_fallbacks.load());
    std::printf("THERMAL_CACHE rank=%d hits=%llu misses=%llu\n",rank,thermal_cache_hits.load(),thermal_cache_misses.load());
    std::printf("SOURCE_CACHE rank=%d hits=%llu misses=%llu\n",rank,decay_cache_hits.load(),decay_cache_misses.load());
    char const* names[]={"decay_kernel","coupled","lagged","singlepass","thermal_solve","transport","thermal_kernel","grey_rates"};
    for(int i=0;i<count;++i) std::printf("SOURCE_PROFILE rank=%d phase=%s seconds=%.17g exclusive=%.17g calls=%llu\n",rank,names[i],seconds[i].load(),exclusive[i].load(),calls[i].load());
    std::printf("SOURCE_PROFILE rank=%d lagged_fallbacks=%llu lagged_iterations=%llu\n",rank,lagged_fallbacks.load(),lagged_iterations.load());
    std::fflush(stdout);
}
}
#define GRACE_SOURCE_PROFILE(name) grace::source_profile::scope source_profile_scope(grace::source_profile::name)
#else
#define GRACE_SOURCE_PROFILE(name) ((void)0)
#endif
#endif
