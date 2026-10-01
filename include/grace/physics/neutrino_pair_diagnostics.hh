#ifndef GRACE_NEUTRINO_PAIR_DIAGNOSTICS_HH
#define GRACE_NEUTRINO_PAIR_DIAGNOSTICS_HH

#include <Kokkos_Core.hpp>

namespace grace::pairs {

enum class failure_code {
    none, kernel_input, kernel_backend, kernel_element, decay_kernel,
    moment_input, reconstruction_jacobian, reconstruction_line_search,
    reconstruction_iterations, implicit_solve
};

// Optional diagnostics: no printing inside Newton or its rejected trial steps.
struct diagnostic {
    failure_code code = failure_code::none;
    int species = -1, iteration = -1, node = -1, partner_node = -1;
    double residual = -1, n = 0, J = 0, emin = 0, emax = 0;
    double kernel_value = 0, energy = 0, partner_energy = 0;
};

inline char const* failure_name(failure_code code) {
    switch(code) {
    case failure_code::none: return "none";
    case failure_code::kernel_input: return "kernel-input";
    case failure_code::kernel_backend: return "kernel-backend-unavailable";
    case failure_code::kernel_element: return "kernel-element";
    case failure_code::decay_kernel: return "decay-kernel";
    case failure_code::moment_input: return "moment-input";
    case failure_code::reconstruction_jacobian: return "reconstruction-jacobian";
    case failure_code::reconstruction_line_search: return "reconstruction-line-search";
    case failure_code::reconstruction_iterations: return "reconstruction-iterations";
    case failure_code::implicit_solve: return "implicit-solve";
    }
    return "unknown";
}

struct failure_record {
    diagnostic detail;
    long long q = -1;
    int i = 0, j = 0, k = 0, species_a = -1, species_b = -1;
    int order = 0, active = 0, multiplicity = 1;
    double T = 0, mu_e = 0, nb = 0, yn = 0, yp = 0, mu_mu = 0, scale = 0;
    // Filled after the failed launch, before any further evolution.
    double xyz[3] = {}, hydro[4] = {}, moments[2][5] = {}, fluid[2][2] = {};
};

// A bounded sample per MPI rank. The failed launch completes; the host prints
// and flushes records before aborting. Never continue evolution after failure.
struct failure_buffer {
    static constexpr int capacity = 4;
    Kokkos::View<int> count;
    Kokkos::View<failure_record*> records;

    void allocate() {
        count = Kokkos::View<int>("pair_failure_count");
        records = Kokkos::View<failure_record*>("pair_failure_records",capacity);
        Kokkos::deep_copy(count,0);
    }

    KOKKOS_INLINE_FUNCTION void save(failure_record const& record) const {
        // Direct unit-test callers may not have installed a host collector.
        if(!count.data()) Kokkos::abort("Pair failure without host diagnostic collector");
        int const slot = Kokkos::atomic_fetch_add(&count(),1);
        if(slot < capacity) records(slot) = record;
    }
};

} // namespace grace::pairs
#endif
