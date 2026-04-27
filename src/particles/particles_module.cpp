/**
 * @file particles_module.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Top-level particle subsystem entry point. Phase 0 skeleton.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_CABANA

#include <grace/particles/particles_module.hh>
#include <grace/particles/particle_storage.hh>

namespace grace {
namespace particles {

class particles_module_impl_t {
  public:
    bool enabled = false;
    tracer_aosoa_t<> particles{"grace_tracers", 0};
};

particles_module_t::particles_module_t()
  : _impl(new particles_module_impl_t())
{}

particles_module_t::~particles_module_t() {
    delete _impl;
}

particles_module_t& particles_module_t::get() {
    static particles_module_t instance;
    return instance;
}

void particles_module_t::initialize() {
    _impl->enabled = true;
}

void particles_module_t::finalize() {
    _impl->particles = tracer_aosoa_t<>{"grace_tracers", 0};
    _impl->enabled = false;
}

bool particles_module_t::enabled() const noexcept {
    return _impl->enabled;
}

std::size_t particles_module_t::local_count() const noexcept {
    return _impl->particles.size();
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_CABANA
