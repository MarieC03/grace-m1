/**
 * @file read_leptonic_4d_table.hh
 * @brief  Host-side reader for the 4D leptonic EOS table.
 *         Returns a ready-to-use leptonic_eos_4d_t object.
 * @date   2025
 *
 * @copyright This file is part of GRACE.  GPL-3 or later.
 */
#ifndef GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH
#define GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH

#include <grace_config.h>
#include <grace/physics/eos/leptonic_eos_4d.hh>
#include <grace/utils/grace_utils.hh>
#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief Read a 4D leptonic EOS table and return the EOS object.
 *
 * Parameters are read from the parameter file under the "eos/leptonic_4d"
 * namespace.
 */
grace::leptonic_eos_4d_t read_leptonic_4d_table() ;

} /* namespace grace */

#endif /* GRACE_PHYS_EOS_READ_LEPTONIC_4D_TABLE_HH */
