/**
 * @file grace/memory/new_delete_allocator.hh
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief Bring <code> std::malloc </code> and <code> std::free </code> into grace's allocator system
 * @version 0.1
 * @date 2023-03-10
 * 
 * Simple allocator for Host memory, used to abstract dynamic allocation policies
 * from singleton code. This system does not support memory allocation on the Device
 * and should only be used for data structures meant to live on the Host such as 
 * the grid hierarchy and its connectivity.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference 
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
 * 
 */
#ifndef GRACE_NEW_DELETE_ALLOCATOR_HH
#define GRACE_NEW_DELETE_ALLOCATOR_HH

#include <cstdlib>
#include <memory_resource> 
#include <iostream> 

namespace memory 
{
//**************************************************************************************************

//**************************************************************************************************
/**
 * @brief Thin wrapper around <code>malloc</code>, <code>realloc</code> and <code>free</code>.
 */
//**************************************************************************************************
class new_delete_allocator 
{   
    public:
        //**************************************************************************************************
        /**
         * @brief Allocate memory using global allocator.
         * 
         * @param size size (in bytes) of requested allocation.
         * @param alignment aligment.
         * @return void* pointer to allocated memory
         */
        //**************************************************************************************************
        [[nodiscard]] static inline void*
        allocate(std::size_t size,
                 std::size_t alignment )
        {
            // ===================================================================
            // [macOS aligned_alloc fix - Method A]
            // -------------------------------------------------------------------
            // std::aligned_alloc (C11/POSIX) requires the alignment to be a
            // power of two that is at least sizeof(void*), and the requested
            // size to be an integral multiple of that alignment. glibc silently
            // tolerates violations (e.g. alignment == alignof(int) == 4), which
            // is why an under-aligned singleton constructs fine on Linux/HPC but
            // returns NULL here on macOS -> placement-new on a null buffer ->
            // this == nullptr -> EXC_BAD_ACCESS on the first member write.
            // Clamp the alignment up to the platform default new alignment and
            // round the size up to a multiple of it so aligned_alloc behaves
            // identically on macOS and Linux.
            // ===================================================================
            std::size_t const min_align = alignof(std::max_align_t) ;
            if ( alignment < min_align ) alignment = min_align ;
            size = (size + alignment - 1) & ~(alignment - 1) ;
            return std::aligned_alloc(alignment, size) ;
        }
        //**************************************************************************************************

        //**************************************************************************************************
        /**
         * @brief deallocate pointer allocated by this class.
         * 
         * @param ptr pointer to bre free'd
         * @param size size (in bytes) of memory owned by <code> ptr </code>.
         * @param alignment aligment of <code> ptr </code>.
         */ 
        static inline void deallocate( void* ptr, 
                                       std::size_t size,
                                       std::size_t alignment ) noexcept 
        {
            std::free(ptr) ; 
        } 
        //**************************************************************************************************
} ; 
//**************************************************************************************************
//**************************************************************************************************
}
#endif 