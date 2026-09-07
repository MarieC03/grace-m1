# Define a function to add test targets and associated tests
function(add_mpi_test target_name source_file)
    add_executable(${target_name} ${source_file})


    # Set common target properties
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_compile_options(${target_name} PRIVATE -g)

    # Common linking libraries
    target_link_libraries(${target_name} PRIVATE 
    mpi_tests_main
    Catch2::Catch2
    $<TARGET_OBJECTS:grace_parallel>
    $<TARGET_OBJECTS:grace_error_handlers>
    MPI::MPI_CXX
    p4est::sc 
    spdlog::spdlog 
    ZLIB::ZLIB
    )
endfunction()

# Define a function to add test targets and associated tests
function(add_kokkos_test target_name source_file)
    add_executable(${target_name} ${source_file})


    # Set common target properties
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_compile_options(${target_name} PRIVATE -g)

    # Common linking libraries.  MPI::MPI_CXX is here for its INCLUDE path, not
    # its library: p4est's sc.h includes <mpi.h>, and kokkos_tests_main links MPI
    # PRIVATE, so nothing else hands these sources the MPI include directory.
    target_link_libraries(${target_name} PRIVATE 
    kokkos_tests_main
    Catch2::Catch2
    Kokkos::kokkos
    MPI::MPI_CXX
    p4est::sc 
    spdlog::spdlog 
    ZLIB::ZLIB
    )
endfunction()


# Define a function to add test targets and associated tests
function(add_parser_test target_name source_file)
    add_executable(${target_name} ${source_file})


    # Set common target properties
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_compile_options(${target_name} PRIVATE -g)

    # Common linking libraries
    target_link_libraries(${target_name} PRIVATE 
    parser_tests_main
    yaml_cpp::yaml
    Catch2::Catch2
    $<TARGET_OBJECTS:grace_parallel>
    $<TARGET_OBJECTS:grace_error_handlers>
    $<TARGET_OBJECTS:grace_singleton> 
    MPI::MPI_CXX
    p4est::sc 
    spdlog::spdlog 
    ZLIB::ZLIB
    )
endfunction()


function(add_grace_test target_name source_file)
    add_executable(${target_name} ${source_file})


    # Set common target properties
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_compile_options(${target_name} PRIVATE -g)

    # Common linking libraries
    target_link_libraries(${target_name} PRIVATE 
    p4est_tests_main
    Catch2::Catch2
    grace_objects
    yaml_cpp::yaml
    Kokkos::kokkos
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    p4est::sc
    p4est::p4est 
    spdlog::spdlog
    ZLIB::ZLIB
    HDF5::HDF5
    $<$<BOOL:${GRACE_ENABLE_PROFILING}>:GRACE_GPUProfiling>
    $<$<BOOL:${GRACE_ENABLE_VTK}>:VTK::VTK>
    $<$<BOOL:${GRACE_ENABLE_LORENE}>:LORENE::LORENE>
    # grace_objects carries import_kadath.cpp whenever FUKA is on, so the tests
    # need the same Kadath link the grace target has (CMakeLists.txt).
    $<$<BOOL:${GRACE_ENABLE_FUKA}>:Kadath::kadath>
    $<$<BOOL:${GRACE_ENABLE_TWO_PUNCTURES}>:TwoPunctures::TwoPunctures>
    )
    if ( GRACE_ENABLE_VTK )
            vtk_module_autoinit(
                TARGETS grace
                MODULES ${VTK_LIBRARIES}
            )
    endif()
endfunction()

function(add_standalone_test target_name source_file)
    add_executable(${target_name} ${source_file})


    # Set common target properties
    target_include_directories(${target_name} PRIVATE "${HEADER_DIR}")
    target_compile_options(${target_name} PRIVATE -g)

    # Common linking libraries
    target_link_libraries(${target_name} PRIVATE 
    Catch2::Catch2WithMain
    )
endfunction()