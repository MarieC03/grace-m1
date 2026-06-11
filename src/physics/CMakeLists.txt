add_subdirectory(eos)
register_grace_object(grace_grmhd grmhd.cpp )
register_grace_object(grace_grmhd_B_from_A grmhd_B_from_A.cpp )
register_grace_object(grace_b_field_injection b_field_injection.cpp )
register_grace_object_conditional(grace_m1 GRACE_ENABLE_M1 m1.cpp grace_weakhub_setup.cpp)
register_grace_object_conditional(grace_kadath_importer GRACE_ENABLE_FUKA import_kadath.cpp )

# bns_nurates is header-only: no .cpp, just add the include path to grace_objects
if(GRACE_ENABLE_M1)
    target_include_directories(grace_m1 PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/bns_nurates/include
    )
endif()
