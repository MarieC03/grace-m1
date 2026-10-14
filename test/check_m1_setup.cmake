# Run the production fatal-error paths of check_m1_setup() / the table-content resolver in a
# child; verify the rejection AND its reason, so an unrelated crash cannot pass as validation.
set(cases log-missing bad-keyword decay-table sink-eos sink-suppression)
set(reasons "needs Weakhub's companion log"
            "must be auto, none, thermal or thermal\\+decay"
            "already carries inverse lepton decay"
            "needs eos.eos_type: leptonic"
            "dilute_muon_suppression: true")
foreach(i RANGE 0 4)
    list(GET cases ${i} case)
    list(GET reasons ${i} reason)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "GRACE_M1_SETUP_INVALID=${case}" "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false"
        "${TEST_PROGRAM}" "[m1-setup-invalid]"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if("${result}" STREQUAL "0" OR NOT "${output}${error}" MATCHES "${reason}")
        message(FATAL_ERROR "Wrong rejection for ${case}: ${result}\n${output}\n${error}")
    endif()
    message(STATUS "Rejected incompatible M1 setup: ${case}")
endforeach()
