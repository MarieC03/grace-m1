# Run the production fatal-error path in a child; verify rejection AND its reason,
# so an unrelated crash cannot masquerade as successful setup validation.
set(cases legacy eos dilute electrons table betaeq order)
set(reasons "muon_decay requires pair_treatment"
            "muon_decay requires the leptonic EOS"
            "muon_decay requires dilute_muon_suppression=false"
            "muon_decay requires dilute_muon_suppression=false"
            "Declare weakhub_muon_decay_content"
            "New pair treatments currently require betaeq_policy: off"
            "muon_decay_kernel_order must be between")
foreach(i RANGE 0 6)
    list(GET cases ${i} case)
    list(GET reasons ${i} reason)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env
        "GRACE_DECAY_INVALID=${case}" "OMP_NUM_THREADS=1" "OMP_PROC_BIND=false"
        "${TEST_PROGRAM}" "[decay-invalid]"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if("${result}" STREQUAL "0" OR NOT "${output}${error}" MATCHES "${reason}")
        message(FATAL_ERROR "Wrong rejection for ${case}: ${result}\n${output}\n${error}")
    endif()
    message(STATUS "Rejected incompatible decay setup: ${case}")
endforeach()
