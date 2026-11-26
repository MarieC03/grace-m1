#include <catch2/catch_test_macros.hpp>
#include <grace/system/checkpoint_schedule.hh>

TEST_CASE("Walltime checkpoint decisions agree despite rank-local clock skew", "[mpi][checkpoint]") {
    int const rank = parallel::mpi_comm_rank();
    REQUIRE(parallel::mpi_comm_size() >= 2);
    // Non-root ranks cross first: nobody may start a collective write yet.
    REQUIRE_FALSE(grace::checkpoint_walltime_due(rank == 0 ? 99. : 101., 100.));
    // Root crosses first: every rank must join, including ranks still below.
    REQUIRE(grace::checkpoint_walltime_due(rank == 0 ? 101. : 99., 100.));
    REQUIRE_FALSE(grace::checkpoint_walltime_due(100., 100.));
    // Subsequent deadlines can differ because ranks finish I/O at different times.
    REQUIRE_FALSE(grace::checkpoint_walltime_due(200., rank == 0 ? 201. : 199.));
    REQUIRE(grace::checkpoint_walltime_due(200., rank == 0 ? 199. : 201.));
    REQUIRE_FALSE(grace::collective_termination_due(rank!=0,false));
    REQUIRE(grace::collective_termination_due(rank==0,false));
    REQUIRE(grace::collective_termination_due(false,rank==1));
}
