#ifndef GRACE_SYSTEM_CHECKPOINT_SCHEDULE_HH
#define GRACE_SYSTEM_CHECKPOINT_SCHEDULE_HH

#include <grace/parallel/mpi_wrappers.hh>

namespace grace {
// Collective: only rank zero's clock/deadline controls the collective write.
// Rank-local elapsed times (and deadlines set after I/O) can straddle a step.
inline bool checkpoint_walltime_due(double elapsed, double deadline,
                                   sc_MPI_Comm comm = sc_MPI_COMM_WORLD) {
    int due = 0;
    if (parallel::mpi_comm_rank(comm) == 0) due = elapsed > deadline;
    parallel::mpi_bcast(&due, 1, 0, comm);
    return due != 0;
}
}
#endif
