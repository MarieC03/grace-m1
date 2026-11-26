#ifndef GRACE_SYSTEM_CHECKPOINT_SCHEDULE_HH
#define GRACE_SYSTEM_CHECKPOINT_SCHEDULE_HH

#include <grace/parallel/mpi_wrappers.hh>

namespace grace {
// A local stop request may originate on any rank. The regular stopping
// criterion (including its clock) is evaluated on rank zero.
inline bool collective_termination_due(bool scheduled, bool requested,
                                      sc_MPI_Comm comm=sc_MPI_COMM_WORLD) {
    int const local=requested || (parallel::mpi_comm_rank(comm)==0 && scheduled);
    int stop=0;
    parallel::mpi_allreduce(&local,&stop,1,sc_MPI_MAX,comm);
    return stop!=0;
}
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
