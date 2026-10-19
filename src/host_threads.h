#ifndef DDLA_HOST_THREADS_H
#define DDLA_HOST_THREADS_H

// ---------------------------------------------------------------------------
// How many threads a CPU handle may use for its own host loops (geam today).
//
// COLLECTIVE over the communicator (one MPI_Comm_split_type), so it is called once, from ddlaSet, which is
// collective over exactly that communicator already; the result lives in the handle (DdlaStream::host_threads).
// A routine that only some ranks reach (a panel owner calling geam alone) therefore never has to communicate.
//
// The rule, shared with the CPU eigensolver kernels (README, "CPU threads"):
//   OMP_NUM_THREADS >= 1  -> that many (an OpenMP-built BLAS follows the same variable);
//   otherwise             -> the CPUs this process may run on divided by the ranks of its node, so an unset
//                            environment never oversubscribes;
//   DDLA_EIG_THREADS >= 1 -> overrides both.
// ---------------------------------------------------------------------------

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <thread>

#if defined(__linux__)
#include <sched.h>
#endif

namespace ddla {
namespace detail {

// CPUs this process may run on: the affinity mask honours a Slurm cpuset or an mpirun binding.
inline int host_allowed_cpus()
{
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

inline int host_thread_budget(MPI_Comm comm)
{
    int ranks_on_node = 1;
    MPI_Comm node = MPI_COMM_NULL;
    if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node) == MPI_SUCCESS) {
        MPI_Comm_size(node, &ranks_on_node);
        MPI_Comm_free(&node);
    }
    int t = 0;
    if (const char* env = std::getenv("OMP_NUM_THREADS")) t = std::atoi(env);
    if (t < 1) t = std::max(1, host_allowed_cpus() / std::max(1, ranks_on_node));
    if (const char* env = std::getenv("DDLA_EIG_THREADS")) {
        const int v = std::atoi(env);
        if (v >= 1) t = v;
    }
    return std::max(1, std::min(t, 512));
}

} // namespace detail
} // namespace ddla

#endif // DDLA_HOST_THREADS_H
