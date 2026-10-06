"""Thread-count selection that respects what the job is actually allowed to use."""
import os


def AvailableThreads():
    """Return (n_threads, source) for FFT/numexpr work.

    Priority:
      1. PYUL_NUM_THREADS environment variable (explicit override, trusted as given).
      2. The CPUs this process may actually run on: the scheduler affinity mask
         (honours SLURM/cgroup/taskset binding), falling back to the node's
         core count where affinity is unavailable (e.g. macOS).
      3. Capped further by SLURM_CPUS_PER_TASK when set.
    A node can report 100+ cores while a job is allotted a handful; using the
    node total oversubscribes the allocation and runs much slower.
    """
    override = os.environ.get("PYUL_NUM_THREADS")
    if override:
        return max(1, int(override)), "PYUL_NUM_THREADS"

    if hasattr(os, "process_cpu_count"):          # Python >= 3.13
        n, source = os.process_cpu_count(), "process_cpu_count"
    elif hasattr(os, "sched_getaffinity"):        # Linux
        n, source = len(os.sched_getaffinity(0)), "CPU affinity"
    else:
        n, source = os.cpu_count() or 1, "node core count"

    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and int(slurm) < n:
        n, source = int(slurm), "SLURM_CPUS_PER_TASK"

    return max(1, n), source
