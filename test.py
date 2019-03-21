import cluster_tools
import subprocess
import concurrent.futures
import time

# "Worker" functions.
def square(n):
    return n * n

def hostinfo():
    return subprocess.check_output('uname -a', shell=True)

def sleep(duration):
    time.sleep(duration)
    return True


def test_submit():
    """Square some numbers on remote hosts!
    """

    def square_numbers(executor):
        with executor:
            job_count = 5
            job_range = range(job_count)
            print("submitting job")
            futures = [executor.submit(square, n) for n in job_range]
            for future, job_index in zip(futures, job_range):
                assert future.result() == square(job_index)

    executors = [
        cluster_tools.get_executor("slurm", debug=True, keep_logs=True),
        cluster_tools.get_executor("multiprocessing", 5),
        cluster_tools.get_executor("sequential")
    ]
    for exc in executors:
        square_numbers(exc)


def example_2():
    """Get host identifying information about the servers running
    our jobs.
    """
    with cluster_tools.SlurmExecutor(False) as executor:
        futures = [executor.submit(hostinfo) for n in range(15)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result().strip())


def example_3():
    with cluster_tools.SlurmExecutor(False) as exc:
        print(list(exc.map(square, [5, 7, 11, 12, 13, 14, 15, 16, 17])))


def sleep_example():
    executor = cluster_tools.get_executor("slurm")
    # executor = cluster_tools.get_executor("multiprocessing", 5)
    # executor = cluster_tools.get_executor("sequential")
    with executor:
        print(list(executor.map(sleep, [10, 10, 10])))

