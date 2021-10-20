"""Abstracts access to a Slurm cluster via its command-line tools.
"""
import re
import os
import sys
from cluster_tools.util import chcall, random_string, call
from .cluster_executor import ClusterExecutor
import logging
import asyncio
from typing import Union
from concurrent import futures

SLURM_STATES = {
    "Failure": [
        "CANCELLED",
        "BOOT_FAIL",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "STOPPED",
        "TIMEOUT",
    ],
    "Success": ["COMPLETED"],
    "Ignore": [
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "PENDING",
        "RESV_DEL_HOLD",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
        "RESIZING",
    ],
    "Unclear": ["SUSPENDED", "REVOKED", "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT"],
}

SLURM_QUEUE_CHECK_INTERVAL = 1 if "pytest" in sys.modules else 60


class SlurmExecutor(ClusterExecutor):
    @staticmethod
    def get_job_array_index():
        return os.environ.get("SLURM_ARRAY_TASK_ID", None)

    @staticmethod
    def get_job_array_id():
        return os.environ.get("SLURM_ARRAY_JOB_ID", None)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("SLURM_JOB_ID")

    @staticmethod
    def format_log_file_name(jobid, suffix=".stdout"):
        return "slurmpy.{}.log{}".format(str(jobid), suffix)

    @classmethod
    def get_job_id_string(cls):
        job_id = cls.get_current_job_id()
        job_array_id = cls.get_job_array_id()
        job_array_index = cls.get_job_array_index()

        # This variable needs to be kept in sync with the job_id_string variable in the
        # inner_submit function.
        job_id_string = (
            job_id if job_array_index is None else f"{job_array_id}_{job_array_index}"
        )
        return job_id_string

    @staticmethod
    def get_max_array_size():
        max_array_size = 2 ** 32
        # See https://unix.stackexchange.com/a/364615
        stdout, stderr, exit_code = call(
            "scontrol show config | sed -n '/^MaxArraySize/s/.*= *//p'"
        )
        if exit_code == 0:
            max_array_size = int(stdout.decode("utf8"))
            # TODO: Debug
            print(f"Slurm MaxArraySize is {max_array_size}.")
        else:
            logging.warning(
                f"Slurm's MaxArraySize couldn't be determined. Reason: {stderr}"
            )
        return max_array_size

    @staticmethod
    def get_max_submit_jobs():
        max_submit_jobs = 2 ** 32
        # Check whether there is a limit per user
        stdout_user, stderr_user, _ = call(
            "sacctmgr list -n user $USER withassoc format=maxsubmitjobsperuser"
        )
        try:
            max_submit_jobs = int(stdout_user.decode("utf8"))
        except:
            # If there is no limit per user check whether there is a general limit
            stdout_qos, stderr_qos, _ = call(
                "sacctmgr list -n qos normal format=maxsubmitjobsperuser"
            )
            try:
                max_submit_jobs = int(stdout_qos.decode("utf8"))
            except:
                logging.warning(
                    f"Slurm's MaxSubmitJobsPerUser couldn't be determined. Reason: {stderr_user}\n{stderr_qos}"
                )
                return max_submit_jobs
        # TODO: Debug
        print(f"Slurm MaxSubmitJobsPerUser is {max_submit_jobs}.")
        return max_submit_jobs

    @staticmethod
    def get_number_of_submitted_jobs():
        number_of_submitted_jobs = 0
        # --array so that each job array element is displayed on a separate line and -h to hide the header
        stdout, stderr, exit_code = call("squeue --array -u $USER -h | wc -l")
        if exit_code == 0:
            number_of_submitted_jobs = int(stdout.decode("utf8"))
            # TODO: Debug
            print(f"Number of currently submitted jobs is {number_of_submitted_jobs}.")
        else:
            logging.warning(
                f"Number of currently submitted jobs couldn't be determined. Reason: {stderr}"
            )
        return number_of_submitted_jobs

    def submit_text(self, job):
        """Submits a Slurm job represented as a job file string. Returns
        the job ID.
        """

        filename = self.get_temp_file_path("_temp_slurm{}.sh".format(random_string()))
        with open(filename, "w") as f:
            f.write(job)
        job_id, stderr = chcall("sbatch --parsable {}".format(filename))
        os.unlink(filename)

        if len(stderr) > 0:
            logging.warning(f"Submitting batch job emitted warnings: {stderr}")

        return int(job_id)

    def inner_submit(
        self, cmdline, job_name=None, additional_setup_lines=[], job_count=None
    ):
        """Starts a Slurm job that runs the specified shell command line."""

        # These place holders will be replaced by sbatch, see https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E
        # This variable needs to be kept in sync with the job_id_string variable in the
        # get_job_id_string function.
        job_id_string = "%j" if job_count is None else "%A_%a"
        log_path = self.format_log_file_path(self.cfut_dir, job_id_string)

        job_resources_lines = []
        if self.job_resources is not None:
            for resource, value in self.job_resources.items():
                job_resources_lines += ["#SBATCH --{}={}".format(resource, value)]

        max_array_size = self.get_max_array_size()
        max_submit_jobs = self.get_max_submit_jobs()
        # Only ever submit at most a third of max_submit_jobs at once.
        # This way, multiple programs submitting slurm jobs will not block each other
        # by "occupying" more than half of the number of submittable jobs.
        batch_size = min(max_array_size, max_submit_jobs // 3)

        scripts = []
        job_id_futures = []
        ranges = []
        number_of_jobs = job_count if job_count is not None else 1
        for job_index_start in range(0, number_of_jobs, batch_size):
            # job_index_end is inclusive
            job_index_end = min(job_index_start + batch_size, number_of_jobs) - 1
            array_index_end = job_index_end - job_index_start

            job_array_line = ""
            if job_count is not None:
                job_array_line = "#SBATCH --array=0-{}".format(array_index_end)

            script_lines = (
                [
                    "#!/bin/sh",
                    "#SBATCH --output={}".format(log_path),
                    '#SBATCH --job-name "{}"'.format(job_name),
                    job_array_line,
                ]
                + job_resources_lines
                + [
                    *additional_setup_lines,
                    "srun {} {}".format(cmdline, job_index_start),
                ]
            )

            job_id_futures.append(futures.Future())
            scripts.append("\n".join(script_lines))
            ranges.append((job_index_start, job_index_end + 1))

        job_sizes = [end - start for start, end in ranges]
        # Replace with asyncio.run when upgrading to python 3.7+
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.submit_once_allowed(scripts, job_sizes, job_id_futures)
        )

        return job_id_futures, ranges

    async def submit_once_allowed(self, scripts, job_sizes, futures):
        max_submit_jobs = self.get_max_submit_jobs()

        for script, job_size, future in zip(scripts, job_sizes, futures):
            while self.get_number_of_submitted_jobs() + job_size > max_submit_jobs:
                await asyncio.sleep(SLURM_QUEUE_CHECK_INTERVAL)
            job_id = self.submit_text(script)
            future.set_result(job_id)

    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:

        job_states = []

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call("scontrol show job {}".format(job_id))
        stdout = stdout.decode("utf8")

        if exit_code == 0:
            job_state_search = re.search("JobState=([a-zA-Z_]*)", str(stdout))
            if job_state_search:
                job_states = [job_state_search.group(1)]
            else:
                logging.error(
                    "Could not extract slurm job state? {}".format(stdout[0:10])
                )
        else:
            stdout, _, exit_code = call("sacct -j {} -o State -P".format(job_id))
            stdout = stdout.decode("utf8")

            if exit_code == 0:
                job_states = stdout.split("\n")[1:]

        if len(job_states) == 0:
            logging.error(
                "Couldn't call scontrol nor sacct to determine job's status. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
            )
            return "ignore"

        def matches_states(slurm_states):
            return len(list(set(job_states) & set(slurm_states))) > 0

        if matches_states(SLURM_STATES["Failure"]):
            return "failed"
        elif matches_states(SLURM_STATES["Ignore"]):
            return "ignore"
        elif matches_states(SLURM_STATES["Unclear"]):
            logging.warn(
                "The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(
                    job_id, job_states
                )
            )
            return "ignore"
        elif matches_states(SLURM_STATES["Success"]):
            return "completed"
        else:
            logging.error(
                "Unhandled slurm job state for job id {}? {}".format(job_id, job_states)
            )
            return "ignore"

    def get_pending_tasks(self):
        try:
            # Get the job ids (%i) of the active user (-u) which are pending (-t) and format
            # them one-per-line (-r) while excluding the header (-h).
            stdout, _ = chcall("squeue -u $(whoami) -t PENDING -r -h --format=%i")
            stdout = stdout.decode("utf8")

            job_ids = set(stdout.split("\n"))
            return job_ids
        except Exception as e:
            logging.error(
                "Couldn't query pending jobs. Polling for finished jobs might be slow."
            )
            return []
