"""Abstracts access to a PBS cluster via its command-line tools.
"""
import re
import os
import threading
import time
from cluster_tools.util import chcall, random_string, local_filename, call
from .cluster_executor import ClusterExecutor
import logging
from typing import Union

PBS_STATES = {
    "Failure": [
        
    ],
    "Success": [
        "Completed"
    ],
    "Ignore": [
        
    ],
    "Unclear": [
        
    ]
}

def submit_text(job):
    """Submits a PBS job represented as a job file string. Returns
    the job ID.
    """

    filename = local_filename("_temp_{}.sh".format(random_string()))
    with open(filename, "w") as f:
        f.write(job)
    jobid_desc, _ = chcall("qsub {}".format(filename))
    match = re.search(r"^[0-9]+", jobid_desc)
    jobid = match.group(0)

    os.unlink(filename)
    return int(jobid)


class PBSExecutor(ClusterExecutor):

    @staticmethod
    def get_job_array_index():
        return os.environ.get("PBS_ARRAY_INDEX", None)

    @staticmethod
    def get_current_job_id():
        return os.environ.get("PBS_JOBID")

    def format_log_file_name(self, jobid):
        return local_filename("pbs.stdout.{}.log").format(str(jobid))

    def inner_submit(
        self,
        cmdline,
        job_name=None,
        additional_setup_lines=[],
        job_count=None,
    ):
        """Starts a PBS job that runs the specified shell command line.
        """

        log_path = self.format_log_file_name("%j" if job_count is None else "%A.%a")

        job_resources_lines = []
        if self.job_resources is not None:
            for resource, value in self.job_resources.items():
                if resource == "time":
                    resource == "walltime"
                job_resources_lines += ["#PBS --{}={}".format(resource, value)]

        job_array_line = ""
        if job_count is not None:
            job_array_line = "#PBS -t 0-{}".format(job_count - 1)

        script_lines = (
            [
                "#!/bin/sh",
                "#PBS -o={}".format(log_path),
                '--N "{}"'.format(job_name),
                job_array_line
            ] + job_resources_lines
            + [*additional_setup_lines, "{}".format(cmdline)]
        )

        return submit_text("\n".join(script_lines))


    def check_for_crashed_job(self, job_id) -> Union["failed", "ignore", "completed"]:

        # If the output file was not found, we determine the job status so that
        # we can recognize jobs which failed hard (in this case, they don't produce output files)
        stdout, _, exit_code = call("checkjob {}".format(job_id))

        if exit_code != 0:
            logging.error(
                "Couldn't call checkjob to determine job's status. {}. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
            )
            return "ignore"
        else:

            job_state_search = re.search('State: ([a-zA-Z_]*)', str(stdout))

            if job_state_search:
                job_state = job_state_search.group(1)

                if job_state in PBS_STATES["Failure"]:
                    return "failed"
                elif job_state in PBS_STATES["Ignore"]:
                    return "ignore"
                elif job_state in PBS_STATES["Unclear"]:
                    logging.warn("The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(job_id, job_state))
                    return "ignore"
                elif job_state in PBS_STATES["Success"]:
                    return "completed"
                else:
                    logging.error("Unhandled pbs job state? {}".format(job_state))
                    return "ignore"
            else:
                logging.error("Could not extract pbs job state? {}".format(stdout[0:10]))
                return "ignore"