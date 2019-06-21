import subprocess
import random
import string
import os
import threading
import time
import logging

def local_filename(filename=""):
    return os.path.join(os.getenv("CFUT_DIR", ".cfut"), filename)

def random_string(length=32, chars=(string.ascii_letters + string.digits)):
    return ''.join(random.choice(chars) for i in range(length))

def call(command, stdin=None):
    """Invokes a shell command as a subprocess, optionally with some
    data sent to the standard input. Returns the standard output data,
    the standard error, and the return code.
    """
    if stdin is not None:
        stdin_flag = subprocess.PIPE
    else:
        stdin_flag = None
    proc = subprocess.Popen(command, shell=True, stdin=stdin_flag,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(stdin)
    return stdout, stderr, proc.returncode

class CommandError(Exception):
    """Raised when a shell command exits abnormally."""
    def __init__(self, command, code, stderr):
        self.command = command
        self.code = code
        self.stderr = stderr

    def __str__(self):
        return "%s exited with status %i: %s" % (repr(self.command),
                                                 self.code,
                                                 repr(self.stderr))

def chcall(command, stdin=None):
    """Like ``call`` but raises an exception when the return code is
    nonzero. Only returns the stdout and stderr data.
    """
    stdout, stderr, code = call(command, stdin)
    if code != 0:
        raise CommandError(command, code, stderr)
    return stdout, stderr


def warn_after(job, seconds):
    '''
    Use as decorator to warn when a function is taking longer than {seconds} seconds.
    '''
    def outer(fn):
        def inner(*args, **kwargs):
            exceeded_timeout = [False]
            start_time = time.time()

            def warn_function():
              logging.warn("Function {} is taking suspiciously long (longer than {} seconds)".format(job, seconds))
              exceeded_timeout[0] = True

            timer = threading.Timer(seconds, warn_function)
            timer.start()

            try:
                result = fn(*args, **kwargs)
                if exceeded_timeout[0]:
                    end_time = time.time()
                    logging.warn("Function {} succeeded after all (took {} seconds)".format(job, int(end_time - start_time)))
            finally:
                timer.cancel()
            return result
        return inner
    return outer


class FileWaitThread(threading.Thread):
    """A thread that polls the filesystem waiting for a list of files to
    be created. When a specified file is created, it invokes a callback.
    """

    def __init__(self, callback, interval=1):
        """The callable ``callback`` will be invoked with value
        associated with the filename of each file that is created.
        ``interval`` specifies the polling rate.
        """
        threading.Thread.__init__(self)
        self.callback = callback
        self.interval = interval
        self.waiting = {}
        self.lock = threading.Lock()
        self.shutdown = False

    def stop(self):
        """Stop the thread soon."""
        with self.lock:
            self.shutdown = True

    def waitFor(self, filename, value):
        """Adds a new filename (and its associated callback value) to
        the set of files being waited upon.
        """
        with self.lock:
            self.waiting[filename] = value

    def run(self):

        def handle_completed_job(job_id, filename, failed_early):
            self.callback(job_id, failed_early)
            del self.waiting[filename]

        while True:
            with self.lock:
                if self.shutdown:
                    return

                # Poll for each file.
                for filename in list(self.waiting):
                    job_id = self.waiting[filename]

                    if os.path.exists(filename):
                        # Check for output file as a fast indicator for job completion
                        handle_completed_job(job_id, filename, False)
                    else:
                        # If the output file was not found, we determine the job status so that
                        # we can recognize jobs which failed hard (in this case, they don't produce output files)
                        stdout, _, exit_code = call("scontrol show job {}".format(job_id))

                        # We have to re-check for the output file since this could be created in the mean time
                        if os.path.exists(filename):
                            handle_completed_job(job_id, filename, False)
                        else:
                            if exit_code != 0:
                                logging.error(
                                    "Couldn't call scontrol to determine job's status. {}. Continuing to poll for output file. This could be an indicator for a failed job which was already cleaned up from the slurm db. If this is the case, the process will hang forever."
                                )
                            else:
                                job_state_search = re.search('JobState=([a-zA-Z_]*)', str(stdout))

                                if job_state_search:
                                    job_state = job_state_search.group(1)

                                    if job_state in SLURM_STATES["Failure"]:
                                        handle_completed_job(job_id, filename, True)
                                    elif job_state in SLURM_STATES["Ignore"]:
                                        # This job state can be ignored
                                        pass
                                    elif job_state in SLURM_STATES["Unclear"]:
                                        logging.warn("The job state for {} is {}. It's unclear whether the job will recover. Will wait further".format(job_id, job_state))
                                    elif job_state in SLURM_STATES["Success"]:
                                        logging.error(
                                            "Job state is completed, but {} couldn't be found.".format(
                                                filename
                                            )
                                        )
                                        handle_completed_job(job_id, filename, True)

            time.sleep(self.interval)