from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
import os
from .remote import setup_logging
from .schedulers.slurm import SlurmExecutor
from .schedulers.pbs import PBSExecutor
from .util import enrich_future_with_uncaught_warning
from . import pickling
import multiprocessing
import logging
import shutil
import tempfile
from pathlib import Path
from functools import partial


def get_existent_kwargs_subset(whitelist, kwargs):
    new_kwargs = {}
    for arg_name in whitelist:
        if arg_name in kwargs:
            new_kwargs[arg_name] = kwargs[arg_name]

    return new_kwargs


PROCESS_POOL_KWARGS_WHITELIST = ["max_workers", "mp_context", "initializer", "initargs"]


class WrappedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, **kwargs):
        new_kwargs = get_existent_kwargs_subset(PROCESS_POOL_KWARGS_WHITELIST, kwargs)

        self.did_overwrite_start_method = False
        self.meta_data = {}
        if kwargs.get("start_method", None) is not None:
            self.did_overwrite_start_method = True
            self.old_start_method = multiprocessing.get_start_method()
            start_method = kwargs["start_method"]
            logging.info(
                f"Overwriting start_method to {start_method}. Previous value: {self.old_start_method}"
            )
            multiprocessing.set_start_method(start_method, force=True)

        assert not (
            "logging_config" in kwargs and "logging_multiprocessing_setup_fn" in kwargs
        ), "Specify either logging_config OR logging_multiprocessing_setup_fn but not both at once"
        if "logging_config" in kwargs:
            self.meta_data["logging_config"] = kwargs["logging_config"]
        if "logging_multiprocessing_setup_fn" in kwargs:
            self.meta_data["logging_multiprocessing_setup_fn"] = kwargs[
                "logging_multiprocessing_setup_fn"
            ]

        ProcessPoolExecutor.__init__(self, **new_kwargs)

    def shutdown(self, *args, **kwargs):

        super().shutdown(*args, **kwargs)

        if self.did_overwrite_start_method:
            logging.info(
                f"Restoring start_method to original value: {self.old_start_method}."
            )
            multiprocessing.set_start_method(self.old_start_method, force=True)
            self.old_start_method = None
            self.did_overwrite_start_method = False

    def submit(self, *args, **kwargs):

        output_pickle_path = None
        if "__cfut_options" in kwargs:
            output_pickle_path = kwargs["__cfut_options"]["output_pickle_path"]
            del kwargs["__cfut_options"]

        if os.environ.get("MULTIPROCESSING_VIA_IO"):
            # If MULTIPROCESSING_VIA_IO is set, _submit_via_io is used to
            # workaround size constraints in pythons multiprocessing
            # implementation. Also see https://github.com/python/cpython/pull/10305/files
            # This should be fixed in python 3.8
            submit_fn = self._submit_via_io
        else:
            submit_fn = super().submit

        call_stack = []

        if multiprocessing.get_start_method() != "fork":
            call_stack.extend(
                [WrappedProcessPoolExecutor._setup_logging_and_execute, self.meta_data]
            )

        if output_pickle_path is not None:
            call_stack.extend(
                [
                    WrappedProcessPoolExecutor._execute_and_persist_function,
                    output_pickle_path,
                ]
            )

        fut = submit_fn(*call_stack, *args, **kwargs)

        enrich_future_with_uncaught_warning(fut)
        return fut

    def _submit_via_io(self, *args, **kwargs):

        func = args[0]
        args = args[1:]

        opt_tmp_dir = os.environ.get("MULTIPROCESSING_VIA_IO_TMP_DIR")
        if opt_tmp_dir is not None:
            dirpath = tempfile.mkdtemp(dir=opt_tmp_dir)
        else:
            dirpath = tempfile.mkdtemp()

        output_pickle_path = Path(dirpath) / "jobdescription.pickle"

        with open(output_pickle_path, "wb") as file:
            pickling.dump((func, args, kwargs), file)

        future = super().submit(
            WrappedProcessPoolExecutor._execute_via_io, output_pickle_path
        )

        future.add_done_callback(
            partial(WrappedProcessPoolExecutor._remove_tmp_file, dirpath)
        )

        return future

    @staticmethod
    def _remove_tmp_file(path, _future):

        shutil.rmtree(path)

    @staticmethod
    def _setup_logging_and_execute(meta_data, *args, **kwargs):

        func = args[0]
        args = args[1:]

        setup_logging(meta_data)

        return func(*args, **kwargs)

    @staticmethod
    def _execute_via_io(serialized_function_info_path):

        with open(serialized_function_info_path, "rb") as file:
            (func, args, kwargs) = pickling.load(file)
        return func(*args, **kwargs)

    @staticmethod
    def _execute_and_persist_function(output_pickle_path, *args, **kwargs):

        func = args[0]
        args = args[1:]

        result = func(*args, **kwargs)

        with open(output_pickle_path, "wb") as file:
            pickling.dump(result, file)

        return result

    def map_unordered(self, func, args):

        futs = self.map_to_futures(func, args)

        # Return a separate generator to avoid that map_unordered
        # is executed lazily (otherwise, jobs would be submitted
        # lazily, as well).
        def result_generator():
            for fut in futures.as_completed(futs):
                yield fut.result()

        return result_generator()

    def map_to_futures(self, func, args, output_pickle_path_getter=None):

        if output_pickle_path_getter is not None:
            futs = [
                self.submit(
                    func,
                    arg,
                    __cfut_options={
                        "output_pickle_path": output_pickle_path_getter(arg)
                    },
                )
                for arg in args
            ]
        else:
            futs = [self.submit(func, arg) for arg in args]

        return futs

    def forward_log(self, fut):
        """
        Similar to the cluster executor, this method Takes a future from which the log file is forwarded to the active
        process. This method blocks as long as the future is not done.
        """

        # Since the default behavior of process pool executors is to show the log in the main process
        # we don't need to do anything except for blocking until the future is done.
        return fut.result()


class SequentialExecutor(WrappedProcessPoolExecutor):
    def __init__(self, **kwargs):
        kwargs["max_workers"] = 1
        WrappedProcessPoolExecutor.__init__(self, **kwargs)


def pickle_identity(obj):
    return pickling.loads(pickling.dumps(obj))


def pickle_identity_executor(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return pickle_identity(result)


class PickleExecutor(WrappedProcessPoolExecutor):
    def submit(self, _func, *_args, **_kwargs):

        (func, args, kwargs) = pickle_identity((_func, _args, _kwargs))
        return super().submit(pickle_identity_executor, func, *args, **kwargs)


def get_executor(environment, **kwargs):
    """
    `kwargs` can be the following optional parameters:
        `logging_config`: An object containing a `level` key specifying the desired log level and/or a
            `format` key specifying the desired log format string. Cannot be specified together
            with `logging_remote_setup_fn`.
        `logging_remote_setup_fn`: A function setting up custom logging. The function will be called for
            remotely executed code (slurm, pbs) to re-setup logging. The function will be called with the
            default log file name. If the caller sets up file logging, this log file name should be adapted,
            for example, by adding a .mylog suffix. Cannot be specified together with `logging_config`.
        `logging_multiprocessing_setup_fn`: A function setting up custom logging. The function will be called for
            locally executed code (multiprocessing, sequential, test_pickling) to re-setup logging if a start_method
            other than the default "fork" is used, because the programming context is not inherited in those cases.
            Cannot be specified together with `logging_config`.
    """
    if environment == "slurm":
        return SlurmExecutor(**kwargs)
    elif environment == "pbs":
        return PBSExecutor(**kwargs)
    elif environment == "multiprocessing":
        return WrappedProcessPoolExecutor(**kwargs)
    elif environment == "sequential":
        return SequentialExecutor(**kwargs)
    elif environment == "test_pickling":
        return PickleExecutor(**kwargs)
    raise Exception("Unknown executor: {}".format(environment))
