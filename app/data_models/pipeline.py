"""Models to employ for implementing DEET jobs in sequential, harmonised _pipelines_."""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal, TypeVar, overload

from loguru import logger
from pydantic import BaseModel, field_validator

F = TypeVar("F", bound=Callable[..., Any])


# restrict env vars to limit risk of unvalidated scripts.
# for script execution from within pipeline.
restricted_env = {
    "PATH": "/usr/bin:/bin",  # limited PATH - no non-standard executables
    "HOME": str(Path.home()),
    "LANG": os.getenv("LANG", "en_GB.UTF-8"),
}

# below: custom exceptions, enums and helper classes


class WrongFiletypeError(Exception):
    """Raise for wrong filetype."""

    def __init__(
        self,
        msg: str = "Supplied filetype is not correct.",
        *args,  # noqa: ANN002
        **kwargs,
    ) -> None:
        """Init the exception with default message."""
        super().__init__(msg, *args, **kwargs)


class MissingBinaryError(Exception):
    """To raise when we're missing a binary required to run a script."""


class JobExecutionError(Exception):
    """To raise when a job hits a generic error."""


class IngressMethod(StrEnum):
    """An enum of ingress methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()
    HTTP = auto()  # we may need to download data
    RANDOM = auto()  # there might be jobs & pipeline stages where we start with a seed


class EgressMethod(StrEnum):
    """An enum of egree methods for a PipelineStage."""

    FILE = auto()
    MEMORY = auto()


class JobFormat(StrEnum):
    """
    An enum of job formats.

    Jobs are the building blocks of pipeline stages.
    The job format describes the 'medium' in which the job
    is provided to the job object.
    """

    SCRIPT = auto()  # when job is supplied in a file.
    CODE = auto()  # when job is supplied within the pipeline.


class JobType(StrEnum):
    """
    An enum of job types.

    This is a descriptive label of the broad category
    of what the job is doing.
    """

    DATA_PROCESSING = auto()
    DATA_COLLECTION = auto()
    CLASSIFICATION = auto()
    EXTRACTION = auto()


class Language(StrEnum):
    """An enum of permitted languages a job can be specified in."""

    PYTHON = auto()
    R = auto()
    SHELL = auto()
    SQL = auto()
    LLM_PROMPT = auto()


class BaseExecutor(ABC):
    """Abstract base class for all executors."""

    @abstractmethod
    def _execute(
        self,
        job: "Job",
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        pass


class ScriptExecutor(BaseExecutor):
    """An executor class for different kinds of scripts."""

    def __init__(
        self,
        python_path: Path | None = None,
        r_path: Path | None = None,
        bash_path: Path = Path("/bin/bash"),
    ) -> None:
        """Create ScriptExecutor instance."""
        python_which = shutil.which("python")
        r_which = shutil.which("R")

        self.python_path = (
            python_path
            if python_path
            else (Path(python_which) if python_which else None)
        )
        self.r_path = r_path if r_path else (Path(r_which) if r_which else None)
        self.bash_path = bash_path

    def _execute(
        self,
        job: "Job",
        args: list[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        if not isinstance(job.job, Path):
            malspecified_job = (
                "ScriptExecutor requires job.job to be a Path, not a Callable."
            )
            raise JobExecutionError(malspecified_job)

        if job.language == Language.PYTHON:
            return self.python_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        if job.language == Language.R:
            return self.r_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        if job.language == Language.SHELL:
            return self.bash_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        missing_language = (
            f"Script execution not implemented for language: {job.language}"
        )
        raise NotImplementedError(missing_language)

    @staticmethod
    def verify_filetype(filename: str, filetype: Literal[".py", ".R", ".sh"]) -> bool:
        """
        Verify a given file is of a given filetype via checking the ending.

        Args:
            filename (str): Name of file.
            filetype (Literal[&quot;.py&quot;, &quot;.R&quot;, &quot;.sh&quot;]):
                    file ending.

        Raises:
            WrongFiletypeError: When ending doesnt match the input.

        Returns:
            bool: True if OK.

        """
        if not filename[-len(filetype) :] == filetype:
            raise WrongFiletypeError
        return True

    def python_executor(
        self, script_path: Path, args: list[str] | None, *, capture_output: bool = True
    ) -> None | str:
        """
        Execute a python script.

        Args:
            script_path (Path): file path to script.
            args (list[str]): args to run with script.
            capture_output (bool, optional): Defaults to True.

        Returns:
            None | str: output from stdout or None.

        """
        self.verify_filetype(script_path.name, ".py")
        if self.python_path is None:
            python_missing = "can't find python binary. please find it/install."
            raise MissingBinaryError(python_missing)

        cmd: list[str] = [str(self.python_path), str(script_path)]
        if args:
            cmd.extend(args)

        env = restricted_env.copy()
        env.update({"PYTHONPATH": ""})

        output = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=capture_output,
            text=True,
            env=env,
        )
        if capture_output:
            return (
                output.stderr
            )  # loguru writes to stderr; we're using loguru for messages.
        # if you want to capture `print`-ed messages, capture stdout.
        return None

    def r_executor(
        self, script_path: Path, args: list[str] | None, *, capture_output: bool = True
    ) -> None | str:
        """Execute an R script."""
        self.verify_filetype(script_path.name, ".R")
        if self.r_path is None:
            r_missing = "can't find r binary. please find it/install."
            raise MissingBinaryError(r_missing)

        cmd: list[str] = [str(self.r_path), str(script_path)]
        if args:
            cmd.extend(args)

        output = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=capture_output,
            text=True,
            env=restricted_env.copy().update(
                {
                    "R_LIBS_USER": "",  # prevent loading from user library paths
                    "R_PROFILE_USER": "",  # disable user profile scripts
                    "R_ENVIRON_USER": "",  # disable user environment files
                    "R_HISTFILE": "",  # disable history file
                }
            ),
        )
        if capture_output:
            return output.stdout
        return None

    def bash_executor(
        self, script_path: Path, args: list[str] | None, *, capture_output: bool = True
    ) -> None | str:
        """Execute a bash script."""
        self.verify_filetype(script_path.name, ".sh")
        if self.r_path is None:
            r_missing = "can't find bash binary. please find it/install."
            raise MissingBinaryError(r_missing)

        cmd: list[str] = [str(self.bash_path), str(script_path)]
        if args:
            cmd.extend(args)

        output = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=capture_output,
            text=True,
            env=restricted_env.copy().update(
                {
                    "SHELL": str(self.bash_path),
                    "IFS": " \t\n",  # safe input field separator
                    "ENV": "",  # disable shell startup file
                    "BASH_ENV": "",  # disable bash startup file
                }
            ),
        )
        if capture_output:
            return output.stdout
        return None


class CodeExecutor(BaseExecutor):
    """Executor for Python callable."""

    def _execute(
        self,
        job: "Job",
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        args = args or []
        kwargs = kwargs or {}
        if isinstance(job.job, Path):
            malspecified_job = "job has to be a callable, not a path to a script."
            raise JobExecutionError(malspecified_job)
        result = job.job(*args, **kwargs)
        if job.capture_output:
            return result
        return None


class Executor:
    """A wrapper for all kinds of executors."""

    def __init__(self, executor: BaseExecutor) -> None:
        """Init new executor instance."""
        self.executor = executor

    def execute(
        self,
        job: "Job",
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        """Execute a job."""
        return self.executor._execute(job, args=args, kwargs=kwargs)  # noqa: SLF001


# below: core data models: Pipeline>PipelineStage>Job


class Job(BaseModel):
    """The attributes describing a specific job."""

    name: str
    job_format: JobFormat
    job_type: JobType | list[JobType]
    language: Language
    ingress_method: IngressMethod | None  # we may have a job that starts with no data
    egress_method: EgressMethod
    job: Callable | Path
    script_args: list[str] | None
    func_args: list[Any] | None = None
    func_kwargs: dict[str, Any] | None = None
    capture_output: bool = True
    executor: Executor

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    def run_job(self) -> None | str:
        """Run the job defined in this model instance."""
        logger.debug(f"Running job {self.name}")

        # Use func_args/func_kwargs for CODE jobs, script_args for SCRIPT jobs
        if self.job_format == JobFormat.CODE:
            output = self.executor.execute(
                job=self, args=self.func_args, kwargs=self.func_kwargs
            )
        elif self.job_format == JobFormat.SCRIPT:
            output = self.executor.execute(job=self, args=self.script_args)

        if self.capture_output:
            logger.debug(output)
        return output


class PipelineStage(BaseModel):
    """A stage in a DEET pipeline."""

    name: str
    skip_jobs_if_failed: bool = True
    input_file: Path | None  # we're not using this either -- it's handled via jobs.
    data: (
        Any | None
    )  # currently we're not really using this; but it might become relevant?
    jobs: Job | list[Job]
    logfile: Path | None
    default_func_args: list[Any] | None = None
    default_func_kwargs: dict[str, Any] | None = None

    @classmethod
    @field_validator("jobs", mode="before")
    def convert_jobs_to_list(cls, v: Job | list[Job]) -> list[Job]:
        """Convert jobs to list of jobs if just one job supplied."""
        if isinstance(v, Job):
            v = [v]
        return v

    @staticmethod
    def write_stage_logfile(payload: str, filepath: Path) -> None:
        """Write logfile for a specific stage."""
        filepath.write_text(payload)

    def run_jobs(
        self,
        func_args: list[Any] | None = None,
        func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Run all jobs in a pipeline stage."""
        if isinstance(
            self.jobs, Job
        ):  # for mypy -- can we remove this given field_validator?
            self.jobs = [self.jobs]

        args_to_use = func_args or self.default_func_args
        kwargs_to_use = func_kwargs or self.default_func_kwargs

        logger.info(f"Pipeline stage {self.name} has {len(self.jobs)} stages.")
        for i, job in enumerate(self.jobs):
            logger.info(f"Running job number: {i}, name: {job.name}.")
            try:
                # override job's func_args/func_kwargs if provided
                if job.job_format == JobFormat.CODE:
                    job.func_args = args_to_use
                    job.func_kwargs = kwargs_to_use

                job_output = job.run_job()
                if (
                    job.capture_output
                    and job_output is not None
                    and self.logfile is not None
                ):
                    self.write_stage_logfile(payload=job_output, filepath=self.logfile)
                # should we `yield` the job_output here?
            except Exception as e:
                if not self.skip_jobs_if_failed:
                    raise
                logger.error(
                    f"Encountered error {e} on job {i}, {job.name}"
                    f" in pipeline stage {self.name}, moving to next job..."
                )
                continue

    # TO DO:
    # - figure out how we can `yield` stuff in a stage, so that outputs
    # from stage_a can be the inputs for stage_b. does this mean we can
    # return in a Job, or do we need to yield here also? does this make
    # the In/EgressMethod relevant again?


class Pipeline(BaseModel):
    """A complete pipeline consisting of several `PipelineStage` objects."""

    name: str
    stages: list[PipelineStage]

    def run(self) -> None:
        """Run all pipeline stages."""
        logger.info(f"Pipeline {self.name} has {len(self.stages)} stages.")
        for stage in self.stages:
            stage.run_jobs()

    # TO DO:
    # - add dunder methods that allow us to do stuff like Pipeline.extend(),
    # Pipeline.insert(), and so on.


# below: utilities; converters & decorators


def jobify(
    name: str,
    job_type: JobType | list[JobType] = JobType.DATA_PROCESSING,
    func_args: list[Any] | None = None,
    func_kwargs: dict[str, Any] | None = None,
    *,
    capture_output: bool = True,
) -> Callable[[F], Job]:
    """Decorate to wrap a function as a Job instance."""

    def decorator(func: F) -> Job:
        """Wrap around target callable."""
        return Job(
            name=name,
            job_format=JobFormat.CODE,
            job_type=job_type,
            language=Language.PYTHON,
            ingress_method=None,
            egress_method=EgressMethod.MEMORY,
            job=func,
            script_args=None,
            func_args=func_args,
            func_kwargs=func_kwargs,
            capture_output=capture_output,
            executor=Executor(executor=CodeExecutor()),
        )

    return decorator


@overload
def stage_from_job(  # the function version type hints
    job: Job,
    stage_name: str | None = None,
    input_file: Path | None = None,
    logfile: Path | None = None,
    *,
    skip_jobs_if_failed: bool = True,
) -> PipelineStage: ...


@overload
def stage_from_job(  # the decorator version type hints
    job: None = None,
    stage_name: str | None = None,
    input_file: Path | None = None,
    logfile: Path | None = None,
    *,
    skip_jobs_if_failed: bool = True,
) -> Callable[[Job], PipelineStage]: ...


def stage_from_job(
    job: Job | None = None,
    stage_name: str | None = None,
    input_file: Path | None = None,
    logfile: Path | None = None,
    *,
    skip_jobs_if_failed: bool = True,
) -> PipelineStage | Callable[[Job], PipelineStage]:
    """
    Create a PipelineStage from a single Job.

    Can be used as a function or as a decorator.

    Args:
        job: The Job to wrap in a PipelineStage. If None, returns a decorator.
        stage_name: Name for the stage. Defaults to job name if not provided.
        input_file: Optional input file for the stage.
        logfile: Optional logfile for the stage.
        skip_jobs_if_failed: Whether to skip remaining jobs if one fails.

    Returns:
        PipelineStage or decorator function.

    Examples:
        As a function:
        >>> stage = stage_from_job(my_job, stage_name="my_stage")

        As a decorator:
        >>> @stage_from_job(stage_name="my_stage")
        ... @jobify(name="my_job")
        ... def my_function():
        ...     pass

    """

    def _create_stage(j: Job) -> PipelineStage:
        return PipelineStage(
            name=stage_name or j.name,
            skip_jobs_if_failed=skip_jobs_if_failed,
            input_file=input_file,
            data=None,
            jobs=j,
            logfile=logfile,
        )

    # ff job is provided, return the stage directly
    if job is not None:
        return _create_stage(job)

    # otherwise, return a decorator
    return _create_stage
