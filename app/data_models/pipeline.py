"""Models to employ for implementing DEET jobd in sequential, harmonised _pipelines_."""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator

# restrict env vars to limit risk of unvalidated scripts.
# for script execution from within pipeline.
restricted_env = {
    "PATH": "/usr/bin:/bin",  # limited PATH - no non-standard executables
    "HOME": str(Path.home()),
    "LANG": os.getenv("LANG", "en_GB.UTF-8"),
}


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


class DataFormat(BaseModel):
    """
    The format data at a given stage of a job (ingress or egress).

    This is regardless of whether it is passed
    along from a previous stage in memory, or
    read from a file.

    This shouldn't represent a basic built-in
    like `str`, but rather reflect a specific str
    format like markdown.
    """

    data_type: type
    name: str | None
    # json schema, markdown,


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
        python_path: Path | None,
        r_path: Path | None,
        bash_path: Path = Path("/bin/bash"),
    ) -> None:
        """Create ScriptExecutor instance."""
        self.python_path = python_path if python_path else Path(shutil.which("python"))
        self.r_path = r_path if r_path else Path(shutil.which("R"))
        self.bash_path = bash_path

    def _execute(
        self,
        job: "Job",
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        if job.language == Language.PYTHON:
            return self.python_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        elif job.language == Language.R:
            return self.r_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        elif job.language == Language.SHELL:
            return self.bash_executor(
                job.job, args=args, capture_output=job.capture_output
            )
        else:
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
            filetype (Literal[&quot;.py&quot;, &quot;.R&quot;, &quot;.sh&quot;]): file ending.

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

        cmd = [self.python_path, str(script_path)]
        if args:
            cmd.extend(args)

        output = subprocess.run(  # noqa: S603
            cmd,
            check=True,
            capture_output=capture_output,
            text=True,
            env=restricted_env.copy().update({"PYTHONPATH": ""}),
        )
        if capture_output:
            return output.stdout
        return None

    def r_executor(
        self, script_path: Path, args: list[str], *, capture_output: bool = True
    ) -> None | str:
        """Execute an R script."""
        self.verify_filetype(script_path.name, ".R")
        if self.r_path is None:
            r_missing = "can't find r binary. please find it/install."
            raise MissingBinaryError(r_missing)

        cmd = [self.r_path, str(script_path)]
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
        self, script_path: Path, args: list[str], *, capture_output: bool = True
    ) -> None | str:
        """Execute a bash script."""
        self.verify_filetype(script_path.name, ".sh")
        if self.r_path is None:
            r_missing = "can't find bash binary. please find it/install."
            raise MissingBinaryError(r_missing)

        cmd = [self.bash_path, str(script_path)]
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


class CodeExecutor:
    """Executor for Python callable."""

    def _execute(
        self,
        job: "Job",
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        args = args or []
        kwargs = kwargs or {}
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
    ) -> Any:
        """Execute a job."""
        return self.executor._execute(job, args=args, kwargs=kwargs)


def jobify(
    name: str,
    job_type: JobType | list[JobType] = JobType.DATA_PROCESSING,
    ingress_method: IngressMethod | None = None,
    egress_method: EgressMethod = EgressMethod.MEMORY,
    *,
    capture_output: bool = True,
    fallback: bool = True,
):
    """Decorate to wrap a function as a Job instance."""

    def decorator(func: Callable):
        """Wrap around target callable."""
        return Job(
            name=name,
            job_format=JobFormat.CODE,
            job_type=job_type,
            language=Language.PYTHON,
            ingress_method=ingress_method,
            egress_method=egress_method,
            job=func,
            capture_output=capture_output,
            executor=Executor(executor=CodeExecutor()),
            fallback=fallback,
        )

    return decorator


class Job(BaseModel):
    """The attributes describing a specific job."""

    name: str
    job_format: JobFormat
    job_type: JobType | list[JobType]
    language: Language
    ingress_method: IngressMethod | None  # we may have a job that starts with no data
    egress_method: EgressMethod
    job: Callable | Path
    capture_output: bool = True
    executor: Executor
    fallback: True  # TRue ? something like that

    def run_job(self) -> None | str:
        """Run the job defined in this model instance."""
        # try:


class PipelineStage(BaseModel):
    """A stage in a DEET pipeline."""

    name: str
    skip_if_failed: bool = True
    input_file: Path | None
    data: Any | None
    jobs: Job | list[Job]
    logfile: Path | None
    executor: Executor

    @classmethod
    @field_validator("jobs", mode="before")
    def convert_jobs_to_list(cls, v: Job | list[Job]) -> list[Job]:
        """Convert jobs to list of jobs if just one job supplied."""
        if isinstance(v, Job):
            v = [v]
        return v

    @classmethod
    @field_validator
    def read_if_file(cls, v) -> None:
        return

    def run_jobs(self):
        pass


class Pipeline(BaseModel):
    """A complete pipeline consisting of several `PipelineStage` objects."""

    name: str
    stages: list[PipelineStage]
