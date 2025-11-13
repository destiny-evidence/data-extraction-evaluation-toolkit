"""Tests for the pipeline data models."""

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.data_models.pipeline import (
    CodeExecutor,
    EgressMethod,
    Executor,
    Job,
    JobExecutionError,
    JobFormat,
    JobType,
    Language,
    MissingBinaryError,
    Pipeline,
    PipelineStage,
    ScriptExecutor,
    WrongFiletypeError,
    jobify,
    stage_from_job,
)


@pytest.fixture
def mock_job_func() -> MagicMock:
    """Fixture for a mock callable to be used as a job."""
    return MagicMock(return_value="Job Done")


@pytest.fixture
def code_job(mock_job_func: MagicMock) -> Job:
    """Fixture for a basic 'code' Job."""
    return Job(
        name="test_code_job",
        job_format=JobFormat.CODE,
        job_type=JobType.DATA_PROCESSING,
        language=Language.PYTHON,
        egress_method=EgressMethod.MEMORY,
        job=mock_job_func,
        script_args=None,
        executor=Executor(executor=CodeExecutor()),
    )


@pytest.fixture
def script_job(tmp_path: Path) -> Job:
    """Fixture for a basic 'script' Job."""
    script_path = tmp_path / "test_script.py"
    script_path.touch()
    return Job(
        name="test_script_job",
        job_format=JobFormat.SCRIPT,
        job_type=JobType.DATA_PROCESSING,
        language=Language.PYTHON,
        egress_method=EgressMethod.FILE,
        job=script_path,
        script_args=["--input", "data.csv"],
        executor=Executor(executor=ScriptExecutor(python_path=Path("/usr/bin/python"))),
    )


# jobs
def test_job_run_code_job(code_job: Job, mock_job_func: MagicMock):
    """Test running a job with a Python callable."""
    code_job.func_args = [1, 2]
    code_job.func_kwargs = {"test": True}
    output = code_job.run_job()

    mock_job_func.assert_called_once_with(1, 2, test=True)
    assert output == "Job Done"


def test_job_run_code_job_no_capture(code_job: Job):
    """Test running a job with capture_output=False."""
    code_job.capture_output = False
    output = code_job.run_job()
    assert output is None


@patch("app.data_models.pipeline.Executor.execute")
def test_job_run_script_job(mock_execute: MagicMock, script_job: Job):
    """Test running a job with a script Path."""
    mock_execute.return_value = "Script output"
    output = script_job.run_job()

    mock_execute.assert_called_once_with(job=script_job, args=script_job.script_args)
    assert output == "Script output"


# executors
def test_executors_code_executor(mock_job_func: MagicMock):
    """Test the CodeExecutor executes a callable."""
    executor = CodeExecutor()
    job = MagicMock(spec=Job, job=mock_job_func, capture_output=True)
    result = executor._execute(job, args=[1], kwargs={"a": 2})
    mock_job_func.assert_called_once_with(1, a=2)
    assert result == "Job Done"


def test_executors_code_executor_with_path():
    """Test CodeExecutor raises error if job is a Path."""
    executor = CodeExecutor()
    job = MagicMock(spec=Job, job=Path("script.py"))
    with pytest.raises(JobExecutionError):
        executor._execute(job)


@patch("shutil.which", side_effect=["/usr/bin/python", "/usr/bin/r"])
def test_executors_script_executor_init(mock_which: MagicMock):
    """Test ScriptExecutor initialization."""
    executor = ScriptExecutor()
    assert executor.python_path == Path("/usr/bin/python")
    assert executor.r_path == Path("/usr/bin/r")
    assert mock_which.call_count == 2
    mock_which.assert_has_calls([call("python"), call("R")])


def test_executors_verify_filetype():
    """Test the filetype verification method."""
    executor = ScriptExecutor()
    assert executor.verify_filetype("script.py", ".py")
    with pytest.raises(WrongFiletypeError):
        executor.verify_filetype("script.py", ".R")


@patch("subprocess.run")
def test_executors_python_executor(mock_run: MagicMock, tmp_path: Path):
    """Test the python_executor method."""
    mock_run.return_value = MagicMock(stderr="log message")
    executor = ScriptExecutor(python_path=Path("/bin/python"))
    script = tmp_path / "test.py"
    script.touch()
    output = executor.python_executor(script, args=["--foo"], capture_output=True)
    mock_run.assert_called_once()
    assert output is not None
    assert "log message" in output


@patch("shutil.which", return_value=None)
def test_executors_python_executor_missing_binary(
    mock_which: MagicMock, tmp_path: Path
):
    """Test python_executor raises error if binary is missing."""
    executor = ScriptExecutor(python_path=None)
    script = tmp_path / "test.py"
    script.touch()
    with pytest.raises(MissingBinaryError):
        executor.python_executor(script, args=None)


@patch("subprocess.run")
def test_executors_script_executor_dispatch(mock_run: MagicMock, script_job: Job):
    """Test ScriptExecutor._execute dispatches to the correct method."""
    executor = ScriptExecutor(python_path=Path("/bin/python"))
    executor._execute(script_job, args=script_job.script_args)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "/bin/python"
    assert str(script_job.job) in cmd
    assert "--input" in cmd


# tests for stages
def test_pipeline_stage_convert_jobs_to_list(code_job: Job):
    """Test that a single job is converted to a list."""
    stage = PipelineStage(name="test_stage", jobs=code_job)
    assert isinstance(stage.jobs, list)
    assert stage.jobs[0] == code_job


@patch("app.data_models.pipeline.Job.run_job")
def test_pipeline_stage_run_jobs(mock_run_job: MagicMock, code_job: Job):
    """Test that run_jobs calls run_job on each job."""
    stage = PipelineStage(name="test_stage", jobs=[code_job, code_job])
    stage.run_jobs()
    assert mock_run_job.call_count == 2


@patch("app.data_models.pipeline.Job.run_job")
def test_pipeline_stage_run_jobs_failure_skip(mock_run_job: MagicMock, code_job: Job):
    """Test that stage continues on job failure if skip_jobs_if_failed is True."""
    mock_run_job.side_effect = [Exception("Job failed!"), "Success"]
    stage = PipelineStage(
        name="test_stage", jobs=[code_job, code_job], skip_jobs_if_failed=True
    )
    stage.run_jobs()
    assert mock_run_job.call_count == 2


@patch("app.data_models.pipeline.Job.run_job")
def test_pipeline_stage_run_jobs_failure_no_skip(
    mock_run_job: MagicMock, code_job: Job
):
    """Raises an exception on job failure if skip_jobs_if_failed is False."""
    mock_run_job.side_effect = Exception("Job failed!")
    stage = PipelineStage(
        name="test_stage",
        jobs=[code_job, code_job],
        skip_jobs_if_failed=False,
    )
    with pytest.raises(Exception, match="Job failed!"):
        stage.run_jobs()
    mock_run_job.assert_called_once()


def test_pipeline_stage_write_stage_logfile(tmp_path: Path):
    """Test writing a stage logfile."""
    logfile = tmp_path / "stage.log"
    PipelineStage.write_stage_logfile("log content", logfile)
    assert logfile.read_text() == "log content"


def test_pipeline_stage_arg_precedence(code_job: Job, mock_job_func: MagicMock):
    """Test argument precedence: job > stage > method."""
    code_job.func_args = ["job_arg"]
    stage = PipelineStage(
        name="test",
        jobs=[code_job],
        default_func_args=["stage_arg"],
    )
    stage.run_jobs(func_args=["method_arg"])
    mock_job_func.assert_called_once_with("job_arg")


# full pipeline
@patch("app.data_models.pipeline.PipelineStage.run_jobs")
def test_pipeline_run(mock_run_jobs: MagicMock):
    """Test that Pipeline.run() calls run_jobs on each stage."""
    stage1 = PipelineStage(name="s1", jobs=[])
    stage2 = PipelineStage(name="s2", jobs=[])
    pipeline = Pipeline(name="test_pipeline", stages=[stage1, stage2])
    pipeline.run()
    assert mock_run_jobs.call_count == 2


# helpers / utility functions
def test_helpers_jobify():
    """Test the @jobify decorator."""

    @jobify(name="decorated_job", job_type=JobType.CLASSIFICATION)
    def my_func(a: int, b: int) -> int:
        return a + b

    assert isinstance(my_func, Job)
    assert my_func.name == "decorated_job"
    assert my_func.job_type == JobType.CLASSIFICATION
    assert my_func.job_format == JobFormat.CODE
    assert callable(my_func.job)
    assert my_func.job(1, 2) == 3


def test_helpers_stage_from_job_function(code_job: Job):
    """Test stage_from_job as a direct function."""
    stage = stage_from_job(code_job, stage_name="my_stage")
    assert isinstance(stage, PipelineStage)
    assert stage.name == "my_stage"
    assert stage.jobs == [code_job]


def test_helpers_stage_from_job_decorator():
    """Test @stage_from_job as a decorator."""

    @stage_from_job(stage_name="decorated_stage")
    @jobify(name="inner_job")
    def my_func() -> str:
        return "hello"

    assert isinstance(my_func, PipelineStage)
    assert my_func.name == "decorated_stage"
    # field_validator should ensure we have a list
    # of jobs, even if only one job has been submitted
    assert isinstance(my_func.jobs, list)
    assert len(my_func.jobs) == 1
    assert my_func.jobs[0].name == "inner_job"
