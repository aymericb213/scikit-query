import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--clustbench-path",
        action="store",
        # Use environment variable as default if available, otherwise use a default path
        default=os.getenv("CLUSTBENCH_DATA_PATH", "clustering-data-v1"),
        help="Path to clustbench data directory"
    )


@pytest.fixture(scope="session", autouse=True)
def clustbench_path(request):
    """
    Fixture that provides the path to clustbench data.
    Can be set via --clustbench-path command line option or CLUSTBENCH_DATA_PATH env variable.
    """
    path = request.config.getoption("--clustbench-path")
    if not os.path.exists(path):
        pytest.fail(f"Clustbench data path does not exist: {path}")
    return path

