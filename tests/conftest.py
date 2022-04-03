import pytest
import urllib.request
import subprocess
import sys
import time
import sys


def spawn_and_wait_server():
    port = f"123{sys.version_info.minor}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tests.http_server:app",
            "--port",
            str(port),
        ]
    )
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{port}")
        except Exception as e:
            time.sleep(0.1)
        else:
            break
    return process


# credits to pytest-xdist's README
@pytest.fixture(scope="session", autouse=True)
def http_server(tmp_path_factory, worker_id):
    if worker_id == "master":
        # single worker: just run the HTTP server
        process = spawn_and_wait_server()
        yield process
        process.kill()
        process.wait()
        return

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    # try to get a lock
    lock = root_tmp_dir / "lock"
    try:
        lock.mkdir(exist_ok=False)
    except FileExistsError:
        yield  # failed, don't run the HTTP server
        return

    # got the lock, run the HTTP server
    process = spawn_and_wait_server()
    yield process
    process.kill()
    process.wait()
