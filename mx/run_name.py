
import os

from mx.export import export

@export
def random_run_name() -> str:
    try:
        run_name = os.environ["RUN_NAME"]
    except KeyError:
        import randomname
        run_name = randomname.get_name()
    return run_name

@export
def get_run_name() -> str | None:
    try:
        return os.environ["RUN_NAME"]
    except KeyError:
        return None

@export
def set_run_name(run_name) -> str:
    os.environ["RUN_NAME"] = run_name
    return run_name
