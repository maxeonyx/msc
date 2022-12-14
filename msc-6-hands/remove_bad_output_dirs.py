import shutil
from pathlib import Path

out_dir = Path("_outputs")
for run in out_dir.iterdir():

    if not any(
        any(file.name == "details.json" for file in pipeline.iterdir())
        for pipeline in run.iterdir()
    ):
        print(f"! Removed {str(run): <50} because none of its pipelines ran to completion")
        shutil.rmtree(run)
    else:
        print(f"  Keeping {str(run): <50} because it has at least one pipeline that ran to completion")
