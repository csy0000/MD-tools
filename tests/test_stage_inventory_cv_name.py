"""A stage's inventory names its CV series where the stage writes it: `<key>.cv.csv`.

`_stage_inventory` derived `<trajectory stem>.cv.csv` -- `solute_prod1.cv.csv`, which no stage has
written since a stage has two trajectories -- while the stage writes through `cv_csv_path`. A
completed, resumed stage was then refused as missing its series, and `--overwrite` left the real
`cMD.cv.csv` and its `.cv.json` behind a CV-disabled run.
"""
from __future__ import annotations

from pathlib import Path


def test_the_inventory_and_the_stage_name_one_series():
    from md_tools.md.stage import cv_csv_path
    from md_tools.run.preflight import _stage_inventory

    for stage in ({"name": "cMD"}, {"name": "eq_nvt_posres", "file_key": "eq_1"}):
        trajectory = Path("run") / f"solute_{stage.get('file_key', 'prod1')}.nc"
        roles = _stage_inventory(output=None, log=None, trajectory=trajectory, restart=None,
                                 checkpoint=None, stage=stage).roles
        assert roles["collective_variables"] == cv_csv_path(trajectory, stage)
        key = stage.get("file_key", stage["name"])
        assert roles["collective_variables"].name == f"{key}.cv.csv"
        assert roles["collective_variables_definition"].name == f"{key}.cv.json"
