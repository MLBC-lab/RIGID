import runpy
import zipfile
from pathlib import Path

import pytest


def test_wheel_inspector_rejects_path_traversal_and_private_artifacts(tmp_path):
    inspect = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/check_distribution.py")
    )["inspect_wheel"]
    for index, entry in enumerate(("../escape", "rigid/runs/model.pt", "rigid/private.docx")):
        wheel = tmp_path / f"unsafe{index}.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(entry, "fixture")
        with pytest.raises(ValueError):
            inspect(wheel)


def test_wheel_inspector_requires_complete_installable_package(tmp_path):
    inspect = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools/check_distribution.py")
    )["inspect_wheel"]
    path = tmp_path / "empty.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rigid/__init__.py", "")
    with pytest.raises(ValueError, match="missing"):
        inspect(path)
