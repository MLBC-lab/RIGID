"""Inspect a wheel and exercise it outside the editable source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath


def inspect_wheel(path: Path) -> dict:
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate archive entries")
        for name in names:
            parts = PurePosixPath(name).parts
            if name.startswith("/") or ".." in parts or "\\" in name:
                raise ValueError(f"Unsafe archive member: {name}")
            if set(parts) & {
                ".git",
                ".venv",
                "__pycache__",
                "runs",
                "verification",
            } or name.endswith((".pt", ".docx", ".pyc")):
                raise ValueError(f"Development/private artifact in wheel: {name}")
        required = {
            "rigid/__init__.py",
            "rigid/cli.py",
            "rigid/data/schema.py",
            "rigid/models/ranker.py",
            "rigid/engine.py",
            "rigid/library.py",
            "rigid/baselines.py",
            "rigid/consensus.py",
            "rigid/protocols.py",
            "rigid/py.typed",
        }
        if missing := required - set(names):
            raise ValueError(f"Required package files are missing: {sorted(missing)}")
        if not any(n.endswith(".dist-info/METADATA") for n in names):
            raise ValueError("Distribution metadata is missing")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"wheel": path.name, "sha256": digest, "members": len(names)}


def check(wheel: Path, output: Path, *, demo: bool = False) -> dict:
    wheel, output = wheel.resolve(), output.resolve()
    result = inspect_wheel(wheel)
    output.mkdir(parents=True, exist_ok=False)
    installation = output / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(installation),
            str(wheel),
        ],
        check=True,
        timeout=120,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(installation)
    probe = (
        "import json,rigid; print(json.dumps({'path':rigid.__file__,'version':rigid.__version__}))"
    )
    loaded = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", probe], cwd=output, env=environment, text=True, timeout=30
        )
    )
    if not Path(loaded["path"]).resolve().is_relative_to(installation):
        raise RuntimeError("Smoke test imported the source checkout instead of the installed wheel")
    commands = [["doctor"], ["benchmark", "--molecules", "10"]]
    if demo:
        commands.append(["demo", "--output", str(output / "demo")])
    checks = []
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "rigid", *command],
            cwd=output,
            env=environment,
            text=True,
            capture_output=True,
            timeout=300,
        )
        if completed.returncode:
            raise RuntimeError(f"Installed-wheel command {command[0]} failed: {completed.stderr}")
        checks.append({"command": command[0], "result": json.loads(completed.stdout)})
    result.update(
        {"version": loaded["version"], "installed_package_verified": True, "checks": checks}
    )
    (output / "distribution-check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    artifact = parser.add_mutually_exclusive_group(required=True)
    artifact.add_argument("--wheel", type=Path)
    artifact.add_argument(
        "--wheel-dir", type=Path, help="Directory containing exactly one built wheel"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.wheel_dir is not None:
        wheels = list(args.wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            parser.error("--wheel-dir must contain exactly one wheel; otherwise use --wheel")
        args.wheel = wheels[0]
    print(json.dumps(check(args.wheel, args.output, demo=args.demo), indent=2))


if __name__ == "__main__":
    main()
