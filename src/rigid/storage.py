"""Atomic artifact publication and bounded-memory file operations."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_file(destination: str | Path) -> Iterator[Path]:
    """Create a same-filesystem temporary file and remove it on every exit path."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(temporary: Path, destination: Path, *, overwrite: bool = False) -> None:
    """Publish a completed file; exclusive publication cannot replace a racing writer.

    Hard links provide atomic create-if-absent on supported local filesystems. A
    filesystem without hard-link support fails explicitly rather than weakening
    the no-overwrite guarantee. Temporary and destination files share a directory.
    """
    if temporary.parent.resolve() != destination.parent.resolve():
        raise ValueError("Staged artifacts must be on the destination filesystem")
    if overwrite:
        os.replace(temporary, destination)
    else:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(f"Output exists: {destination}") from None
        except OSError as error:
            raise OSError(
                f"Atomic exclusive publication failed for {destination}: {error}"
            ) from error


def write_bytes(path: str | Path, content: bytes, *, overwrite: bool = False) -> None:
    target = Path(path)
    with staged_file(target) as temporary:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        publish_file(temporary, target, overwrite=overwrite)


def write_text(path: str | Path, content: str, *, overwrite: bool = False) -> None:
    write_bytes(path, content.encode("utf-8"), overwrite=overwrite)
