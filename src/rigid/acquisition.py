"""Explicit, checksum-pinned HTTPS acquisition without archive extraction."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .provenance import atomic_json
from .storage import publish_file


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urlsplit(newurl)
        if parts.scheme != "https" or parts.username or parts.password:
            raise ValueError("Refusing redirect outside credential-free HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_file(
    url: str,
    destination: str | Path,
    sha256: str,
    *,
    maximum_bytes: int = 2_000_000_000,
    timeout: int = 60,
) -> Path:
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Use an HTTPS URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("Provide the expected SHA-256 from a trusted source")
    if maximum_bytes < 1 or timeout < 1:
        raise ValueError("Download size and timeout must be positive")
    path = Path(destination)
    if path.exists() or path.with_suffix(path.suffix + ".source.json").exists():
        raise FileExistsError(f"Destination already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=path.parent)
    size, digest = 0, hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            request = Request(url, headers={"User-Agent": "RIGID/0.1"})
            with build_opener(_HTTPSRedirectHandler()).open(request, timeout=timeout) as response:
                if urlsplit(response.url).scheme != "https":
                    raise ValueError("Refusing redirect to an unencrypted transport")
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > maximum_bytes:
                        raise ValueError("Download exceeded the configured size limit")
                    digest.update(block)
                    handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != sha256.lower():
            raise ValueError("Downloaded file checksum mismatch; destination was not created")
        publish_file(Path(temporary), path)
        atomic_json(
            path.with_suffix(path.suffix + ".source.json"),
            {
                "source_url": urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")),
                "sha256": digest.hexdigest(),
                "bytes": size,
            },
            overwrite=False,
        )
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
