"""Small environment checks and measured fingerprint throughput."""

from __future__ import annotations

import importlib.util
import time

import numpy as np

from .provenance import environment_report


def doctor() -> dict:
    from .native import native_available

    report = environment_report()
    report["native_kernel_available"] = native_available()
    report["cuda_available"] = False
    if importlib.util.find_spec("torch") is not None:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_device_count"] = torch.cuda.device_count()
    report["optional_capabilities"] = {
        name: importlib.util.find_spec(module) is not None
        for name, module in (("training", "torch"), ("gctx", "h5py"), ("plots", "matplotlib"))
    }
    return report


def benchmark_fingerprints(*, molecules: int = 1000, bits: int = 2048, seed: int = 17) -> dict:
    from .native import pack_fingerprints, pairwise_tanimoto

    if not 1 <= molecules <= 20000 or not 8 <= bits <= 65536:
        raise ValueError("Use 1..20000 molecules and 8..65536 bits")
    rng = np.random.default_rng(seed)
    matrix = pack_fingerprints(rng.random((molecules, bits)) < 0.05)
    query = matrix[: min(100, molecules)]
    started = time.perf_counter()
    result = pairwise_tanimoto(query, matrix)
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "synthetic": True,
        "queries": len(query),
        "molecules": molecules,
        "bits": bits,
        "seconds": elapsed,
        "comparisons_per_second": result.size / elapsed,
        "result_checksum": float(result.sum()),
        "environment": environment_report(),
    }
