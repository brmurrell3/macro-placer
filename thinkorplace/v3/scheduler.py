"""Run the chains side by side until the deadline and pick the best published placement.

Three chains, one CPU thread each: one warm chain from the supplied placement
and two cold chains with disjoint seed numbers (chain k makes seeds
seed_base + k, seed_base + k + 3, ...). The warm chain won on 10 of the 17
designs measured, the cold chains on the other 7, so neither kind is enough
alone. The scheduler waits until the deadline or until every chain has
finished, reads each chain's best file once (it is written atomically, so it
is complete or absent), ignores anything published after the deadline, and
returns the placement with the lowest proxy.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

from thinkorplace.v3.chain import BEST_FILE, ChainConfig, chain_process, read_best

CHAINS, WARM_CHAINS, SEED_BASE = 3, 1, 5000
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
STOP_GRACE_SECONDS = 2.0


def stop(processes: list) -> None:
    """Terminate every chain still running, then kill any that ignore it."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=STOP_GRACE_SECONDS)
        if process.is_alive():
            process.kill()
            process.join()


def run_chains(directory: str, out: Path, t0: float, budget: float, chains: int = CHAINS,
               warm_chains: int = WARM_CHAINS, seed_base: int = SEED_BASE) -> Tuple[Optional[torch.Tensor], dict]:
    """Return the best placement published before t0 + budget (None if there is none) and a summary."""
    deadline = t0 + budget
    context = multiprocessing.get_context("spawn")
    processes = []
    # a spawned process reads its BLAS thread limits from the environment at import time
    previous = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    try:
        os.environ.update({key: "1" for key in THREAD_VARIABLES})
        for index in range(chains):
            config = ChainConfig(index=index, directory=directory, out=str(out / f"chain-{index}"), t0=t0,
                                 budget=budget, warm=index < warm_chains, first_seed=seed_base + index,
                                 seed_stride=chains)
            process = context.Process(target=chain_process, args=(config,), name=f"v3-chain-{index}")
            process.start()
            processes.append(process)
    except BaseException:
        stop(processes)
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    try:
        while time.time() < deadline and any(process.is_alive() for process in processes):
            time.sleep(0.1)
        decided_at = time.time() - t0
        best = {}
        for index in range(chains):
            row = read_best(out / f"chain-{index}" / BEST_FILE)
            if row is not None and row["published_at"] <= budget:
                best[index] = row
        still_running = [index for index, process in enumerate(processes) if process.is_alive()]
    finally:
        stop(processes)

    endings = []
    for index, process in enumerate(processes):
        path = out / f"chain-{index}" / "chain.json"
        record = json.loads(path.read_text()) if path.exists() else {"status": "no record"}
        record.pop("published", None)
        endings.append({"exit_code": process.exitcode, **record})
    published = [{"chain": index, **{k: v for k, v in row.items() if k != "positions"}} for index, row in best.items()]
    summary = {"decided_at_seconds": decided_at, "chains_running_at_deadline": still_running,
               "published": published, "chains": endings}
    if not published:
        return None, summary
    chosen = min(published, key=lambda row: row["proxy"])
    summary["chosen"] = chosen
    return best[chosen["chain"]]["positions"], summary
