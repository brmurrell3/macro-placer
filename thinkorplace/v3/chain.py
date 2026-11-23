"""One chain: seeds, descents, then the stacked search, all against one wall-clock deadline.

Several chains run side by side in separate processes, one thread each, and
share nothing but the clock. A cold chain spends its time in three phases:

  seeds     until 10% of the budget: make cold seeds and score their spans.
  descents  until 40%: descend seeds in order of span score (coarse stage,
            fine stage, legalisation). A further descent starts only if one
            of the last one's length still ends inside the phase.
  search    until the deadline: the stacked search from the best legal
            handoff, then from the next best if 12 more sweeps still fit.

A warm chain replaces the seed phase by the four warm seeds and descends
them all. The stacked search from a handoff is four six-sweep segments of
coordinate descent (the first on the widened congestion score) with a soft
cluster interlude between each pair; the last segment keeps sweeping until
the time is better spent on the next handoff.

A chain publishes a placement by validating it and scoring it with the
runtime's incremental evaluator (the proxy in float64 arithmetic; it differs
from the reference scorer by up to about 0.005). It keeps only its best, as
one file written atomically, so the scheduler can read it at the deadline.
Searching stops early enough (1.5 times the slowest publication, plus a
second) that the last placement is published before the deadline.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement
from thinkorplace.common.incremental_evaluator import IncrementalProxyEvaluator
from thinkorplace.common.runtime_safety import clamp_to_canvas
from thinkorplace.v3.descent import INTERLUDES, coarse_descent, fine_descent, interlude
from thinkorplace.v3.legalize import legalize
from thinkorplace.v3.search import CoordinateDescent
from thinkorplace.v3.seeds import Seed, cold_seed, warm_seeds

SEED_PHASE_END, DESCENT_PHASE_END = 0.10, 0.40  # fractions of the budget
SEGMENT_SWEEPS = 6
NEXT_START_SWEEPS = 12  # a further handoff is searched only if this many sweeps still fit
AFTER_INTERLUDE_SWEEPS = 3  # an interlude runs only if this many sweeps still fit after it
WIDENED_MIN_DESCENTS = 9  # the widened score pays off from about sweep 6; six sweeps take about nine descents' time
# estimate of an interlude's length before the first one has run: its 150 steps against a descent's 96 + 404
INTERLUDE_SHARE_OF_DESCENT = 0.35
PUBLISH_SAFETY_FACTOR, PUBLISH_SAFETY_SECONDS = 1.5, 1.0
RETRY_DESCENTS, RETRY_SLACK_SECONDS = 2, 5  # a chain with no legal handoff retries while this much time is left
BEST_FILE = "best.npz"


def load_design(directory: str):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_benchmark_from_dir(directory)


def write_json(path: Path, data: dict) -> None:
    """Write atomically, so a reader never sees half a file."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        json.dump(data, temp, indent=2)
    os.replace(temp.name, path)


def write_best(path: Path, positions: torch.Tensor, proxy: float, stage: str, published_at: float) -> None:
    """Placement and its score in one file, renamed into place, so a reader sees all of it or none."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".npz") as temp:
        np.savez(temp, positions=positions.numpy(), proxy=proxy, stage=stage, published_at=published_at)
    os.replace(temp.name, path)


def read_best(path: Path) -> Optional[dict]:
    """The published placement and its record, or None if there is no complete file."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return {"positions": torch.from_numpy(data["positions"].copy()), "proxy": float(data["proxy"]),
                    "stage": str(data["stage"]), "published_at": float(data["published_at"])}
    except (OSError, ValueError, KeyError):
        return None


@dataclass
class ChainConfig:
    index: int
    directory: str
    out: str
    t0: float
    budget: float
    warm: bool
    first_seed: int
    seed_stride: int


@dataclass
class Handoff:
    label: object
    positions: torch.Tensor
    proxy: float


class Chain:
    def __init__(self, config: ChainConfig):
        self.config = config
        self.out = Path(config.out)
        self.deadline = config.t0 + config.budget
        self.next_seed = config.first_seed
        self.best_proxy = math.inf
        self.publish_seconds = 0.0
        self.sweep_seconds: List[float] = []
        self.descent_seconds: Optional[float] = None
        self.interlude_seconds: Optional[float] = None
        self.descents = 0
        self.record = {"chain": config.index, "warm": config.warm, "status": "setup", "seeds": 0, "descents": 0,
                       "legal_handoffs": 0, "starts": 0, "published": []}
        self.bench, self.plc = load_design(config.directory)
        # coordinate descent gets its own load of the design, as in the measured research code, so its
        # evaluator never shares a PlacementCost object with the gradient stages and publication
        self.cd = CoordinateDescent(*load_design(config.directory))

    # ── clock ──
    def elapsed(self) -> float:
        return time.time() - self.config.t0

    def stop_at(self) -> float:
        """The time searching must stop so that the last placement can still be published."""
        return self.deadline - (PUBLISH_SAFETY_FACTOR * self.publish_seconds + PUBLISH_SAFETY_SECONDS)

    def left(self) -> float:
        return self.stop_at() - time.time()

    def mean_sweep_seconds(self) -> float:
        return sum(self.sweep_seconds) / len(self.sweep_seconds) if self.sweep_seconds else 0.0

    def expected_interlude_seconds(self) -> float:
        if self.interlude_seconds is not None:
            return self.interlude_seconds
        return INTERLUDE_SHARE_OF_DESCENT * (self.descent_seconds or 0.0)

    # ── output ──
    def flush(self) -> None:
        write_json(self.out / "chain.json", self.record)

    def publish(self, positions: torch.Tensor, stage: str) -> Optional[float]:
        """Validate and score a placement; keep it if it is this chain's best. Returns its proxy, or None if invalid."""
        started = time.time()
        positions = positions.to(torch.float32)
        row = {"stage": stage, "at": self.elapsed()}
        proxy = None
        if validate_placement(positions, self.bench)[0]:
            proxy = float(IncrementalProxyEvaluator(self.bench, self.plc, positions).current_cost()["proxy"])
            row["proxy"] = proxy
            if time.time() > self.deadline:
                row["after_deadline"] = True
            elif math.isfinite(proxy) and proxy < self.best_proxy:
                self.best_proxy = proxy
                write_best(self.out / BEST_FILE, positions, proxy, stage, self.elapsed())
                row["kept"] = True
        else:
            row["invalid"] = True
        self.publish_seconds = max(self.publish_seconds, time.time() - started)
        self.record["published"].append(row)
        self.flush()
        return proxy

    # ── seeds ──
    def make_cold_seeds(self, count: Optional[int] = None) -> List[Seed]:
        """Cold seeds until the end of the seed phase (or exactly `count`), lowest span score first."""
        until = self.config.t0 + SEED_PHASE_END * self.config.budget
        seeds, spent = [], []
        while True:
            started = time.time()
            seeds.append(cold_seed(self.next_seed, self.bench, self.plc))
            spent.append(time.time() - started)
            self.next_seed += self.config.seed_stride
            if count is not None and len(seeds) >= count:
                break
            if count is None and time.time() + sum(spent) / len(spent) > until:
                break
        self.record["seeds"] += len(seeds)
        return sorted(seeds, key=lambda seed: seed.span_score)

    # ── descents ──
    def another_descent_fits(self, handoffs: List[Handoff], until: float) -> bool:
        """With a legal handoff in hand, a descent must end inside the phase; without one, before the search stops."""
        if self.descent_seconds is None or not (handoffs or self.descents):
            return True  # the first descent always runs
        end = time.time() + self.descent_seconds
        return end <= until if handoffs else end <= self.stop_at()

    def descend(self, seeds: List[Seed]) -> List[Handoff]:
        """Descend and legalise seeds in order while the descent phase lasts; return legal handoffs, best first."""
        until = self.config.t0 + DESCENT_PHASE_END * self.config.budget
        handoffs = []
        for seed in seeds:
            if not self.another_descent_fits(handoffs, until):
                break
            started = time.time()
            raw = fine_descent(coarse_descent(seed.positions, self.bench, self.plc), self.bench, self.plc)
            legal = legalize(raw, self.bench)
            self.descent_seconds = time.time() - started
            self.descents += 1
            if legal is not None:
                legal = legal.to(torch.float32)
                proxy = self.publish(legal, f"handoff of seed {seed.label}")
                if proxy is not None:
                    handoffs.append(Handoff(seed.label, legal, proxy))
        self.record["descents"] = self.descents
        self.record["legal_handoffs"] += len(handoffs)
        return sorted(handoffs, key=lambda handoff: handoff.proxy)

    # ── stacked search ──
    def search_from(self, handoff: Handoff, another_handoff_waits: bool) -> None:
        self.record["starts"] += 1
        current = handoff.positions
        segments = len(INTERLUDES) + 1
        for index in range(segments):
            last = index == segments - 1
            widened = index == 0 and self.left() >= WIDENED_MIN_DESCENTS * (self.descent_seconds or 0.0)

            def more(done: int, last: bool = last) -> bool:
                if done < SEGMENT_SWEEPS:
                    return True
                if last:  # keep sweeping unless the next handoff is worth opening
                    return not (another_handoff_waits and self.left() >= NEXT_START_SWEEPS * self.mean_sweep_seconds())
                # too little time for an interlude and three sweeps after it: keep sweeping instead
                return self.left() < self.expected_interlude_seconds() + AFTER_INTERLUDE_SWEEPS * self.mean_sweep_seconds()

            result = self.cd.segment(current, widened, more, self.stop_at)
            self.sweep_seconds.extend(result.complete_sweep_seconds)
            current = result.positions
            self.publish(current, f"seed {handoff.label} segment {index + 1} sweep {result.sweeps}")
            if last or time.time() >= self.stop_at():
                break
            started = time.time()
            # the noise seed of interlude k is 2k, the seeds the measured runs used
            moved = interlude(current, self.bench, self.plc, INTERLUDES[index], seed=2 * (index + 1))
            after = clamp_to_canvas(moved, self.bench).to(torch.float32)
            self.interlude_seconds = time.time() - started
            if validate_placement(after, self.bench)[0]:
                current = after

    def run(self) -> None:
        self.record["status"] = "seeds"
        seeds = warm_seeds(self.bench, self.plc) if self.config.warm else self.make_cold_seeds()
        self.record["status"] = "descents"
        handoffs = self.descend(seeds)
        while not handoffs and self.left() > RETRY_DESCENTS * (self.descent_seconds or 0.0) + RETRY_SLACK_SECONDS:
            handoffs = self.descend(self.make_cold_seeds(count=1))  # nothing legal yet: one more seed
        self.record["status"] = "search"
        self.flush()
        for index, handoff in enumerate(handoffs):
            self.search_from(handoff, index + 1 < len(handoffs))
            if time.time() >= self.stop_at():
                break
        self.record.update(status="complete", finished_at=self.elapsed())
        self.flush()


def chain_process(config: ChainConfig) -> None:
    """Entry point of a chain's process: one thread, output to the chain's own log."""
    out = Path(config.out)
    out.mkdir(parents=True)
    log = os.open(out / "log.txt", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1)
    os.dup2(log, 2)
    torch.set_num_threads(1)
    chain = None
    try:
        chain = Chain(config)
        chain.run()
    except Exception as error:  # noqa: BLE001 - recorded for the scheduler, which picks from the other chains
        record = chain.record if chain else {"chain": config.index}
        record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(out / "chain.json", record)
