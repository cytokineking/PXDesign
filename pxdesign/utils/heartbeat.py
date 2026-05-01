# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lightweight heartbeat writer for long-running PXDesign jobs.

Motivation
----------
PXDesign inference often generates many samples inside a single model forward
call (e.g., N_sample=10_000) and can run for a long time with very sparse stdout.
This module provides a best-effort, low-overhead heartbeat file that can be
rsync'ed and inspected remotely.

Design
------
- Each distributed rank writes its own file:  <output_dir>/status_rank{rank}.json
- Rank 0 additionally writes an aggregated file: <output_dir>/status.json
- Writing is throttled by an interval (default 15s) and is best-effort:
  heartbeat failures never crash the job.

Configuration
-------------
Environment variables (optional):
- PXDESIGN_STATUS_DIR: directory to write status.json files into (defaults to none; disabled)
- PXDESIGN_HEARTBEAT_INTERVAL: seconds between writes (default 15)

Metadata env vars (optional, set by runner):
- PXDESIGN_STAGE: current stage label ("diffusion", "evaluation", ...)
- PXDESIGN_TASK_NAME: current task/sample name
- PXDESIGN_SEED: current seed
- PXDESIGN_GLOBAL_RUN: current global pipeline run index
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Cache for HeartbeatReporter.from_env()
_HB_CACHE: dict[tuple[str, int], "HeartbeatReporter"] = {}
logger = logging.getLogger(__name__)
_EVAL_TOOL_ORDER = ("af2_complex", "af2_monomer", "ptx_mini", "ptx")
_EVAL_TOOL_GROUP = {
    "af2_complex": "af2_eval",
    "af2_monomer": "af2_eval",
    "ptx_mini": "protenix_eval",
    "ptx": "protenix_eval",
}
_STAGE_RANK = {
    "startup": 0,
    "diffusion": 1,
    "evaluation": 2,
    "ranking": 3,
    "completed": 4,
}


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    except Exception:
        return None


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Best-effort atomic JSON write (tmp + rename).
    Never raises (heartbeat should not crash a run).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        tmp.replace(path)
    except Exception:
        pass


def _read_json_dict(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _to_nonnegative_int(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        n = int(value)
    except Exception:
        return 0
    return max(n, 0)


def _extract_eval_extra(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_extra = payload.get("extra")
    if not isinstance(raw_extra, dict):
        return None
    raw_eval = raw_extra.get("eval")
    return raw_eval if isinstance(raw_eval, dict) else None


def _has_eval_tool_progress(payload: Dict[str, Any]) -> bool:
    eval_extra = _extract_eval_extra(payload)
    if not isinstance(eval_extra, dict):
        return False
    tools = eval_extra.get("tool_progress")
    return isinstance(tools, dict) and bool(tools)


def _pipeline_value(payload: Dict[str, Any], key: str) -> Any:
    raw_pipeline = payload.get("pipeline")
    if not isinstance(raw_pipeline, dict):
        return None
    return raw_pipeline.get(key)


def _progress_value(payload: Dict[str, Any], key: str) -> Any:
    raw_progress = payload.get("progress")
    if not isinstance(raw_progress, dict):
        return None
    return raw_progress.get(key)


def _stage_rank(stage: Any) -> Optional[int]:
    if stage is None:
        return None
    return _STAGE_RANK.get(str(stage))


def _same_logical_run(existing: Dict[str, Any], new: Dict[str, Any]) -> bool:
    saw_shared_identity = False
    for key in ("global_run", "seed", "task_name"):
        old_value = _pipeline_value(existing, key)
        new_value = _pipeline_value(new, key)
        if old_value in (None, "") or new_value in (None, ""):
            continue
        saw_shared_identity = True
        if str(old_value) != str(new_value):
            return False
    return saw_shared_identity


def _heartbeat_rejection_reason(
    existing: Optional[Dict[str, Any]], new: Dict[str, Any]
) -> Optional[str]:
    new_stage = _pipeline_value(new, "stage")
    new_counter = _progress_value(new, "primary_counter")
    if new_stage == "evaluation" and new_counter == "diffusion_samples":
        return "incoherent evaluation heartbeat uses diffusion_samples counter"

    if not isinstance(existing, dict) or not _same_logical_run(existing, new):
        return None

    old_stage = _pipeline_value(existing, "stage")
    old_rank = _stage_rank(old_stage)
    new_rank = _stage_rank(new_stage)
    if old_rank is not None and new_rank is not None and new_rank < old_rank:
        return f"stage regression from {old_stage} to {new_stage}"

    old_counter = _progress_value(existing, "primary_counter")
    if old_stage == new_stage and old_counter == new_counter and old_counter:
        old_produced = _to_nonnegative_int(_progress_value(existing, "produced_total"))
        new_produced = _to_nonnegative_int(_progress_value(new, "produced_total"))
        if new_produced < old_produced:
            return (
                f"counter regression for {new_stage}/{new_counter}: "
                f"{old_produced} -> {new_produced}"
            )

    if (
        old_stage == "evaluation"
        and new_stage == "evaluation"
        and _has_eval_tool_progress(existing)
        and not _has_eval_tool_progress(new)
    ):
        return "weak evaluation heartbeat would erase eval.tool_progress"

    return None


def _should_preserve_existing_aggregate(
    existing: Optional[Dict[str, Any]], new: Dict[str, Any]
) -> bool:
    if not isinstance(existing, dict):
        return False
    if _pipeline_value(new, "stage") != "evaluation":
        return False
    if _pipeline_value(existing, "stage") != "evaluation":
        return False
    if not _same_logical_run(existing, new):
        return False
    return _has_eval_tool_progress(existing) and not _has_eval_tool_progress(new)


def _aggregate_eval_extra(
    rank_eval: List[Tuple[int, Dict[str, Any]]],
    *,
    produced_sum: int,
    expected_sum: int,
) -> Optional[Dict[str, Any]]:
    if not rank_eval:
        return None

    saw_tool_progress = False
    tool_progress: Dict[str, Dict[str, Any]] = {}
    for tool_name in _EVAL_TOOL_ORDER:
        enabled = False
        done_sum = 0
        total_sum = 0
        saw_tool = False
        for _, eval_extra in rank_eval:
            raw_tools = eval_extra.get("tool_progress")
            if not isinstance(raw_tools, dict):
                continue
            raw_entry = raw_tools.get(tool_name)
            if not isinstance(raw_entry, dict):
                continue
            saw_tool = True
            if bool(raw_entry.get("enabled", True)):
                enabled = True
                done_sum += _to_nonnegative_int(raw_entry.get("done"))
                total_sum += _to_nonnegative_int(raw_entry.get("total"))

        saw_tool_progress = saw_tool_progress or saw_tool
        if enabled:
            tool_progress[tool_name] = {
                "enabled": True,
                "done": done_sum,
                "total": total_sum,
            }
        else:
            tool_progress[tool_name] = {"enabled": False, "done": 0, "total": 0}

    if not saw_tool_progress:
        return None

    active_tool = next(
        (
            tool_name
            for tool_name in _EVAL_TOOL_ORDER
            if bool(tool_progress[tool_name].get("enabled"))
            and _to_nonnegative_int(tool_progress[tool_name].get("done"))
            < _to_nonnegative_int(tool_progress[tool_name].get("total"))
        ),
        None,
    )
    active_group = _EVAL_TOOL_GROUP.get(active_tool) if active_tool else None
    rank0_eval = next(
        (extra for rank, extra in rank_eval if rank == 0),
        rank_eval[0][1],
    )

    merged: Dict[str, Any] = {
        "tool_progress": tool_progress,
        "active_tool": active_tool,
        "active_group": active_group,
        "eval_done": _to_nonnegative_int(produced_sum),
        "eval_total": _to_nonnegative_int(expected_sum),
    }
    for key in ("task", "step"):
        value = rank0_eval.get(key)
        if isinstance(value, str) and value:
            merged[key] = value
    expected_outputs = rank0_eval.get("expected_outputs")
    if isinstance(expected_outputs, dict):
        merged["expected_outputs"] = expected_outputs
    if active_tool:
        active_entry = tool_progress[active_tool]
        merged["active_done"] = _to_nonnegative_int(active_entry.get("done"))
        merged["active_total"] = _to_nonnegative_int(active_entry.get("total"))
    return merged


def _get_dist_info() -> Dict[str, int]:
    """
    Try to detect distributed info from Protenix DIST_WRAPPER, then fall back to env vars.
    """
    try:
        from protenix.utils.distributed import DIST_WRAPPER

        return {
            "rank": int(getattr(DIST_WRAPPER, "rank", 0)),
            "world_size": int(getattr(DIST_WRAPPER, "world_size", 1)),
            "local_rank": int(getattr(DIST_WRAPPER, "local_rank", 0)),
        }
    except Exception:
        return {
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        }


class HeartbeatReporter:
    """
    Heartbeat writer used by PXDesign diffusion and pipeline code.

    This is intentionally dependency-light and safe to call inside long loops.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        interval_seconds: Optional[float] = None,
        throughput_window_seconds: float = 60.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.interval_seconds = (
            float(os.environ.get("PXDESIGN_HEARTBEAT_INTERVAL", "15"))
            if interval_seconds is None
            else float(interval_seconds)
        )
        if self.interval_seconds <= 0:
            self.interval_seconds = 15.0
        self.throughput_window_seconds = float(throughput_window_seconds)

        d = _get_dist_info()
        self.rank = d["rank"]
        self.world_size = d["world_size"]
        self.local_rank = d["local_rank"]

        self._start_ts: float = 0.0
        self._last_write_ts: float = 0.0
        self._expected_total: Optional[int] = None
        self._produced_total: int = 0
        self._recent: List[Tuple[float, int]] = []
        self._state: str = "running"
        self._primary_counter: str = "diffusion_samples"

    # ----------------------------
    # helpers
    # ----------------------------
    def _rank_path(self) -> Path:
        return self.output_dir / f"status_rank{self.rank}.json"

    def _global_path(self) -> Path:
        return self.output_dir / "status.json"

    # ----------------------------
    # public API
    # ----------------------------
    @classmethod
    def from_env(cls) -> Optional["HeartbeatReporter"]:
        out = os.environ.get("PXDESIGN_STATUS_DIR", "")
        if not out:
            return None
        # Cache per (resolved output dir, rank) so callers in different modules
        # can share a single reporter instance (preserves start_ts / throughput).
        d = _get_dist_info()
        key = (str(Path(out).expanduser().resolve()), int(d["rank"]))
        hb = _HB_CACHE.get(key)
        if hb is None:
            hb = cls(out)
            _HB_CACHE[key] = hb
        return hb

    def start(self, *, expected_total: Optional[int] = None) -> None:
        self._start_ts = time.time()
        self._recent = [(self._start_ts, 0)]
        if expected_total is not None:
            self._expected_total = max(int(expected_total), 0)
        self.update(produced_total=0, expected_total=self._expected_total, force=True)

    def update(
        self,
        *,
        produced_total: int,
        expected_total: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        state: Optional[str] = None,
        primary_counter: Optional[str] = None,
        force: bool = False,
    ) -> None:
        now = time.time()
        if not force and (now - self._last_write_ts) < self.interval_seconds:
            return

        previous_state = (
            self._expected_total,
            self._produced_total,
            list(self._recent),
            self._state,
            self._primary_counter,
        )

        self._produced_total = max(int(produced_total), 0)
        if expected_total is not None:
            self._expected_total = max(int(expected_total), 0)
        self._state = state or "running"
        if primary_counter:
            self._primary_counter = str(primary_counter)

        # throughput window
        self._recent.append((now, self._produced_total))
        cutoff = now - self.throughput_window_seconds
        while len(self._recent) > 1 and self._recent[0][0] < cutoff:
            self._recent.pop(0)

        if len(self._recent) >= 2:
            dt = max(self._recent[-1][0] - self._recent[0][0], 1e-6)
            dd = max(self._recent[-1][1] - self._recent[0][1], 0)
            rate_per_sec = dd / dt
        else:
            rate_per_sec = 0.0

        expected_total = max(int(self._expected_total or 0), 0)
        expected_for_percent = max(expected_total, 1)
        produced = max(int(self._produced_total), 0)
        percent = min(max(produced / expected_for_percent, 0.0), 1.0)

        remaining = max(expected_total - produced, 0)
        eta_seconds: Optional[float] = None
        if state == "completed":
            percent = 1.0
            eta_seconds = 0.0
        elif rate_per_sec > 0:
            eta_seconds = remaining / rate_per_sec
        else:
            elapsed = max(now - (self._start_ts or now), 1e-6)
            overall_rate = produced / elapsed
            if overall_rate > 0:
                eta_seconds = remaining / overall_rate

        # context from env (runner sets these)
        stage = os.environ.get("PXDESIGN_STAGE", "")
        task_name = os.environ.get("PXDESIGN_TASK_NAME", "")
        seed = os.environ.get("PXDESIGN_SEED", "")
        global_run = os.environ.get("PXDESIGN_GLOBAL_RUN", "")

        payload: Dict[str, Any] = {
            "job": {
                "output_dir": str(self.output_dir.resolve()),
                "pid": os.getpid(),
                "host": socket.gethostname(),
            },
            "pipeline": {
                "stage": stage or None,
                "task_name": task_name or None,
                "seed": int(seed) if str(seed).isdigit() else (seed or None),
                "global_run": int(global_run)
                if str(global_run).lstrip("-").isdigit()
                else (global_run or None),
            },
            "status": {
                "state": self._state,
                "started_at": _iso(self._start_ts or None),
                "updated_at": _iso(now),
                "percent": percent,
                "eta_seconds": eta_seconds,
                "eta_timestamp": _iso(now + eta_seconds)
                if isinstance(eta_seconds, (int, float))
                else None,
            },
            "compute": {
                "global_rank": self.rank,
                "local_rank": self.local_rank,
                "world_size": self.world_size,
            },
            "progress": {
                "expected_total": expected_total,
                "produced_total": produced,
                "throughput_per_min": rate_per_sec * 60.0,
                "throughput_window_sec": self.throughput_window_seconds,
                "primary_counter": self._primary_counter,
            },
        }

        if extra:
            payload["extra"] = extra

        existing_rank = _read_json_dict(self._rank_path())
        rejection_reason = _heartbeat_rejection_reason(existing_rank, payload)
        if rejection_reason:
            (
                self._expected_total,
                self._produced_total,
                self._recent,
                self._state,
                self._primary_counter,
            ) = previous_state
            logger.warning(
                "Skipping heartbeat write: reason=%s old_stage=%s new_stage=%s "
                "old_counter=%s new_counter=%s old_produced=%s new_produced=%s",
                rejection_reason,
                _pipeline_value(existing_rank or {}, "stage"),
                _pipeline_value(payload, "stage"),
                _progress_value(existing_rank or {}, "primary_counter"),
                _progress_value(payload, "primary_counter"),
                _progress_value(existing_rank or {}, "produced_total"),
                _progress_value(payload, "produced_total"),
            )
            return

        _atomic_write_json(self._rank_path(), payload)

        # Aggregate on rank 0
        if self.rank == 0:
            agg = self._aggregate(now=now)
            existing_global = _read_json_dict(self._global_path())
            if _should_preserve_existing_aggregate(existing_global, agg):
                logger.warning(
                    "Preserving existing aggregate heartbeat with eval.tool_progress; "
                    "new aggregate lacks eval metadata"
                )
            else:
                _atomic_write_json(self._global_path(), agg)

        self._last_write_ts = now

    def complete(self, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self.update(
            produced_total=self._produced_total,
            expected_total=self._expected_total,
            extra=extra,
            state="completed",
            force=True,
        )

    def touch(
        self,
        *,
        extra: Optional[Dict[str, Any]] = None,
        state: Optional[str] = None,
        primary_counter: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Best-effort liveness update without changing progress counters."""
        self.update(
            produced_total=self._produced_total,
            expected_total=self._expected_total,
            extra=extra,
            state=state,
            primary_counter=primary_counter,
            force=force,
        )

    # ----------------------------
    # aggregation (rank 0)
    # ----------------------------
    def _aggregate(self, *, now: float) -> Dict[str, Any]:
        per_rank: List[Dict[str, Any]] = []
        rank_eval: List[Tuple[int, Dict[str, Any]]] = []
        produced_sum = 0
        expected_sum = 0

        for r in range(max(int(self.world_size), 1)):
            p = self.output_dir / f"status_rank{r}.json"
            try:
                d = json.loads(p.read_text())
                prog = d.get("progress", {}) if isinstance(d, dict) else {}
                produced = int(prog.get("produced_total", 0) or 0)
                expected = int(prog.get("expected_total", 0) or 0)
                produced_sum += max(produced, 0)
                expected_sum += max(expected, 0)
                eval_extra = _extract_eval_extra(d)
                if eval_extra is not None:
                    rank_eval.append((r, eval_extra))
                per_rank.append(
                    {
                        "rank": r,
                        "produced_total": produced,
                        "expected_total": expected,
                        "stage": (d.get("pipeline", {}) or {}).get("stage"),
                        "task_name": (d.get("pipeline", {}) or {}).get("task_name"),
                        "updated_at": (d.get("status", {}) or {}).get("updated_at"),
                    }
                )
            except Exception:
                per_rank.append(
                    {
                        "rank": r,
                        "produced_total": 0,
                        "expected_total": 0,
                        "stage": None,
                        "task_name": None,
                        "updated_at": None,
                    }
                )

        expected = max(expected_sum, 1)
        percent = min(max(produced_sum / expected, 0.0), 1.0)

        # Keep rank 0 stage/task as the global "label" (best-effort)
        stage = os.environ.get("PXDESIGN_STAGE", "") or None
        if stage is None:
            stage = next(
                (
                    r.get("stage")
                    for r in per_rank
                    if r.get("rank") == 0 and r.get("stage")
                ),
                None,
            )
            if stage is None:
                stage = next((r.get("stage") for r in per_rank if r.get("stage")), None)
        task_name = os.environ.get("PXDESIGN_TASK_NAME", "") or None
        if task_name is None:
            task_name = next(
                (
                    r.get("task_name")
                    for r in per_rank
                    if r.get("rank") == 0 and r.get("task_name")
                ),
                None,
            )
        seed = os.environ.get("PXDESIGN_SEED", "") or None
        global_run = os.environ.get("PXDESIGN_GLOBAL_RUN", "") or None

        payload: Dict[str, Any] = {
            "job": {
                "output_dir": str(self.output_dir.resolve()),
                "pid": os.getpid(),
                "host": socket.gethostname(),
            },
            "pipeline": {
                "stage": stage,
                "task_name": task_name,
                "seed": int(seed) if seed and seed.isdigit() else seed,
                "global_run": int(global_run)
                if global_run and global_run.lstrip("-").isdigit()
                else global_run,
            },
            "status": {
                "state": self._state,
                "started_at": _iso(self._start_ts or None),
                "updated_at": _iso(now),
                "percent": percent,
            },
            "compute": {
                "global_rank": 0,
                "world_size": self.world_size,
            },
            "progress": {
                "expected_total": expected_sum,
                "produced_total": produced_sum,
                "primary_counter": self._primary_counter,
            },
            "distributed": {
                "per_rank": per_rank,
            },
        }
        if stage == "evaluation":
            eval_extra = _aggregate_eval_extra(
                rank_eval,
                produced_sum=produced_sum,
                expected_sum=expected_sum,
            )
            if eval_extra is not None:
                payload["extra"] = {"eval": eval_extra}
        return payload
