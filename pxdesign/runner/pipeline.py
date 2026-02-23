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

"""PXDesign pipeline (v2-only).

This is the only supported pipeline implementation going forward.

Key properties
--------------
- Clean v2 output layout under <dump_dir>/runs/run_XXX/...
- Diffusion is resume-safe by filling missing design_id outputs.
- Resume is world-size agnostic: changing GPU count still resumes.
- Eval outputs are written under run_XXX/eval/, never under diffusion.
- Final ranking is purely derived and can be rerun cheaply.

Important breaking changes
--------------------------
- No support for the legacy global_run_* / *_chunk{i} layout.
- No task duplication or N_sample/world_size splitting.

Notes
-----
This pipeline is primarily tuned for the single-task workflow, but it supports
multiple tasks in one input by keeping a per-task active set for early-stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from protenix.config import save_config
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from pxdbench.aggregate import aggregate_binder_eval
from pxdbench.run import run_task
from pxdbench.utils import convert_cifs_to_pdbs, str2bool

from pxdesign.runner.helpers import save_top_designs, use_target_template_or_not
from pxdesign.runner.inference import InferenceRunner
from pxdesign.runner.presets import PRESETS
from pxdesign.utils.heartbeat import HeartbeatReporter
from pxdesign.utils.infer import (
    convert_to_bioassembly_dict,
    derive_seed,
    download_inference_cache,
    get_configs,
)
from pxdesign.utils.inputs import process_input_file
from pxdesign.utils.pipeline import check_tool_weights

logger = logging.getLogger(__name__)
_PROCESS_START_NS = int(time.time_ns())


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


def _clamp_env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        value = int(str(raw).strip())
    except Exception:
        return int(default)
    if value < min_value:
        return int(min_value)
    if value > max_value:
        return int(max_value)
    return int(value)


def _is_enabled(name: str, default: bool = False) -> bool:
    return str2bool(os.environ.get(name, str(default)).strip())


def _canonical_hash(values: list[str] | None) -> str:
    if not values:
        return ""
    payload = ",".join(sorted([str(v) for v in values]))
    return hashlib.sha256(payload.encode()).hexdigest()


def _make_attempt_token(
    run_id: int,
    task_name: str,
    run_seed: int,
    world_size: int,
    attempt_ns: int | None = None,
) -> str:
    token = (
        f"run={int(run_id)}|task={str(task_name)}|seed={int(run_seed)}|"
        f"world={int(world_size)}"
    )
    if attempt_ns is None:
        return token
    return f"{token}|attempt_ns={int(attempt_ns)}"


def _is_nonempty_file(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        return False


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def _read_json_obj(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _wait_for_active_tasks_state(
    *,
    path: str,
    run_id: int,
    run_seed: int,
    timeout_s: int,
    poll_s: int,
) -> list[str]:
    deadline = time.time() + max(int(timeout_s), 1)
    poll_s = max(int(poll_s), 1)
    while True:
        data = _read_json_obj(path)
        if (
            isinstance(data, dict)
            and int(data.get("run_id", -1)) == int(run_id)
            and int(data.get("run_seed", -1)) == int(run_seed)
            and int(data.get("updated_ns", -1)) >= int(_PROCESS_START_NS)
            and isinstance(data.get("active_tasks"), list)
        ):
            return sorted(set(str(x) for x in (data.get("active_tasks") or [])))

        if time.time() >= deadline:
            raise RuntimeError(
                f"Timeout waiting for active_tasks state for run={int(run_id)} seed={int(run_seed)}: path={path}"
            )
        time.sleep(poll_s)


def _start_heartbeat_keepalive(
    hb: Optional[HeartbeatReporter],
    *,
    interval_s: float = 30.0,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[tuple[threading.Event, threading.Thread]]:
    if hb is None or interval_s <= 0:
        return None

    stop = threading.Event()

    def _loop():
        while not stop.wait(interval_s):
            try:
                hb.touch(extra=extra)
            except Exception:
                pass

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return stop, thread


def _update_eval_heartbeat(
    hb: Optional[HeartbeatReporter],
    *,
    task_name: str,
    task_eval_dir: str,
    pdb_names: list[str],
    eval_cfg,
    seed: int,
) -> None:
    if hb is None:
        return
    rank = int(DIST_WRAPPER.rank)
    world_size = max(int(DIST_WRAPPER.world_size), 1)
    owned_names = pdb_names[rank::world_size]
    if not owned_names:
        return
    pending_owned = _pending_pdb_names(owned_names, task_eval_dir, eval_cfg, seed)
    owned_total = int(len(owned_names))
    owned_done = int(owned_total - len(pending_owned))

    num_seqs = int(getattr(eval_cfg, "num_seqs", 1) or 1)
    model_ids = _model_ids_from_cfg(eval_cfg)
    af2_dir = os.path.join(task_eval_dir, "af2_pred")
    ptx_dir = os.path.join(task_eval_dir, "ptx_pred")
    ptx_mini_dir = os.path.join(task_eval_dir, "ptx_mini_pred")

    eval_complex = bool(getattr(eval_cfg, "eval_complex", False))
    eval_monomer = bool(getattr(eval_cfg, "eval_binder_monomer", False))
    eval_ptx_mini = bool(getattr(eval_cfg, "eval_protenix_mini", False))
    eval_ptx = bool(getattr(eval_cfg, "eval_protenix", False))

    def _count_af2_done(monomer: bool) -> int:
        if not (eval_monomer if monomer else eval_complex):
            return owned_total
        done = 0
        for name in owned_names:
            ok = True
            for seq_idx in range(num_seqs):
                if not _has_af2_outputs(af2_dir, name, seq_idx, model_ids, monomer=monomer):
                    ok = False
                    break
            if ok:
                done += 1
        return int(done)

    def _count_ptx_done(ptx_root: str, enabled: bool) -> int:
        if not enabled:
            return owned_total
        done = 0
        for name in owned_names:
            ok = True
            for seq_idx in range(num_seqs):
                if not _has_ptx_outputs(ptx_root, name, seq_idx, seed):
                    ok = False
                    break
            if ok:
                done += 1
        return int(done)

    expected_outputs = {
        "af2_complex": eval_complex,
        "af2_monomer": eval_monomer,
        "ptx_mini": eval_ptx_mini,
        "ptx": eval_ptx,
        "num_seqs": num_seqs,
        "model_ids": model_ids,
    }
    tool_progress = {
        "af2_complex": {
            "enabled": eval_complex,
            "done": _count_af2_done(monomer=False),
            "total": owned_total,
        },
        "af2_monomer": {
            "enabled": eval_monomer,
            "done": _count_af2_done(monomer=True),
            "total": owned_total,
        },
        "ptx_mini": {
            "enabled": eval_ptx_mini,
            "done": _count_ptx_done(ptx_mini_dir, eval_ptx_mini),
            "total": owned_total,
        },
        "ptx": {
            "enabled": eval_ptx,
            "done": _count_ptx_done(ptx_dir, eval_ptx),
            "total": owned_total,
        },
    }
    hb.update(
        produced_total=owned_done,
        expected_total=owned_total,
        primary_counter="eval_designs",
        extra={
            "eval": {
                "task": task_name,
                "owned_total": owned_total,
                "owned_done": owned_done,
                "owned_pending": int(len(pending_owned)),
                "global_total": int(len(pdb_names)),
                "expected_outputs": expected_outputs,
                "tool_progress": tool_progress,
            }
        },
        force=True,
    )


def _run_dir(dump_dir: str, run_id: int) -> str:
    return os.path.join(dump_dir, "runs", f"run_{int(run_id):03d}")


def _results_dir(dump_dir: str, version: int) -> str:
    if int(version) <= 1:
        return os.path.join(dump_dir, "results")
    return os.path.join(dump_dir, f"results_v{int(version)}")


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def allocate_results_dir(dump_dir: str) -> str:
    """
    Choose a versioned results directory.

    - First finalization writes to <dump_dir>/results/
    - Subsequent finalizations write to <dump_dir>/results_v2/, results_v3/, ...
    - If an existing results_vX exists but is not marked completed, reuse it.
    """
    for v in range(1, 10_000):
        p = _results_dir(dump_dir, v)
        if not os.path.exists(p):
            return p
        manifest = _read_json(os.path.join(p, "manifest.json"))
        if manifest is None:
            # unknown/incomplete directory; reuse rather than creating endless versions
            return p
        if str(manifest.get("status", "")).lower() != "completed":
            return p
    raise RuntimeError("Too many results_v* directories; please clean up old results folders.")


def _diffusion_struct_dir(dump_dir: str, run_id: int, task_name: str) -> str:
    return os.path.join(_run_dir(dump_dir, run_id), "diffusion", "structures", task_name)


def _eval_task_dir(dump_dir: str, run_id: int, task_name: str) -> str:
    return os.path.join(_run_dir(dump_dir, run_id), "eval", task_name)


def _final_dir(dump_dir: str, run_id: int) -> str:
    return os.path.join(_run_dir(dump_dir, run_id), "final")


def _existing_indices(struct_dir: str, task_name: str) -> set[int]:
    """Parse <task>_sample_XXXXXX.cif -> {design_id,...}."""
    out: set[int] = set()
    if not os.path.isdir(struct_dir):
        return out
    for fp in Path(struct_dir).glob(f"{task_name}_sample_*.cif"):
        base = fp.name
        parts = base.rsplit("_sample_", 1)
        if len(parts) != 2:
            continue
        idx_str = parts[1].removesuffix(".cif")
        if idx_str.isdigit():
            out.add(int(idx_str))
    return out


def _get_completed_ptx_samples(ptx_pred_dir: str, seed: int) -> set[str]:
    """
    Return set of sample names that have completed Protenix predictions.
    
    Checks for the presence of predictions/<name>_seed_<seed>_sample_0.cif
    under each sample subdirectory in ptx_pred_dir.
    
    Args:
        ptx_pred_dir: Path to the ptx_pred directory (e.g., .../eval/<task>/ptx_pred)
        seed: The run seed used for predictions
        
    Returns:
        Set of sample names (e.g., {"task_sample_000000_seq0", "task_sample_000001_seq0"})
        that have completed Protenix predictions.
    """
    completed: set[str] = set()
    if not os.path.isdir(ptx_pred_dir):
        return completed
    
    for sample_dir in Path(ptx_pred_dir).iterdir():
        if not sample_dir.is_dir() or sample_dir.name.startswith("."):
            continue
        if sample_dir.name == "protenix_inputs.json":
            continue
        sample_name = sample_dir.name
        # Check for seed subdirectory and predictions
        pred_subdir = sample_dir / f"seed_{seed}" / "predictions"
        if pred_subdir.is_dir():
            # Check for at least one .cif file (the main output)
            cifs = list(pred_subdir.glob("*.cif"))
            if cifs:
                completed.add(sample_name)
    return completed


def _count_success_from_csv(csv_path: str) -> int:
    """Best-effort: sum af2_easy_success (fallback to 0 if unknown format)."""
    if not os.path.exists(csv_path):
        return 0
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        for col in [
            "af2_easy_success",
            "AF2-IG-easy-success",
            "pass_af2",
        ]:
            if col in df.columns:
                s = df[col]
                # allow bool / 0/1 / strings
                if s.dtype == bool:
                    return int(s.sum())
                try:
                    return int(s.astype(int).sum())
                except Exception:
                    return int(s.astype(bool).sum())
        return 0
    except Exception:
        return 0


def _default_analysis_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(max(4, cpu_count - 2), 32)


def _resolve_analysis_workers(value: int | None) -> int:
    if value is None:
        return _default_analysis_workers()
    try:
        val = int(value)
    except Exception:
        return _default_analysis_workers()
    if val <= 0:
        return _default_analysis_workers()
    return val


def _model_ids_from_cfg(eval_cfg) -> list[int]:
    model_ids: list[int] = []
    try:
        model_ids = list(eval_cfg.tools.af2.model_ids)
    except Exception:
        try:
            model_ids = list(eval_cfg.tools.af2.get("model_ids", []))
        except Exception:
            model_ids = []
    if not model_ids:
        model_ids = [0]
    return [int(x) for x in model_ids]


def _has_af2_outputs(
    af2_dir: str, name: str, seq_idx: int, model_ids: list[int], monomer: bool = False
) -> bool:
    return _af2_output_health(
        af2_dir=af2_dir,
        name=name,
        seq_idx=seq_idx,
        model_ids=model_ids,
        monomer=monomer,
    )[0]


def _af2_output_health(
    af2_dir: str,
    name: str,
    seq_idx: int,
    model_ids: list[int],
    monomer: bool = False,
) -> tuple[bool, list[str]]:
    suffix = "_MONOMER_ONLY" if monomer else ""
    required_keys = {"pLDDT", "pTM", "i_pTM", "pAE", "i_pAE", "unscaled_i_pAE"} if not monomer else {
        "pLDDT_MONOMER",
        "pTM_MONOMER",
        "pAE_MONOMER",
    }
    reasons: list[str] = []
    for model_id in model_ids:
        model_num = int(model_id) + 1
        fp = os.path.join(
            af2_dir, f"{name}_seq{int(seq_idx)}{suffix}_model{model_num}.json"
        )
        pdb_fp = os.path.join(
            af2_dir,
            f"{name}_seq{int(seq_idx)}{suffix}_model{model_num}.pdb",
        )
        if not _is_nonempty_file(fp):
            reasons.append(
                f"model{model_num}:missing_output_json:{fp}"
            )
            continue
        if not _is_nonempty_file(pdb_fp):
            reasons.append(
                f"model{model_num}:missing_output_pdb:{pdb_fp}"
            )
            continue
        model_data = _read_json_obj(fp)
        if model_data is None:
            reasons.append(f"model{model_num}:invalid_json:{fp}")
            continue
        missing_keys = [k for k in required_keys if k not in model_data]
        if missing_keys:
            reasons.append(
                f"model{model_num}:missing_keys:{','.join(sorted(missing_keys))}:{fp}"
            )
            continue

    if reasons:
        return False, reasons
    return True, reasons


def _af2_output_summary_entry(
    name: str,
    seq_idx: int,
    model_ids: list[int],
    af2_dir: str,
    monomer: bool = False,
) -> tuple[bool, list[str], list[str]]:
    ok, reasons = _af2_output_health(
        af2_dir=af2_dir,
        name=name,
        seq_idx=seq_idx,
        model_ids=model_ids,
        monomer=monomer,
    )
    if ok:
        return True, [], []
    reasons = [f"{name}_seq{int(seq_idx)}:{reason}" for reason in reasons]
    return False, [f"{name}_seq{int(seq_idx)}"], reasons


def _has_ptx_outputs(ptx_dir: str, name: str, seq_idx: int, seed: int) -> bool:
    pred_dir = Path(ptx_dir) / f"{name}_seq{int(seq_idx)}" / f"seed_{int(seed)}" / "predictions"
    if not pred_dir.is_dir():
        return False
    return bool(list(pred_dir.glob("*_summary_confidence_sample_0.json")))


def _pending_pdb_names(
    pdb_names: list[str],
    task_eval_dir: str,
    eval_cfg,
    seed: int,
) -> list[str]:
    af2_dir = os.path.join(task_eval_dir, "af2_pred")
    ptx_dir = os.path.join(task_eval_dir, "ptx_pred")
    ptx_mini_dir = os.path.join(task_eval_dir, "ptx_mini_pred")
    model_ids = _model_ids_from_cfg(eval_cfg)
    num_seqs = int(getattr(eval_cfg, "num_seqs", 1) or 1)

    pending = []
    for name in pdb_names:
        complete = True
        for seq_idx in range(num_seqs):
            if getattr(eval_cfg, "eval_complex", False):
                if not _has_af2_outputs(af2_dir, name, seq_idx, model_ids, monomer=False):
                    complete = False
                    break
            if getattr(eval_cfg, "eval_binder_monomer", False):
                if not _has_af2_outputs(af2_dir, name, seq_idx, model_ids, monomer=True):
                    complete = False
                    break
            if getattr(eval_cfg, "eval_protenix_mini", False):
                if not _has_ptx_outputs(ptx_mini_dir, name, seq_idx, seed):
                    complete = False
                    break
            if getattr(eval_cfg, "eval_protenix", False):
                if not _has_ptx_outputs(ptx_dir, name, seq_idx, seed):
                    complete = False
                    break
        if not complete:
            pending.append(name)
    return pending


def _attempt_dir_name(attempt_token: str) -> str:
    digest = hashlib.sha256(str(attempt_token).encode()).hexdigest()[:20]
    return f"attempt_{digest}"


def _shard_manifest_path(task_eval_dir: str, rank: int, attempt_token: str) -> str:
    return os.path.join(
        task_eval_dir,
        "attempts",
        _attempt_dir_name(attempt_token),
        f"shard_{int(rank)}_inputs.json",
    )


def _chain_authority_path(task_eval_dir: str) -> str:
    return os.path.join(task_eval_dir, "chain_authority.json")


def _normalize_chain_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = str(value).strip()
        return [cleaned] if cleaned else []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if item is None:
                continue
            cleaned = str(item).strip()
            if cleaned:
                out.append(cleaned)
        return sorted(set(out))
    return []


def _extract_chain_hints_from_input(task_input: dict | None) -> tuple[list[str], list[str]]:
    if not isinstance(task_input, dict):
        return [], []

    cond_chains = _normalize_chain_ids(
        ((task_input.get("condition") or {}).get("filter") or {}).get("chain_id")
    )

    binder_chains = _normalize_chain_ids(task_input.get("binder_chains"))
    if not binder_chains:
        binder_chains = _normalize_chain_ids(task_input.get("binder_chain"))
    if not binder_chains and isinstance(task_input.get("generation"), list):
        for item in task_input.get("generation") or []:
            if not isinstance(item, dict):
                continue
            binder_chains = _normalize_chain_ids(item.get("binder_chains"))
            if binder_chains:
                break
            binder_chains = _normalize_chain_ids(item.get("binder_chain"))
            if binder_chains:
                break
            binder_chains = _normalize_chain_ids(item.get("chain_id"))
            if binder_chains:
                break

    return cond_chains, binder_chains


def _resolve_authoritative_chain_payload_rank0(
    *,
    task_eval_dir: str,
    task_name: str,
    struct_dir: str,
    probe_names: list[str],
    task_input: dict | None,
    timeout_s: int,
    poll_s: int,
) -> dict:
    cond_hint, binder_hint = _extract_chain_hints_from_input(task_input)
    if cond_hint and binder_hint:
        return _chain_payload(cond_hint, binder_hint)

    probe_names = [str(x) for x in (probe_names or []) if str(x)]
    if not probe_names:
        raise RuntimeError(
            f"Cannot resolve chain authority for {task_name}: no probe names available."
        )

    probe_root = os.path.join(task_eval_dir, "_cache", "chain_probe", task_name)
    probe_cif_dir = os.path.join(probe_root, "cifs")
    probe_pdb_dir = os.path.join(probe_root, "pdbs")
    if os.path.isdir(probe_root):
        shutil.rmtree(probe_root)
    os.makedirs(probe_cif_dir, exist_ok=True)

    selected_probe_name = None
    deadline = time.time() + max(int(timeout_s), 1)
    base_probe_timeout = max(1, int(timeout_s) // max(len(probe_names), 1))
    for name in probe_names:
        remaining = int(deadline - time.time())
        if remaining <= 0:
            break
        probe_timeout = max(1, min(remaining, max(base_probe_timeout, 5)))
        src_cif = os.path.join(struct_dir, f"{name}.cif")
        dst_cif = os.path.join(probe_cif_dir, f"{name}.cif")
        if _copy_with_retry(src_cif, dst_cif, timeout_s=probe_timeout, poll_s=poll_s):
            selected_probe_name = name
            break
    if selected_probe_name is None:
        raise RuntimeError(
            f"Cannot resolve chain authority for {task_name}: failed to stage any probe CIF "
            f"within timeout_s={int(timeout_s)}"
        )

    try:
        _, _, inferred_cond, inferred_binder = convert_cifs_to_pdbs(
            probe_cif_dir,
            out_pdb_dir=probe_pdb_dir,
            condition_chains=cond_hint or None,
        )
    finally:
        if os.path.isdir(probe_root):
            shutil.rmtree(probe_root)

    inferred_cond = _normalize_chain_ids(inferred_cond)
    inferred_binder = _normalize_chain_ids(inferred_binder)

    if cond_hint and inferred_cond and _canonical_hash(cond_hint) != _canonical_hash(
        inferred_cond
    ):
        raise RuntimeError(
            f"Condition chain mismatch for {task_name}: input={cond_hint} inferred={inferred_cond}"
        )
    if binder_hint and inferred_binder and _canonical_hash(
        binder_hint
    ) != _canonical_hash(inferred_binder):
        raise RuntimeError(
            f"Binder chain mismatch for {task_name}: input={binder_hint} inferred={inferred_binder}"
        )

    cond_final = cond_hint if cond_hint else inferred_cond
    binder_final = binder_hint if binder_hint else inferred_binder
    if not cond_final and not binder_final:
        raise RuntimeError(
            f"Failed to resolve non-empty chain authority for {task_name}."
        )
    return _chain_payload(cond_final, binder_final)


def _write_chain_authority(
    *,
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    world_size: int,
    attempt_token: str,
    pending_names_digest: str,
    chain_payload: dict,
) -> None:
    now_ns = int(time.time_ns())
    payload = {
        "task": str(task_name),
        "run_id": int(run_id),
        "run_seed": int(run_seed),
        "world_size": int(world_size),
        "attempt_token": str(attempt_token),
        "pending_names_digest": str(pending_names_digest),
        "chain_payload": chain_payload,
        "process_start_ns": int(_PROCESS_START_NS),
        "updated_ns": int(now_ns),
        "updated_at": _iso_now(),
        "version": 1,
    }
    _atomic_write_json(_chain_authority_path(task_eval_dir), payload)


def _wait_for_chain_authority(
    *,
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    world_size: int,
    pending_names_digest: str,
    timeout_s: int,
    poll_s: int,
) -> dict:
    deadline = time.time() + max(int(timeout_s), 1)
    poll_s = max(int(poll_s), 1)
    path = _chain_authority_path(task_eval_dir)

    while True:
        data = _read_json_obj(path)
        if (
            isinstance(data, dict)
            and int(data.get("run_id", -1)) == int(run_id)
            and int(data.get("run_seed", -1)) == int(run_seed)
            and int(data.get("world_size", -1)) == int(world_size)
            and str(data.get("pending_names_digest", "")) == str(pending_names_digest)
            and isinstance(data.get("chain_payload"), dict)
            and int(data.get("updated_ns", -1)) >= int(_PROCESS_START_NS)
        ):
            attempt_token = str(data.get("attempt_token", "")).strip()
            if not attempt_token:
                raise RuntimeError(
                    f"Chain authority missing attempt_token for {task_name}: path={path}"
                )
            return {
                "attempt_token": attempt_token,
                "chain_payload": dict(data.get("chain_payload")),
            }

        if time.time() >= deadline:
            raise RuntimeError(
                f"Timeout waiting for chain authority for {task_name}: path={path}"
            )
        time.sleep(poll_s)


def _rank_cache_root(task_eval_dir: str, rank: int, task_name: str) -> str:
    return os.path.join(
        task_eval_dir,
        "_cache",
        f"cif_to_pdb_rank{int(rank)}",
        task_name,
    )


def _copy_with_retry(
    src: str,
    dst: str,
    timeout_s: int,
    poll_s: int,
) -> bool:
    deadline = time.time() + max(int(timeout_s), 1)
    while True:
        try:
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                try:
                    os.unlink(dst)
                except Exception:
                    pass
            try:
                os.link(src, dst)
            except Exception:
                shutil.copy2(src, dst)
            return _is_nonempty_file(dst)
        except Exception:
            if time.time() >= deadline:
                return False
            time.sleep(max(1, int(poll_s)))


def _prepare_rank_cache(
    task_eval_dir: str,
    task_name: str,
    rank: int,
    owned_names: list[str],
    struct_dir: str,
    *,
    condition_chains: list[str] | None,
    timeout_s: int,
    poll_s: int,
) -> tuple[str, list[str], list[str], list[str]]:
    cache_dir = _rank_cache_root(task_eval_dir, rank, task_name)
    cache_tmp_root = cache_dir + ".tmp"
    owned_names = sorted(set(owned_names))

    for p in (cache_tmp_root, cache_dir):
        if os.path.isdir(p):
            shutil.rmtree(p)

    os.makedirs(cache_tmp_root, exist_ok=True)
    staged_cif_dir = os.path.join(cache_tmp_root, "cifs")
    staged_pdb_dir = os.path.join(cache_tmp_root, "pdbs")
    os.makedirs(staged_cif_dir, exist_ok=True)

    promoted = False
    try:
        for name in owned_names:
            src = os.path.join(struct_dir, f"{name}.cif")
            dst = os.path.join(staged_cif_dir, f"{name}.cif")
            ok = _copy_with_retry(src, dst, timeout_s=timeout_s, poll_s=poll_s)
            if not ok:
                raise RuntimeError(
                    f"Failed to stage source CIF {src} for {task_name} rank {rank}"
                )

        if owned_names:
            pdb_dir, converted_names, new_cond_chains, new_binder_chains = (
                convert_cifs_to_pdbs(
                    staged_cif_dir,
                    out_pdb_dir=staged_pdb_dir,
                    condition_chains=condition_chains,
                )
            )
            converted_names = sorted(set(converted_names))
            if converted_names != owned_names:
                raise RuntimeError(
                    f"Converted CIF count mismatch for {task_name} rank {rank}: "
                    f"expected={owned_names} got={converted_names}"
                )
            for name in converted_names:
                if not _is_nonempty_file(os.path.join(staged_pdb_dir, f"{name}.pdb")):
                    raise RuntimeError(
                        f"Converted PDB missing or empty for {task_name} rank {rank}: {name}"
                    )
        else:
            os.makedirs(staged_pdb_dir, exist_ok=True)
            new_cond_chains = condition_chains or []
            converted_names = []
            new_binder_chains = []
            pdb_dir = staged_pdb_dir

        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        os.rename(staged_pdb_dir, cache_dir)
        promoted = True
    finally:
        if promoted:
            if os.path.isdir(cache_tmp_root):
                shutil.rmtree(cache_tmp_root)
        else:
            if os.path.isdir(cache_tmp_root):
                shutil.rmtree(cache_tmp_root)
            if os.path.isdir(cache_dir):
                shutil.rmtree(cache_dir)

    if not promoted:
        raise RuntimeError(
            f"Failed to prepare rank cache for task {task_name} rank {rank}"
        )

    return cache_dir, converted_names, list(new_cond_chains), list(new_binder_chains)


def _chain_payload(cond_chains: list[str], binder_chains: list[str]) -> dict:
    cond = sorted(cond_chains)
    binder = sorted(binder_chains)
    return {
        "cond_chains": cond,
        "binder_chains": binder,
        "chain_digest": _canonical_hash(cond + binder),
        "chain_count": int(len(cond) + len(binder)),
    }


def _validate_chain_payload(
    payload: dict, expected_chain_payload: Optional[dict]
) -> bool:
    if expected_chain_payload is None:
        return True
    return (
        _canonical_hash(payload.get("cond_chains", []))
        == _canonical_hash(expected_chain_payload.get("cond_chains", []))
        and _canonical_hash(payload.get("binder_chains", []))
        == _canonical_hash(expected_chain_payload.get("binder_chains", []))
    )


def _shard_output_summary(
    rank: int,
    task_eval_dir: str,
    owned_names: list[str],
    eval_cfg,
    seed: int,
) -> dict:
    af2_dir = os.path.join(task_eval_dir, "af2_pred")
    ptx_dir = os.path.join(task_eval_dir, "ptx_pred")
    ptx_mini_dir = os.path.join(task_eval_dir, "ptx_mini_pred")
    model_ids = _model_ids_from_cfg(eval_cfg)
    num_seqs = int(getattr(eval_cfg, "num_seqs", 1) or 1)

    af2_complex_enabled = bool(getattr(eval_cfg, "eval_complex", False))
    af2_monomer_enabled = bool(getattr(eval_cfg, "eval_binder_monomer", False))
    ptx_enabled = bool(getattr(eval_cfg, "eval_protenix", False))
    ptx_mini_enabled = bool(getattr(eval_cfg, "eval_protenix_mini", False))

    summary = {
        "rank": int(rank),
        "owned_count": int(len(owned_names)),
        "owned_names": sorted(owned_names),
        "tools": {},
    }

    if af2_complex_enabled:
        missing = []
        observed = 0
        expected_models = int(len(owned_names) * num_seqs * len(model_ids))
        observed_models = 0
        for name in sorted(owned_names):
            for seq_idx in range(num_seqs):
                ok, missing_names, missing_details = _af2_output_summary_entry(
                    name,
                    seq_idx,
                    model_ids,
                    af2_dir=af2_dir,
                    monomer=False,
                )
                if ok:
                    observed += 1
                else:
                    missing.extend(missing_names)
                    missing.extend(missing_details)
                for model_id in model_ids:
                    model_ok, _ = _af2_output_health(
                        af2_dir=af2_dir,
                        name=name,
                        seq_idx=seq_idx,
                        model_ids=[int(model_id)],
                        monomer=False,
                    )
                    if model_ok:
                        observed_models += 1
        expected = int(len(owned_names) * num_seqs)
        summary["tools"]["af2_complex"] = {
            "enabled": True,
            "expected_name_seq": int(expected),
            "observed_name_seq": int(observed),
            "expected_model_outputs": int(expected_models),
            "observed_model_outputs": int(observed_models),
            "remaining_name_seq": sorted(set(missing)),
            "model_ids": [int(m) for m in model_ids],
            "completed_at": _iso_now() if int(observed) >= int(expected) else None,
        }

    if af2_monomer_enabled:
        missing = []
        observed = 0
        expected_models = int(len(owned_names) * num_seqs * len(model_ids))
        observed_models = 0
        for name in sorted(owned_names):
            for seq_idx in range(num_seqs):
                ok, missing_names, missing_details = _af2_output_summary_entry(
                    name,
                    seq_idx,
                    model_ids,
                    af2_dir=af2_dir,
                    monomer=True,
                )
                if ok:
                    observed += 1
                else:
                    missing.extend(missing_names)
                    missing.extend(missing_details)
                for model_id in model_ids:
                    model_ok, _ = _af2_output_health(
                        af2_dir=af2_dir,
                        name=name,
                        seq_idx=seq_idx,
                        model_ids=[int(model_id)],
                        monomer=True,
                    )
                    if model_ok:
                        observed_models += 1
        expected = int(len(owned_names) * num_seqs)
        summary["tools"]["af2_monomer"] = {
            "enabled": True,
            "expected_name_seq": int(expected),
            "observed_name_seq": int(observed),
            "expected_model_outputs": int(expected_models),
            "observed_model_outputs": int(observed_models),
            "remaining_name_seq": sorted(set(missing)),
            "model_ids": [int(m) for m in model_ids],
            "completed_at": _iso_now() if int(observed) >= int(expected) else None,
        }

    if ptx_mini_enabled:
        missing = []
        observed = 0
        for name in sorted(owned_names):
            for seq_idx in range(num_seqs):
                if _has_ptx_outputs(ptx_mini_dir, name, seq_idx, seed):
                    observed += 1
                else:
                    missing.append(f"{name}_seq{int(seq_idx)}")
        expected = int(len(owned_names) * num_seqs)
        summary["tools"]["ptx_mini"] = {
            "enabled": True,
            "expected_name_seq": int(expected),
            "observed_name_seq": int(observed),
            "remaining_name_seq": sorted(set(missing)),
        }

    if ptx_enabled:
        missing = []
        observed = 0
        for name in sorted(owned_names):
            for seq_idx in range(num_seqs):
                if _has_ptx_outputs(ptx_dir, name, seq_idx, seed):
                    observed += 1
                else:
                    missing.append(f"{name}_seq{int(seq_idx)}")
        expected = int(len(owned_names) * num_seqs)
        summary["tools"]["ptx"] = {
            "enabled": True,
            "expected_name_seq": int(expected),
            "observed_name_seq": int(observed),
            "remaining_name_seq": sorted(set(missing)),
        }

    for key, value in summary["tools"].items():
        if not value["enabled"]:
            value["complete"] = True
        else:
            value["complete"] = int(value["observed_name_seq"]) >= int(
                value["expected_name_seq"]
            )

    summary["tools_complete"] = all(
        item.get("complete", False) for item in summary["tools"].values()
    ) if summary["tools"] else True
    summary["completed"] = bool(summary["tools_complete"])

    return summary


def _write_shard_manifest(
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    rank: int,
    world_size: int,
    owned_names: list[str],
    attempt_token: str,
    pending_names_digest: str,
    cond_chains: list[str],
    binder_chains: list[str],
    chain_payload: dict,
    output_summary: dict,
) -> None:
    owned_names = sorted(owned_names)
    manifest = {
        "task": str(task_name),
        "run_id": int(run_id),
        "run_seed": int(run_seed),
        "attempt_token": str(attempt_token),
        "task_eval_dir": str(task_eval_dir),
        "rank": int(rank),
        "world_size": int(world_size),
        "owned_count": int(len(owned_names)),
        "owned_names": list(owned_names),
        "no_work": bool(len(owned_names) == 0),
        "owned_slice": {
            "rank": int(rank),
            "step": int(world_size),
        },
        "pending_names_digest": str(pending_names_digest),
        "cond_chains": sorted(cond_chains),
        "binder_chains": sorted(binder_chains),
        "chain_payload": chain_payload,
        "outputs": output_summary,
        "updated_at": _iso_now(),
        "version": 1,
    }
    _atomic_write_json(_shard_manifest_path(task_eval_dir, rank, attempt_token), manifest)


def _build_aggregate_inputs(
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    world_size: int,
    all_pdb_names: list[str],
    all_output_manifests: list[dict],
    chain_payload: dict,
    struct_dir: str,
) -> str:
    agg_dir = os.path.join(task_eval_dir, "_cache", "cif_to_pdb", "aggregate")
    agg_tmp = agg_dir + ".tmp"
    if os.path.isdir(agg_tmp):
        shutil.rmtree(agg_tmp)
    if os.path.isdir(agg_dir):
        shutil.rmtree(agg_dir)
    os.makedirs(agg_tmp, exist_ok=True)

    manifest_by_rank: dict[int, dict] = {}
    owner_map: dict[str, int] = {}
    for manifest in all_output_manifests:
        if not isinstance(manifest, dict):
            continue
        rank = int(manifest.get("rank", -1))
        manifest_by_rank[rank] = manifest
        for owned_name in manifest.get("owned_names", []) or []:
            if owned_name in owner_map:
                raise RuntimeError(
                    f"Duplicate ownership detected for {owned_name} in aggregate map for {task_name}"
                )
            owner_map[owned_name] = rank

    rank_cache_roots: list[Path] = []
    cache_root = Path(task_eval_dir) / "_cache"
    if cache_root.is_dir():
        rank_cache_roots = [p for p in cache_root.glob("cif_to_pdb_rank*") if p.is_dir()]

    source_by_name: dict[str, str] = {}
    unresolved_names: list[str] = []
    unresolved_by_rank: dict[int, list[str]] = {}
    for name in sorted(all_pdb_names):
        assigned = owner_map.get(name)
        candidates: list[str] = []
        if assigned is not None:
            if assigned not in manifest_by_rank:
                raise RuntimeError(
                    f"Unknown owner rank {assigned} for {name} when building aggregate for {task_name}"
                )
            candidates.append(
                os.path.join(
                    _rank_cache_root(task_eval_dir, assigned, task_name),
                    f"{name}.pdb",
                )
            )
        candidates.append(
            os.path.join(task_eval_dir, "_cache", "cif_to_pdb", task_name, f"{name}.pdb")
        )
        for root in rank_cache_roots:
            candidates.append(str(root / task_name / f"{name}.pdb"))
        selected = next((c for c in candidates if _is_nonempty_file(c)), None)
        if selected is None:
            unresolved_names.append(name)
            rank_key = int(assigned) if assigned is not None else -1
            unresolved_by_rank.setdefault(rank_key, []).append(name)
        else:
            source_by_name[name] = selected

    if unresolved_names:
        rehydrate_timeout = _clamp_env_int(
            "PXDESIGN_STAGEIN_SOURCE_TIMEOUT_S", 900, 30, 7200
        )
        rehydrate_poll = _clamp_env_int(
            "PXDESIGN_STAGEIN_SOURCE_POLL_S", 10, 2, 60
        )
        rehydrate_cif_dir = os.path.join(agg_tmp, "_rehydrate_cifs")
        rehydrate_pdb_dir = os.path.join(agg_tmp, "_rehydrate_pdbs")
        os.makedirs(rehydrate_cif_dir, exist_ok=True)
        for name in unresolved_names:
            src_cif = os.path.join(struct_dir, f"{name}.cif")
            dst_cif = os.path.join(rehydrate_cif_dir, f"{name}.cif")
            if not _copy_with_retry(
                src_cif,
                dst_cif,
                timeout_s=rehydrate_timeout,
                poll_s=rehydrate_poll,
            ):
                missing_detail = "; ".join(
                    [
                        f"rank={rk}:names={','.join(sorted(vals))}"
                        for rk, vals in sorted(unresolved_by_rank.items(), key=lambda x: x[0])
                    ]
                )
                raise RuntimeError(
                    f"Missing required aggregate source CIF for {task_name}: {src_cif}; "
                    f"missing_by_rank={missing_detail}"
                )
        _, converted_names, _, _ = convert_cifs_to_pdbs(
            rehydrate_cif_dir,
            out_pdb_dir=rehydrate_pdb_dir,
            condition_chains=_normalize_chain_ids((chain_payload or {}).get("cond_chains")) or None,
        )
        converted_set = set(converted_names or [])
        for name in unresolved_names:
            if name not in converted_set:
                raise RuntimeError(
                    f"Failed to rehydrate missing aggregate PDB for {task_name}: {name}"
                )
            src = os.path.join(rehydrate_pdb_dir, f"{name}.pdb")
            if not _is_nonempty_file(src):
                raise RuntimeError(
                    f"Rehydrated aggregate PDB missing or empty for {task_name}: {src}"
                )
            source_by_name[name] = src

    for name in sorted(all_pdb_names):
        src = source_by_name.get(name)
        if not src or not _is_nonempty_file(src):
            raise RuntimeError(
                f"Missing required aggregate source {name} while building {task_name}"
            )
        dst = os.path.join(agg_tmp, f"{name}.pdb")
        try:
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        if not _is_nonempty_file(dst):
            raise RuntimeError(f"Aggregate link/copy produced empty output for {name}")

    for tmp_name in ("_rehydrate_cifs", "_rehydrate_pdbs"):
        tmp_path = os.path.join(agg_tmp, tmp_name)
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)

    os.rename(agg_tmp, agg_dir)
    aggregate_inputs = {
        "run_id": int(run_id),
        "run_seed": int(run_seed),
        "world_size": int(world_size),
        "task": str(task_name),
        "pdb_names": list(sorted(all_pdb_names)),
        "pdb_names_digest": _canonical_hash(sorted(all_pdb_names)),
        "chain_payload": chain_payload,
        "shards": {
            manifest.get("rank"): {
                "owned_count": int(manifest.get("owned_count", 0)),
                "owned_names": manifest.get("owned_names", []),
            }
            for manifest in all_output_manifests
        },
        "aggregate_pdb_dir": agg_dir,
        "updated_at": _iso_now(),
    }
    _atomic_write_json(os.path.join(task_eval_dir, "aggregate_inputs.json"), aggregate_inputs)
    return agg_dir


def _wait_for_shards_ready(
    task_eval_dir: str,
    task_name: str,
    attempt_token: str,
    pending_names_digest: str,
    expected_owned_map: dict[int, list[str]],
    eval_cfg,
    seed: int,
    timeout_s: int,
    poll_s: int,
    expected_chain_payload: dict,
) -> list[dict]:
    deadline = time.time() + max(int(timeout_s), 1)
    poll_s = max(int(poll_s), 1)
    expected_world_size = int(len(expected_owned_map))

    while True:
        manifests: list[dict] = []
        missing: list[int] = []
        mismatches: list[str] = []
        incomplete: list[str] = []
        incomplete_details: list[str] = []
        all_complete = True

        for rank in range(expected_world_size):
            path = _shard_manifest_path(task_eval_dir, rank, attempt_token)
            manifest = _read_json_obj(path)
            if manifest is None:
                missing.append(rank)
                all_complete = False
                continue

            if int(manifest.get("rank", -1)) != rank:
                mismatches.append(f"rank={rank}:manifest_rank={manifest.get('rank')}")
                all_complete = False
                continue

            if str(manifest.get("attempt_token", "")) != str(attempt_token):
                mismatches.append(f"rank={rank}:attempt_token")
                all_complete = False
                continue

            if str(manifest.get("pending_names_digest", "")) != str(pending_names_digest):
                mismatches.append(f"rank={rank}:pending_names_digest")
                all_complete = False
                continue

            if int(manifest.get("world_size", 0)) != expected_world_size:
                mismatches.append(f"rank={rank}:world_size")
                all_complete = False
                continue

            owned_slice = (
                manifest.get("owned_slice")
                if isinstance(manifest.get("owned_slice"), dict)
                else {}
            )
            if int(owned_slice.get("rank", -1)) != rank or int(
                owned_slice.get("step", 0)
            ) != expected_world_size:
                mismatches.append(f"rank={rank}:owned_slice")
                all_complete = False
                continue

            expected_owned = sorted(set(expected_owned_map.get(rank, [])))
            manifest_owned = sorted(set(manifest.get("owned_names", []) or []))
            if expected_owned != manifest_owned:
                mismatches.append(f"rank={rank}:owned_names")
                all_complete = False
                continue

            chain_payload = (
                manifest.get("chain_payload")
                if isinstance(manifest.get("chain_payload"), dict)
                else {}
            )
            if expected_owned and not _validate_chain_payload(
                chain_payload, expected_chain_payload
            ):
                mismatches.append(f"rank={rank}:chain")
                all_complete = False
                continue

            recompute = _shard_output_summary(
                rank,
                task_eval_dir,
                expected_owned,
                eval_cfg,
                seed,
            )
            if not bool(recompute.get("completed", False)):
                all_complete = False
                missing_items = []
                for tool_name, tool_data in (recompute.get("tools") or {}).items():
                    if not tool_data.get("complete", False):
                        remaining = sorted(tool_data.get("remaining_name_seq", []))
                        if remaining:
                            missing_items.append(f"{tool_name}:{','.join(remaining)}")
                if missing_items:
                    incomplete.extend([f"rank={rank}:{x}" for x in missing_items])
                    incomplete_details.append(
                        f"rank={rank}:pending={','.join(missing_items)}"
                    )

            manifest["recomputed_outputs"] = recompute
            manifests.append(manifest)

        if mismatches:
            raise RuntimeError(
                "Shard manifest mismatch for "
                f"{task_name}: " + "; ".join(mismatches)
            )

        if (
            all_complete
            and len(manifests) == expected_world_size
            and len(missing) == 0
        ):
            manifests_by_rank = {int(m.get("rank", -1)): m for m in manifests}
            return [manifests_by_rank.get(r, {}) for r in range(expected_world_size)]

        if time.time() >= deadline:
            raise RuntimeError(
                f"Timeout waiting for shard readiness: task={task_name}, "
                f"missing_manifests={missing}, incomplete={incomplete}, "
                f"details={incomplete_details}"
            )
        time.sleep(poll_s)


# -----------------------------------------------------------------------------
# Pipeline state (v2)
# -----------------------------------------------------------------------------


def _load_pipeline_state(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _init_or_update_pipeline_state(
    *,
    dump_dir: str,
    input_sha256: str,
    n_max_runs: int,
    seeds: list[int],
    target_n_sample: int,
) -> dict:
    """Create or extend pipeline_state.json (rank 0 only)."""
    state_path = os.path.join(dump_dir, "pipeline_state.json")
    state = _load_pipeline_state(state_path)

    def _scan_runs() -> dict[int, dict]:
        """
        Best-effort rebuild of run metadata from on-disk v2 folders.
        This is intentionally forgiving: missing/corrupt state files should not
        prevent resume-by-disk.
        """
        out: dict[int, dict] = {}
        runs_root = Path(dump_dir) / "runs"
        if not runs_root.exists():
            return out
        for rp in runs_root.glob("run_*"):
            suffix = rp.name.replace("run_", "", 1)
            if not suffix.isdigit():
                continue
            run_id = int(suffix)
            run_seed = None

            # Prefer diffusion_state.json (written by pipeline v2)
            diff_state = rp / "diffusion" / "diffusion_state.json"
            if diff_state.exists():
                try:
                    d = json.loads(diff_state.read_text())
                    if isinstance(d, dict) and str(d.get("run_seed", "")).lstrip("-").isdigit():
                        run_seed = int(d["run_seed"])
                except Exception:
                    pass

            # Fallback: orig_seqs.json contains run_seed too
            if run_seed is None:
                orig_seqs_p = rp / "diffusion" / "orig_seqs.json"
                if orig_seqs_p.exists():
                    try:
                        d = json.loads(orig_seqs_p.read_text())
                        if isinstance(d, dict) and str(d.get("run_seed", "")).lstrip("-").isdigit():
                            run_seed = int(d["run_seed"])
                    except Exception:
                        pass

            if run_seed is None:
                run_seed = int((time.time_ns() + run_id) % (2**31 - 1))

            out[run_id] = {
                "run_id": int(run_id),
                "run_seed": int(run_seed),
                "target_N_sample": int(target_n_sample),
            }
        return out

    if state is None:
        # Rebuild from disk if possible, otherwise start fresh.
        scanned = _scan_runs()
        state = {
            "version": 2,
            "layout": "v2",
            "job": {"created_at": _iso_now(), "input_sha256": input_sha256},
            "runs": [scanned[k] for k in sorted(scanned.keys())],
        }
    else:
        if state.get("layout") != "v2" or int(state.get("version", 0) or 0) != 2:
            raise RuntimeError(
                f"Refusing to run v2 pipeline in dump_dir with non-v2 pipeline_state: {state_path}"
            )
        prev_sha = (state.get("job") or {}).get("input_sha256")
        if prev_sha and prev_sha != input_sha256:
            raise RuntimeError(
                "dump_dir already contains a pipeline_state.json for a different input. "
                "Use a new dump_dir or delete pipeline_state.json."
            )

    # Normalize runs into a dense [0..max] list so that index == run_id.
    runs_in: list[dict] = list(state.get("runs") or [])
    runs_by_id: dict[int, dict] = {}
    for r in runs_in:
        try:
            rid = int(r.get("run_id"))
        except Exception:
            continue
        if rid < 0:
            continue
        if "run_seed" not in r:
            continue
        runs_by_id[rid] = {
            "run_id": int(rid),
            "run_seed": int(r.get("run_seed")),
            "target_N_sample": int(r.get("target_N_sample", target_n_sample) or target_n_sample),
        }

    max_existing_id = max(runs_by_id.keys()) if runs_by_id else -1
    max_needed_id = max(int(n_max_runs) - 1, max_existing_id)
    runs: list[dict] = []
    for rid in range(max_needed_id + 1):
        if rid in runs_by_id:
            runs.append(runs_by_id[rid])
        else:
            # Fill gaps deterministically to keep run_id indexing stable.
            if seeds and rid < len(seeds):
                run_seed = int(seeds[rid])
            else:
                run_seed = int((time.time_ns() + rid) % (2**31 - 1))
            runs.append(
                {"run_id": int(rid), "run_seed": int(run_seed), "target_N_sample": int(target_n_sample)}
            )

    # If user provided seeds, enforce consistency.
    if seeds:
        if len(seeds) != int(n_max_runs):
            raise AssertionError("The number of seeds must equal N_max_runs")
        if runs:
            existing = [int(r.get("run_seed")) for r in runs[: len(seeds)]]
            if existing != list(seeds)[: len(existing)]:
                raise RuntimeError(
                    "Provided --seeds do not match existing pipeline_state.json. "
                    "Use a new dump_dir or delete pipeline_state.json."
                )

    # Truncate/extend to requested N_max_runs (state is append-only on disk).
    # We keep extra runs in the file; pipeline runtime may ignore them when N_max_runs decreases.
    if len(runs) < int(n_max_runs):
        # should not happen because we constructed max_needed_id above, but keep safe.
        while len(runs) < int(n_max_runs):
            rid = len(runs)
            run_seed = int(seeds[rid]) if seeds and rid < len(seeds) else int((time.time_ns() + rid) % (2**31 - 1))
            runs.append(
                {"run_id": int(rid), "run_seed": int(run_seed), "target_N_sample": int(target_n_sample)}
            )

    state["runs"] = runs
    _atomic_write_json(state_path, state)
    return state


# -----------------------------------------------------------------------------
# CLI parsing (kept compatible with existing CLI wrapper)
# -----------------------------------------------------------------------------


def _get_overridden_keys(argv) -> set:
    """Infer which long-form CLI options were explicitly set by user."""
    if argv is None:
        return set()

    overridden = set()
    it = iter(argv)
    for token in it:
        if not token.startswith("-"):
            continue
        if token.startswith("--"):
            name = token[2:]
            if "=" in name:
                name = name.split("=", 1)[0]
            overridden.add(name.replace("-", "_"))
    return overridden


def parse_pipeline_args(argv=None):
    """Parse pipeline-level CLI arguments. Remaining args go to get_configs."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preset",
        type=str,
        choices=["preview", "extended", "custom"],
        default="preview",
        help=(
            "High-level pipeline preset. "
            "'preview' / 'extended' set a bundle of defaults "
            "for sampling and ranking. 'custom' disables presets."
        ),
    )

    parser.add_argument(
        "--N_max_runs",
        type=int,
        default=1,
        help="Max number of global pipeline rounds.",
    )
    parser.add_argument(
        "--target_template_rmsd_thres",
        type=float,
        default=2.0,
        help="Max RMSD between GT target and prediction to treat as 'template-like'.",
    )

    # Output and ranking caps
    parser.add_argument(
        "--return_topk",
        type=int,
        default=5,
        help="How many designs to keep per task after ranking.",
    )
    parser.add_argument(
        "--min_total_return",
        type=int,
        default=10,
        help="If total successes < this, pad with failed designs up to this total.",
    )
    parser.add_argument(
        "--max_success_return",
        type=int,
        default=20,
        help="Max number of success rows to return.",
    )
    parser.add_argument(
        "--extended_w_af2",
        type=float,
        default=0.5,
        help="Weight for AF2 rank in extended (AF2+Protenix) ranking.",
    )
    parser.add_argument(
        "--extended_w_ptx",
        type=float,
        default=0.5,
        help="Weight for PTX rank in extended (AF2+Protenix) ranking.",
    )

    # Early-stop knobs
    parser.add_argument(
        "--early_stop",
        type=str2bool,
        default=True,
        help="Whether to early-stop the global pipeline if enough successes are accumulated.",
    )
    parser.add_argument(
        "--min_early_stop_rounds",
        type=int,
        default=0,
        help="Min number of rounds before early-stop is allowed.",
    )
    parser.add_argument(
        "--min_early_stop_successes",
        type=int,
        default=1,
        help="Min number of total successes required to trigger early-stop.",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=0,
        help="CPU workers for post-eval analysis (0=auto).",
    )
    parser.add_argument(
        "--length-min-per-len",
        type=int,
        default=10,
        help="Min designs per length when sampling binder length ranges.",
    )

    overridden_keys = _get_overridden_keys(argv)
    pipeline_args, remaining = parser.parse_known_args(argv)

    preset_name = pipeline_args.preset
    if preset_name and preset_name != "custom":
        preset_cfg = PRESETS.get(preset_name, {})
        for key, value in preset_cfg.items():
            if key in overridden_keys:
                continue
            setattr(pipeline_args, key, value)

    return pipeline_args, remaining


def parse_args(argv=None):
    """Top-level parser: pipeline args + model/eval configs via get_configs."""
    pipeline_args, remaining_args = parse_pipeline_args(argv)
    configs = get_configs(remaining_args)
    for tool_name in ["ptx_mini", "ptx"]:
        configs["eval"]["binder"]["tools"][tool_name].update(
            {
                "dtype": configs.dtype,
                "use_deepspeed_evo_attention": configs.use_deepspeed_evo_attention,
            }
        )
    return configs, vars(pipeline_args)


def detect_use_ptx_filter(configs) -> bool:
    """Detect whether Protenix filter is enabled in eval configs."""
    binder_cfg = configs.eval.binder
    for attr in ["eval_protenix", "eval_protenix_mini"]:
        if hasattr(binder_cfg, attr) and getattr(binder_cfg, attr):
            return True
    return False


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


class DesignPipeline(InferenceRunner):
    """Inference runner with v2 run context fields."""

    def __init__(self, *args, use_ptx_filter: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ptx_filter = bool(use_ptx_filter)
        self.run_id: int = 0
        self.run_dir: str = _run_dir(self.configs.dump_dir, 0)
        self.active_tasks: set[str] | None = None


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def main(argv=None):
    configs, p = parse_args(argv)
    p["analysis_workers"] = _resolve_analysis_workers(p.get("analysis_workers"))
    setattr(
        configs,
        "length_min_per_len",
        int(p.get("length_min_per_len", 10)),
    )

    os.environ.setdefault("PXDESIGN_STATUS_DIR", str(configs.dump_dir))
    os.environ["PXDESIGN_STAGE"] = "startup"

    os.makedirs(configs.dump_dir, exist_ok=True)

    # Input + caches
    configs.input_json_path = process_input_file(configs.input_json_path, out_dir=configs.dump_dir)
    download_inference_cache(configs)
    check_tool_weights()

    # Produce pipeline_input.json (rank 0) and make it visible to all ranks.
    pipeline_input_path = os.path.join(configs.dump_dir, "pipeline_input.json")
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            convert_to_bioassembly_dict(x, configs.dump_dir)
        _atomic_write_json(pipeline_input_path, orig_inputs)
    else:
        # Best-effort wait for rank-0 to finish writing (no dist barrier yet).
        while not os.path.exists(pipeline_input_path):
            time.sleep(0.2)

    configs.input_json_path = pipeline_input_path

    # v2 pipeline state
    input_sha256 = _sha256_file(pipeline_input_path)
    if DIST_WRAPPER.rank == 0:
        _init_or_update_pipeline_state(
            dump_dir=str(configs.dump_dir),
            input_sha256=input_sha256,
            n_max_runs=int(p["N_max_runs"]),
            seeds=list(configs.seeds or []),
            target_n_sample=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
        )
    else:
        state_path = os.path.join(configs.dump_dir, "pipeline_state.json")
        while not os.path.exists(state_path):
            time.sleep(0.2)

    state_path = os.path.join(configs.dump_dir, "pipeline_state.json")
    state = _load_pipeline_state(state_path)
    if state is None:
        raise RuntimeError(f"Failed to load pipeline_state.json: {state_path}")

    runs = list(state.get("runs") or [])
    if len(runs) < int(p["N_max_runs"]):
        raise RuntimeError("pipeline_state.json missing required runs.")

    # v2: replicate tasks across ranks; partition design_id instead.
    setattr(configs, "distributed_tasks", False)

    use_ptx_filter = detect_use_ptx_filter(configs)
    runner = DesignPipeline(configs, use_ptx_filter=use_ptx_filter)

    # Determine tasks
    with open(configs.input_json_path, "r") as f:
        inputs = json.load(f)
    task_names = [str(x["name"]) for x in inputs]
    task_input_by_name = {
        str(item.get("name")): item
        for item in inputs
        if isinstance(item, dict) and item.get("name")
    }
    active_tasks: set[str] = set(task_names)

    # Early-stop tracking (rank 0 logic; broadcast each loop)
    cumulative_success: dict[str, int] = {t: 0 for t in task_names}

    last_orig_seqs: dict[str, Any] = {}
    last_use_target_template: bool = False
    finished_run_id: int = 0
    finished_run_seed: int = -1

    for run_id in range(int(p["N_max_runs"])):
        finished_run_id = int(run_id)
        run_seed = int(runs[run_id]["run_seed"])
        finished_run_seed = int(run_seed)

        run_dir = _run_dir(configs.dump_dir, run_id)
        runner.run_id = int(run_id)
        runner.run_dir = str(run_dir)
        runner.active_tasks = set(active_tasks)

        # --------------------
        # Diffusion (all ranks)
        # --------------------
        os.environ["PXDESIGN_STAGE"] = "diffusion"
        os.environ["PXDESIGN_GLOBAL_RUN"] = str(run_id)
        os.environ["PXDESIGN_SEED"] = str(run_seed)

        # Per-rank RNG seed derived from run_seed
        rank_seed = int(derive_seed(run_seed, int(DIST_WRAPPER.rank), digits=9))
        seed_everything(seed=rank_seed, deterministic=True)

        last_orig_seqs = runner._inference(run_seed)

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

        # Rank 0: write diffusion_state.json + persist orig_seqs for ranking-only reruns
        if DIST_WRAPPER.rank == 0:
            diff_dir = os.path.join(run_dir, "diffusion")
            os.makedirs(diff_dir, exist_ok=True)

            task_states = {}
            expected_total = int(getattr(configs.sample_diffusion, "N_sample", 0) or 0)
            for t in task_names:
                struct_dir = _diffusion_struct_dir(configs.dump_dir, run_id, t)
                done = _existing_indices(struct_dir, t)
                done = {i for i in done if 0 <= i < expected_total}
                task_states[t] = {
                    "expected_total": expected_total,
                    "present": int(len(done)),
                }

            _atomic_write_json(
                os.path.join(diff_dir, "diffusion_state.json"),
                {
                    "run_id": int(run_id),
                    "run_seed": int(run_seed),
                    "updated_at": _iso_now(),
                    "tasks": task_states,
                },
            )
            _atomic_write_json(
                os.path.join(diff_dir, "orig_seqs.json"),
                {"run_id": int(run_id), "run_seed": int(run_seed), "orig_seqs": last_orig_seqs},
            )

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

        # --------------------
        # Evaluation (all ranks)
        # --------------------
        os.environ["PXDESIGN_STAGE"] = "evaluation"
        hb = HeartbeatReporter.from_env()
        if hb is not None:
            hb.touch(
                extra={"stage_transition": "evaluation", "run_id": int(run_id)},
                force=True,
            )

        eval_root = os.path.join(run_dir, "eval")
        os.makedirs(eval_root, exist_ok=True)
        active_tasks_state_path = os.path.join(eval_root, "active_tasks_state.json")

        # Optional: target-template decision for PTX filter
        if runner.use_ptx_filter:
            target_template_state = os.path.join(eval_root, "target_template_state.json")
            template_token_prefix = _make_attempt_token(
                run_id=run_id,
                task_name="target_template",
                run_seed=run_seed,
                world_size=int(DIST_WRAPPER.world_size),
            )
            if DIST_WRAPPER.rank == 0:
                use_target_template = False
                if last_orig_seqs:
                    first_task = list(last_orig_seqs.keys())[0]
                    gt_cif_path = os.path.join(
                        _diffusion_struct_dir(configs.dump_dir, run_id, first_task),
                        f"{first_task}_sample_{0:06d}.cif",
                    )
                    target_pred_dir = os.path.join(
                        _eval_task_dir(configs.dump_dir, run_id, first_task),
                        "target_pred",
                    )
                    use_target_template = bool(
                        use_target_template_or_not(
                            configs.eval,
                            p,
                            gt_cif_path,
                            last_orig_seqs.get(first_task),
                            first_task,
                            target_pred_dir,
                            device="cuda:0",
                            seed=run_seed,
                        )
                    )
                template_attempt_token = _make_attempt_token(
                    run_id=run_id,
                    task_name="target_template",
                    run_seed=run_seed,
                    world_size=int(DIST_WRAPPER.world_size),
                    attempt_ns=int(time.time_ns()),
                )
                _atomic_write_json(
                    target_template_state,
                    {
                        "run_id": int(run_id),
                        "run_seed": int(run_seed),
                        "world_size": int(DIST_WRAPPER.world_size),
                        "attempt_token": str(template_attempt_token),
                        "process_start_ns": int(_PROCESS_START_NS),
                        "use_target_template": bool(use_target_template),
                        "updated_ns": int(time.time_ns()),
                        "updated_at": _iso_now(),
                    },
                )
                last_use_target_template = bool(use_target_template)
            else:
                deadline = time.time() + 300
                while True:
                    state_obj = _read_json_obj(target_template_state)
                    if (
                        isinstance(state_obj, dict)
                        and int(state_obj.get("run_id", -1)) == int(run_id)
                        and int(state_obj.get("run_seed", -1)) == int(run_seed)
                        and int(state_obj.get("world_size", -1))
                        == int(DIST_WRAPPER.world_size)
                        and str(state_obj.get("attempt_token", "")).startswith(
                            template_token_prefix
                        )
                        and int(state_obj.get("updated_ns", -1))
                        >= int(_PROCESS_START_NS)
                    ):
                        last_use_target_template = bool(
                            state_obj.get("use_target_template", False)
                        )
                        break
                    if time.time() >= deadline:
                        raise RuntimeError(
                            "Timeout waiting for rank-0 target template decision."
                        )
                    time.sleep(0.2)
        else:
            last_use_target_template = False

        if last_use_target_template:
            configs.eval.binder.tools.ptx.use_template = True
            configs.eval.binder.tools.ptx.use_msa = False
            configs.eval.binder.tools.ptx.model_name = "protenix_mini_tmpl_v0.5.0"
            logger.info("[pipeline] Using target template in Protenix filter")

        eval_state_path = os.path.join(eval_root, "eval_state.json")
        eval_state = {}
        if DIST_WRAPPER.rank == 0 and os.path.exists(eval_state_path):
            try:
                eval_state = json.loads(Path(eval_state_path).read_text())
            except Exception:
                eval_state = {}

        eval_tasks_state: dict[str, dict] = dict(eval_state.get("tasks") or {}) if DIST_WRAPPER.rank == 0 else {}
        expected_total = int(getattr(configs.sample_diffusion, "N_sample", 0) or 0)
        task_eval_meta: list[dict[str, Any]] = []
        sharded_prep_enabled = _is_enabled("PXDESIGN_SHARDED_PREP", True)
        if DIST_WRAPPER.rank == 0:
            logger.info(
                "[pipeline] PXDESIGN_SHARDED_PREP=%s",
                str(bool(sharded_prep_enabled)).lower(),
            )

        for task_name in sorted(active_tasks):
            os.environ["PXDESIGN_TASK_NAME"] = str(task_name)
            struct_dir = _diffusion_struct_dir(configs.dump_dir, run_id, task_name)
            done = _existing_indices(struct_dir, task_name)
            done = {i for i in done if 0 <= i < expected_total}
            diffusion_count = int(len(done))

            task_eval_dir = _eval_task_dir(configs.dump_dir, run_id, task_name)
            os.makedirs(task_eval_dir, exist_ok=True)

            if not os.path.isdir(struct_dir):
                if DIST_WRAPPER.rank == 0:
                    logger.warning(f"No diffusion directory for {task_name}: {struct_dir}")
                continue

            world_size = int(DIST_WRAPPER.world_size)
            pdb_names = [f"{task_name}_sample_{int(i):06d}" for i in sorted(done)]
            if not pdb_names:
                if DIST_WRAPPER.rank == 0:
                    logger.info(
                        f"[pipeline] No matching designs for {task_name} in index range "
                        f"0..{expected_total-1}. Skipping eval."
                    )
                continue

            pending_names = _pending_pdb_names(
                pdb_names,
                task_eval_dir,
                configs.eval.binder,
                run_seed,
            )
            pending_names = sorted(set(pending_names))
            pending_names_digest = _canonical_hash(pending_names)
            if sharded_prep_enabled:
                owned_names_by_rank = {
                    r: pending_names[r::world_size] for r in range(world_size)
                }
            else:
                owned_names_by_rank = {r: [] for r in range(world_size)}
                owned_names_by_rank[0] = list(pending_names)
            my_owned_names = list(owned_names_by_rank.get(int(DIST_WRAPPER.rank), []))
            shard_union = []
            for rank_owned in owned_names_by_rank.values():
                shard_union.extend(rank_owned)
            if sorted(shard_union) != sorted(pending_names):
                raise RuntimeError(
                    f"[pipeline] Shard ownership coverage mismatch for task {task_name}"
                )

            stagein_timeout = _clamp_env_int(
                "PXDESIGN_STAGEIN_SOURCE_TIMEOUT_S", 900, 30, 7200
            )
            stagein_poll = _clamp_env_int(
                "PXDESIGN_STAGEIN_SOURCE_POLL_S", 10, 2, 60
            )

            if DIST_WRAPPER.rank == 0:
                authoritative_chain_payload = _resolve_authoritative_chain_payload_rank0(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    struct_dir=struct_dir,
                    probe_names=pending_names or pdb_names,
                    task_input=task_input_by_name.get(task_name),
                    timeout_s=stagein_timeout,
                    poll_s=stagein_poll,
                )
                attempt_token = _make_attempt_token(
                    run_id=run_id,
                    task_name=task_name,
                    run_seed=run_seed,
                    world_size=world_size,
                    attempt_ns=int(time.time_ns()),
                )
                _write_chain_authority(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    run_id=run_id,
                    run_seed=run_seed,
                    world_size=world_size,
                    attempt_token=attempt_token,
                    pending_names_digest=pending_names_digest,
                    chain_payload=authoritative_chain_payload,
                )

            chain_authority_obj = _wait_for_chain_authority(
                task_eval_dir=task_eval_dir,
                task_name=task_name,
                run_id=run_id,
                run_seed=run_seed,
                world_size=world_size,
                pending_names_digest=pending_names_digest,
                timeout_s=stagein_timeout,
                poll_s=stagein_poll,
            )
            attempt_token = str(chain_authority_obj.get("attempt_token", ""))
            authoritative_chain_payload = dict(chain_authority_obj.get("chain_payload", {}))
            if not attempt_token:
                raise RuntimeError(
                    f"[pipeline] Missing attempt_token in chain authority for task {task_name}"
                )
            cond_chains = list(authoritative_chain_payload.get("cond_chains", []))
            binder_chains = list(authoritative_chain_payload.get("binder_chains", []))

            pdb_dir, converted_names, local_cond_chains, local_binder_chains = _prepare_rank_cache(
                task_eval_dir=task_eval_dir,
                task_name=task_name,
                rank=int(DIST_WRAPPER.rank),
                owned_names=my_owned_names,
                struct_dir=struct_dir,
                condition_chains=cond_chains or None,
                timeout_s=stagein_timeout,
                poll_s=stagein_poll,
            )

            local_chain_payload = _chain_payload(local_cond_chains, local_binder_chains)
            if my_owned_names:
                if not _validate_chain_payload(
                    local_chain_payload, authoritative_chain_payload
                ):
                    raise RuntimeError(
                        f"[pipeline] Rank {int(DIST_WRAPPER.rank)} chain mismatch for task {task_name}"
                    )

            if sorted(converted_names) != sorted(my_owned_names):
                raise RuntimeError(
                    f"[pipeline] Rank {int(DIST_WRAPPER.rank)} conversion mismatch for task {task_name}"
                )

            _update_eval_heartbeat(
                hb,
                task_name=task_name,
                task_eval_dir=task_eval_dir,
                pdb_names=pdb_names,
                eval_cfg=configs.eval.binder,
                seed=run_seed,
            )

            my_pdb_names = list(my_owned_names)

            if my_pdb_names:
                msa_cache_dir = os.path.join(task_eval_dir, "msa_cache")
                os.environ["PXDESIGN_MSA_CACHE_DIR"] = msa_cache_dir
                os.environ["PXDESIGN_MSA_CACHE_FILE"] = os.path.join(
                    msa_cache_dir, "cache.json"
                )
                eval_input = {
                    "task": "binder",
                    "name": task_name,
                    "pdb_dir": pdb_dir,
                    "pdb_names": my_pdb_names,
                    "cond_chains": cond_chains,
                    "binder_chains": binder_chains,
                    "out_dir": task_eval_dir,
                    "orig_seqs": last_orig_seqs.get(task_name),
                    "pred_only": True,
                }
                eval_hb_interval = float(
                    os.environ.get("PXDESIGN_EVAL_HEARTBEAT_INTERVAL", "30") or 30
                )
                keepalive = _start_heartbeat_keepalive(
                    hb,
                    interval_s=eval_hb_interval,
                    extra={"eval_step": "run_task"},
                )
                try:
                    run_task(
                        eval_input,
                        configs.eval,
                        device_id=DIST_WRAPPER.local_rank,
                        seed=run_seed,
                    )
                finally:
                    if keepalive is not None:
                        stop_event, thread = keepalive
                        stop_event.set()
                        thread.join(timeout=1.0)

                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                )

                if DIST_WRAPPER.rank == 0 and hb is not None:
                    hb.touch(
                        extra={"eval_task": task_name, "eval_step": "run_task_complete"},
                        primary_counter="eval_designs",
                        force=True,
                    )

            output_summary = _shard_output_summary(
                int(DIST_WRAPPER.rank),
                task_eval_dir=task_eval_dir,
                owned_names=my_owned_names,
                eval_cfg=configs.eval.binder,
                seed=run_seed,
            )
            _write_shard_manifest(
                task_eval_dir=task_eval_dir,
                task_name=task_name,
                run_id=run_id,
                run_seed=run_seed,
                rank=int(DIST_WRAPPER.rank),
                world_size=world_size,
                owned_names=my_owned_names,
                attempt_token=attempt_token,
                pending_names_digest=pending_names_digest,
                cond_chains=cond_chains,
                binder_chains=binder_chains,
                chain_payload=authoritative_chain_payload,
                output_summary=output_summary,
            )

            task_eval_meta.append(
                {
                    "task_name": task_name,
                    "task_eval_dir": task_eval_dir,
                    "struct_dir": struct_dir,
                    "pdb_names": pdb_names,
                    "pending_names": pending_names,
                    "pending_names_digest": pending_names_digest,
                    "attempt_token": attempt_token,
                    "cond_chains": cond_chains,
                    "binder_chains": binder_chains,
                    "chain_payload": authoritative_chain_payload,
                    "owned_names_by_rank": owned_names_by_rank,
                    "diffusion_count": diffusion_count,
                }
            )

        if DIST_WRAPPER.rank == 0:
            for meta in task_eval_meta:
                task_name = meta["task_name"]
                task_eval_dir = meta["task_eval_dir"]
                struct_dir = meta["struct_dir"]
                pdb_names = meta["pdb_names"]
                pending_names_digest = meta["pending_names_digest"]
                attempt_token = meta["attempt_token"]
                owned_names_by_rank = {int(k): v for k, v in (meta["owned_names_by_rank"] or {}).items()}
                chain_payload = meta["chain_payload"]
                diffusion_count = meta["diffusion_count"]
                agg_timeout = _clamp_env_int(
                    "PXDESIGN_AGG_READY_TIMEOUT_S", 1800, 60, 21600
                )
                agg_poll = _clamp_env_int(
                    "PXDESIGN_AGG_READY_POLL_S", 30, 5, 120
                )

                ready_manifests = _wait_for_shards_ready(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    attempt_token=attempt_token,
                    pending_names_digest=pending_names_digest,
                    expected_owned_map=owned_names_by_rank,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    timeout_s=agg_timeout,
                    poll_s=agg_poll,
                    expected_chain_payload=chain_payload,
                )

                aggregate_pdb_dir = _build_aggregate_inputs(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    run_id=run_id,
                    run_seed=run_seed,
                    world_size=int(DIST_WRAPPER.world_size),
                    all_pdb_names=pdb_names,
                    all_output_manifests=ready_manifests,
                    chain_payload=chain_payload,
                    struct_dir=struct_dir,
                )

                if not pdb_names:
                    continue

                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                )

                os.environ["PXDESIGN_TASK_NAME"] = str(task_name)
                eval_hb_interval = float(
                    os.environ.get("PXDESIGN_EVAL_HEARTBEAT_INTERVAL", "30") or 30
                )
                keepalive = _start_heartbeat_keepalive(
                    hb,
                    interval_s=eval_hb_interval,
                    extra={"eval_step": "aggregate"},
                )
                aggregate_binder_eval(
                    task_name=task_name,
                    eval_dir=task_eval_dir,
                    pdb_dir=aggregate_pdb_dir,
                    pdb_names=pdb_names,
                    cond_chains=chain_payload.get("cond_chains", []),
                    binder_chains=chain_payload.get("binder_chains", []),
                    cfg=configs.eval.binder,
                    seed=run_seed,
                    analysis_workers=int(p.get("analysis_workers")),
                )
                if keepalive is not None:
                    stop_event, thread = keepalive
                    stop_event.set()
                    thread.join(timeout=1.0)

                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                )

                csv_path = os.path.join(task_eval_dir, "sample_level_output.csv")
                run_success = _count_success_from_csv(csv_path)
                cumulative_success[task_name] = cumulative_success.get(task_name, 0) + int(run_success)

                eval_tasks_state[task_name] = {
                    "diffusion_cif_count": diffusion_count,
                    "updated_at": _iso_now(),
                }

            _atomic_write_json(
                eval_state_path,
                {
                    "run_id": int(run_id),
                    "run_seed": int(run_seed),
                    "updated_at": _iso_now(),
                    "tasks": eval_tasks_state,
                },
            )

            logger.info(
                "[pipeline] run %d cumulative_success=%s active_tasks=%s",
                int(run_id),
                cumulative_success,
                sorted(active_tasks),
            )

            # Early-stop update
            newly_done: list[str] = []
            for t in sorted(active_tasks):
                if (
                    bool(p["early_stop"])
                    and (int(run_id) + 1) >= int(p["min_early_stop_rounds"])
                    and int(cumulative_success.get(t, 0)) >= int(p["min_early_stop_successes"])
                ):
                    newly_done.append(t)

            # Always finish on last run
            if int(run_id) == int(p["N_max_runs"]) - 1:
                newly_done = list(active_tasks)

            for t in newly_done:
                active_tasks.discard(t)

        if DIST_WRAPPER.rank == 0:
            _atomic_write_json(
                active_tasks_state_path,
                {
                    "run_id": int(run_id),
                    "run_seed": int(run_seed),
                    "active_tasks": sorted(active_tasks),
                    "updated_ns": int(time.time_ns()),
                    "updated_at": _iso_now(),
                },
            )
            next_active_tasks = sorted(active_tasks)
        else:
            active_sync_timeout = _clamp_env_int(
                "PXDESIGN_ACTIVE_TASKS_TIMEOUT_S", 21600, 60, 86400
            )
            active_sync_poll = _clamp_env_int(
                "PXDESIGN_ACTIVE_TASKS_POLL_S", 1, 1, 30
            )
            next_active_tasks = _wait_for_active_tasks_state(
                path=active_tasks_state_path,
                run_id=run_id,
                run_seed=run_seed,
                timeout_s=active_sync_timeout,
                poll_s=active_sync_poll,
            )
        active_tasks = set(next_active_tasks)

        # Stop if all tasks are done
        if not active_tasks:
            if DIST_WRAPPER.rank == 0:
                logger.info("[pipeline] early-stop satisfied; finalizing")
            break

    # --------------------
    # Final ranking (rank 0)
    # --------------------
    if DIST_WRAPPER.rank == 0:
        os.environ["PXDESIGN_STAGE"] = "ranking"
        hb = HeartbeatReporter.from_env()
        if hb is not None:
            hb.update(
                produced_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                expected_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                extra={"stage_transition": "ranking", "run_id": int(finished_run_id)},
                force=True,
            )

        final_dir = _final_dir(configs.dump_dir, finished_run_id)
        os.makedirs(final_dir, exist_ok=True)

        results_dir = allocate_results_dir(str(configs.dump_dir))
        save_top_designs(
            p,
            configs,
            last_orig_seqs,
            use_template=bool(last_use_target_template),
            final_dir=str(final_dir),
            results_dir=str(results_dir),
        )

        _atomic_write_json(
            os.path.join(final_dir, "final_state.json"),
            {
                "run_id": int(finished_run_id),
                "run_seed": int(finished_run_seed),
                "updated_ns": int(time.time_ns()),
                "updated_at": _iso_now(),
                "message": "final ranking complete",
            },
        )
    else:
        final_state_path = os.path.join(
            _final_dir(configs.dump_dir, finished_run_id), "final_state.json"
        )
        final_state_timeout = _clamp_env_int(
            "PXDESIGN_FINAL_STATE_TIMEOUT_S", 21600, 60, 86400
        )
        final_state_poll = _clamp_env_int(
            "PXDESIGN_FINAL_STATE_POLL_S", 2, 1, 30
        )
        deadline = time.time() + max(int(final_state_timeout), 1)
        while True:
            state_obj = _read_json_obj(final_state_path)
            if (
                isinstance(state_obj, dict)
                and int(state_obj.get("run_id", -1)) == int(finished_run_id)
                and int(state_obj.get("run_seed", -1)) == int(finished_run_seed)
                and int(state_obj.get("updated_ns", -1)) >= int(_PROCESS_START_NS)
            ):
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timeout waiting for final ranking state: {final_state_path} "
                    f"(run_id={int(finished_run_id)}, run_seed={int(finished_run_seed)})"
                )
            time.sleep(max(int(final_state_poll), 1))

    os.environ["PXDESIGN_STAGE"] = "completed"
    hb = HeartbeatReporter.from_env()
    if hb is not None:
        hb.complete(extra={"message": "pipeline complete"})


if __name__ == "__main__":
    main()
