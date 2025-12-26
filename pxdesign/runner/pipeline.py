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


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))


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
    suffix = "_MONOMER_ONLY" if monomer else ""
    for model_id in model_ids:
        model_num = int(model_id) + 1
        fp = os.path.join(
            af2_dir, f"{name}_seq{int(seq_idx)}{suffix}_model{model_num}.json"
        )
        if not os.path.exists(fp):
            return False
    return True


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
    active_tasks: set[str] = set(task_names)

    # Early-stop tracking (rank 0 logic; broadcast each loop)
    cumulative_success: dict[str, int] = {t: 0 for t in task_names}

    last_orig_seqs: dict[str, Any] = {}
    last_use_target_template: bool = False
    finished_run_id: int = 0

    for run_id in range(int(p["N_max_runs"])):
        finished_run_id = int(run_id)
        run_seed = int(runs[run_id]["run_seed"])

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
        if DIST_WRAPPER.rank == 0 and hb is not None:
            hb.update(
                produced_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                expected_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                extra={"stage_transition": "evaluation", "run_id": int(run_id)},
                force=True,
            )

        # Optional: target-template decision for PTX filter
        if runner.use_ptx_filter:
            use_target_template = None
            if DIST_WRAPPER.rank == 0 and last_orig_seqs:
                first_task = list(last_orig_seqs.keys())[0]
                gt_cif_path = os.path.join(
                    _diffusion_struct_dir(configs.dump_dir, run_id, first_task),
                    f"{first_task}_sample_{0:06d}.cif",
                )
                target_pred_dir = os.path.join(
                    _eval_task_dir(configs.dump_dir, run_id, first_task),
                    "target_pred",
                )
                use_target_template = use_target_template_or_not(
                    configs.eval,
                    p,
                    gt_cif_path,
                    last_orig_seqs.get(first_task),
                    first_task,
                    target_pred_dir,
                    device="cuda:0",
                    seed=run_seed,
                )
            last_use_target_template = bool(use_target_template) if DIST_WRAPPER.rank == 0 else False
        else:
            last_use_target_template = False

        if DIST_WRAPPER.world_size > 1:
            tmpl_list = DIST_WRAPPER.all_gather_object(
                last_use_target_template if DIST_WRAPPER.rank == 0 else None
            )
            last_use_target_template = [x for x in tmpl_list if x is not None][0]

        if last_use_target_template:
            configs.eval.binder.tools.ptx.use_template = True
            configs.eval.binder.tools.ptx.use_msa = False
            configs.eval.binder.tools.ptx.model_name = "protenix_mini_tmpl_v0.5.0"
            logger.info("[pipeline] Using target template in Protenix filter")

        eval_root = os.path.join(run_dir, "eval")
        os.makedirs(eval_root, exist_ok=True)
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

        for task_name in sorted(active_tasks):
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

            # CIF->PDB conversion is eval scratch; keep diffusion/structures clean.
            cache_cif_dir = os.path.join(
                task_eval_dir,
                "_cache",
                "cif_to_pdb",
                task_name,
            )

            if DIST_WRAPPER.rank == 0:
                os.makedirs(cache_cif_dir, exist_ok=True)
                for i in sorted(done):
                    src = os.path.join(struct_dir, f"{task_name}_sample_{int(i):06d}.cif")
                    dst = os.path.join(cache_cif_dir, os.path.basename(src))
                    if os.path.exists(dst):
                        continue
                    try:
                        os.link(src, dst)
                    except Exception:
                        shutil.copy(src, dst)

            if DIST_WRAPPER.world_size > 1:
                torch.distributed.barrier()

            if DIST_WRAPPER.rank == 0:
                pdb_dir, pdb_names, cond_chains, binder_chains = convert_cifs_to_pdbs(cache_cif_dir)
                pdb_names = [
                    n
                    for n in pdb_names
                    if any(f"_sample_{i:06d}" in n for i in done)
                ]
            else:
                pdb_dir, pdb_names, cond_chains, binder_chains = None, None, None, None

            if DIST_WRAPPER.world_size > 1:
                shared = DIST_WRAPPER.all_gather_object(
                    (pdb_dir, pdb_names, cond_chains, binder_chains) if DIST_WRAPPER.rank == 0 else None
                )
                pdb_dir, pdb_names, cond_chains, binder_chains = [
                    x for x in shared if x is not None
                ][0]

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
            my_pdb_names = pending_names[DIST_WRAPPER.rank :: DIST_WRAPPER.world_size]

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
                run_task(eval_input, configs.eval, device_id=DIST_WRAPPER.local_rank, seed=run_seed)

                if DIST_WRAPPER.rank == 0 and hb is not None:
                    hb.update(
                        produced_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                        expected_total=int(getattr(configs.sample_diffusion, "N_sample", 0) or 0),
                        extra={"eval_task": task_name, "eval_step": "run_task_complete"},
                        force=True,
                    )

            task_eval_meta.append(
                {
                    "task_name": task_name,
                    "task_eval_dir": task_eval_dir,
                    "pdb_dir": pdb_dir,
                    "pdb_names": pdb_names,
                    "cond_chains": cond_chains,
                    "binder_chains": binder_chains,
                    "diffusion_count": diffusion_count,
                }
            )

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

        if DIST_WRAPPER.rank == 0:
            for meta in task_eval_meta:
                task_name = meta["task_name"]
                task_eval_dir = meta["task_eval_dir"]
                pdb_dir = meta["pdb_dir"]
                pdb_names = meta["pdb_names"]
                cond_chains = meta["cond_chains"]
                binder_chains = meta["binder_chains"]
                diffusion_count = meta["diffusion_count"]

                if not pdb_names:
                    continue

                aggregate_binder_eval(
                    task_name=task_name,
                    eval_dir=task_eval_dir,
                    pdb_dir=pdb_dir,
                    pdb_names=pdb_names,
                    cond_chains=cond_chains,
                    binder_chains=binder_chains,
                    cfg=configs.eval.binder,
                    seed=run_seed,
                    analysis_workers=int(p.get("analysis_workers")),
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

        # Broadcast active_tasks to all ranks
        active_list = DIST_WRAPPER.all_gather_object(sorted(active_tasks) if DIST_WRAPPER.rank == 0 else None)
        active_tasks = set([x for x in active_list if x is not None][0])

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()

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
                "updated_at": _iso_now(),
                "message": "final ranking complete",
            },
        )

    if DIST_WRAPPER.world_size > 1:
        torch.distributed.barrier()

    os.environ["PXDESIGN_STAGE"] = "completed"
    hb = HeartbeatReporter.from_env()
    if hb is not None:
        hb.complete(extra={"message": "pipeline complete"})


if __name__ == "__main__":
    main()
