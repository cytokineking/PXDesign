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

import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from pxdbench.metrics import diversity, secondary
from pxdbench.metrics.Kalign import (
    Binder_align_and_calculate_rmsd,
    align_and_calculate_rmsd,
)
from pxdbench.permutation import permute_generated_min_complex_rmsd
from pxdbench.tasks.base import BaseTask
from pxdbench.utils import concat_dict_values, save_eval_results

logger = logging.getLogger(__name__)

_SEQ_RE = re.compile(r"^(?P<name>.+)_seq(?P<idx>\d+)\.txt$")
_AF2_MODEL_RE = re.compile(r"_model(?P<idx>\d+)\.json$")
_AF2_OUTPUT_RE = re.compile(
    r"^(?P<name>.+)_seq(?P<seq_idx>\d+)"
    r"(?P<monomer>_MONOMER_ONLY)?_model(?P<model_idx>\d+)"
    r"\.(?P<ext>json|pdb)$"
)
_SAMPLE_DIR_RE = re.compile(r"^(?P<name>.+)_seq(?P<seq_idx>\d+)$")

Af2Index = Dict[Tuple[str, int, bool], Dict[int, Dict[str, str]]]


def _read_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_model_idx(path: str) -> int | None:
    match = _AF2_MODEL_RE.search(path)
    if not match:
        return None
    try:
        return int(match.group("idx"))
    except Exception:
        return None


def _load_sequences(seq_dir: str) -> Dict[Tuple[str, int], str]:
    sequences: Dict[Tuple[str, int], str] = {}
    if not os.path.isdir(seq_dir):
        return sequences
    for fname in os.listdir(seq_dir):
        match = _SEQ_RE.match(fname)
        if not match:
            continue
        name = match.group("name")
        seq_idx = int(match.group("idx"))
        fpath = os.path.join(seq_dir, fname)
        try:
            with open(fpath, "r") as f:
                seq = f.read().strip()
            sequences[(name, seq_idx)] = seq
        except Exception:
            continue
    return sequences


def _scan_sample_keys(*paths: str) -> List[Tuple[str, int]]:
    keys = set()
    for base in paths:
        if not base or not os.path.isdir(base):
            continue
        for fname in os.listdir(base):
            af2_match = _AF2_OUTPUT_RE.match(fname)
            if af2_match:
                keys.add(
                    (
                        af2_match.group("name"),
                        int(af2_match.group("seq_idx")),
                    )
                )
                continue

            stem = fname
            for ext in (".json", ".pdb", ".cif"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            sample_match = _SAMPLE_DIR_RE.match(stem)
            if sample_match:
                keys.add(
                    (
                        sample_match.group("name"),
                        int(sample_match.group("seq_idx")),
                    )
                )
    return sorted(keys)


def _index_af2_outputs(af2_dir: str) -> Af2Index:
    index: Af2Index = {}
    if not os.path.isdir(af2_dir):
        return index

    start = time.time()
    scanned = 0
    matched = 0
    logger.info("[aggregate] AF2 index build start dir=%s", af2_dir)
    for fname in os.listdir(af2_dir):
        scanned += 1
        match = _AF2_OUTPUT_RE.match(fname)
        if not match:
            continue
        matched += 1
        sample_key = (
            match.group("name"),
            int(match.group("seq_idx")),
            bool(match.group("monomer")),
        )
        model_idx = int(match.group("model_idx"))
        ext = match.group("ext")
        index.setdefault(sample_key, {}).setdefault(model_idx, {})[ext] = (
            os.path.join(af2_dir, fname)
        )
    logger.info(
        "[aggregate] AF2 index build end dir=%s scanned=%d matched=%d samples=%d elapsed_s=%.2f",
        af2_dir,
        scanned,
        matched,
        len(index),
        time.time() - start,
    )
    return index


def _collect_af2_metrics(
    af2_dir: str, name: str, seq_idx: int, monomer: bool = False
) -> Tuple[Dict[str, Any], List[int]]:
    suffix = "_MONOMER_ONLY" if monomer else ""
    pattern = os.path.join(af2_dir, f"{name}_seq{seq_idx}{suffix}_model*.json")
    files = glob(pattern)
    model_stats: Dict[int, Dict[str, Any]] = {}
    for fp in files:
        model_idx = _extract_model_idx(fp)
        if model_idx is None:
            continue
        stats = _read_json(fp)
        if stats is None:
            continue
        model_stats[model_idx] = stats
    if not model_stats:
        return {}, []
    model_ids = sorted(model_stats.keys())
    stats_list = [model_stats[m] for m in model_ids]
    return concat_dict_values(stats_list), model_ids


def _collect_af2_metrics_from_index(
    af2_index: Af2Index,
    af2_dir: str,
    name: str,
    seq_idx: int,
    monomer: bool = False,
) -> Tuple[Dict[str, Any], List[int], List[str]]:
    sample_models = af2_index.get((name, int(seq_idx), bool(monomer)), {})
    model_stats: Dict[int, Dict[str, Any]] = {}
    for model_idx in sorted(sample_models):
        fp = sample_models[model_idx].get("json")
        if not fp:
            continue
        stats = _read_json(fp)
        if stats is None:
            continue
        model_stats[model_idx] = stats
    if not model_stats:
        return {}, [], []
    model_ids = sorted(model_stats.keys())
    stats_list = [model_stats[m] for m in model_ids]
    return (
        concat_dict_values(stats_list),
        model_ids,
        _af2_pdb_paths(af2_dir, name, seq_idx, model_ids, monomer=monomer),
    )


def _af2_pdb_paths(
    af2_dir: str, name: str, seq_idx: int, model_ids: List[int], monomer: bool = False
) -> List[str]:
    suffix = "_MONOMER_ONLY" if monomer else ""
    return [
        os.path.join(af2_dir, f"{name}_seq{seq_idx}{suffix}_model{m}.pdb")
        for m in model_ids
    ]


def _mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(np.mean(vals))


def _bump_lookup_stat(lookup_stats: Dict[str, int] | None, key: str) -> None:
    if lookup_stats is not None:
        lookup_stats[key] = lookup_stats.get(key, 0) + 1


def _collect_ptx_metrics(
    ptx_dir: str,
    name: str,
    seq_idx: int,
    seed: int,
    binder_chain: str,
    suffix: str = "",
    lookup_stats: Dict[str, int] | None = None,
) -> Tuple[Dict[str, Any], str | None]:
    sample_name = f"{name}_seq{seq_idx}"
    pred_root = os.path.join(
        ptx_dir, sample_name, f"seed_{int(seed)}", "predictions"
    )
    if not os.path.isdir(pred_root):
        _bump_lookup_stat(lookup_stats, "missing_pred_root")
        return {}, None

    summary_path = os.path.join(
        pred_root,
        f"{sample_name}_seed_{int(seed)}_summary_confidence_sample_0.json",
    )
    alternate_summary_path = os.path.join(
        pred_root, f"{sample_name}_summary_confidence_sample_0.json"
    )
    summary_files = []
    if os.path.exists(summary_path):
        summary_files = [summary_path]
        _bump_lookup_stat(lookup_stats, "summary_exact")
    elif os.path.exists(alternate_summary_path):
        summary_files = [alternate_summary_path]
        _bump_lookup_stat(lookup_stats, "summary_alternate_exact")
    else:
        summary_files = glob(
            os.path.join(pred_root, "*_summary_confidence_sample_0.json")
        )
        if summary_files:
            _bump_lookup_stat(lookup_stats, "summary_glob")
    if not summary_files:
        _bump_lookup_stat(lookup_stats, "summary_missing")
        return {}, None
    summary = _read_json(summary_files[0])
    if summary is None:
        _bump_lookup_stat(lookup_stats, "summary_invalid")
        return {}, None

    chain_ptm = summary.get("chain_ptm") or []
    chain_iptm = summary.get("chain_iptm") or []

    if hasattr(chain_ptm, "tolist"):
        chain_ptm = chain_ptm.tolist()
    if hasattr(chain_iptm, "tolist"):
        chain_iptm = chain_iptm.tolist()

    if binder_chain == "A":
        binder_idx = 0
    else:
        binder_idx = len(chain_ptm) - 1 if chain_ptm else 0

    target_indices = [i for i in range(len(chain_ptm)) if i != binder_idx]
    ptm_target = [chain_ptm[i] for i in target_indices] if target_indices else []

    def _as_float(value):
        try:
            return float(value)
        except Exception:
            return None

    plddt = _as_float(summary.get("plddt"))
    ptm = _as_float(summary.get("ptm"))
    iptm = _as_float(summary.get("iptm"))
    ptm_binder = (
        _as_float(chain_ptm[binder_idx]) if binder_idx < len(chain_ptm) else None
    )
    iptm_binder = (
        _as_float(chain_iptm[binder_idx]) if binder_idx < len(chain_iptm) else None
    )
    ptm_target_val = _mean([_as_float(v) for v in ptm_target if v is not None])

    metrics = {
        f"ptx{suffix}_plddt": round(plddt, 4) if plddt is not None else None,
        f"ptx{suffix}_ptm": round(ptm, 4) if ptm is not None else None,
        f"ptx{suffix}_iptm": round(iptm, 4) if iptm is not None else None,
        f"ptx{suffix}_ptm_binder": round(ptm_binder, 4)
        if ptm_binder is not None
        else None,
        f"ptx{suffix}_iptm_binder": round(iptm_binder, 4)
        if iptm_binder is not None
        else None,
        f"ptx{suffix}_ptm_target": round(ptm_target_val, 4)
        if ptm_target_val is not None
        else None,
    }

    pdb_path = os.path.join(
        pred_root, f"{sample_name}_seed_{int(seed)}_sample_0.pdb"
    )
    alternate_pdb_path = os.path.join(pred_root, f"{sample_name}_sample_0.pdb")
    if os.path.exists(pdb_path):
        pdb_files = [pdb_path]
        _bump_lookup_stat(lookup_stats, "pdb_exact")
    elif os.path.exists(alternate_pdb_path):
        pdb_files = [alternate_pdb_path]
        _bump_lookup_stat(lookup_stats, "pdb_alternate_exact")
    else:
        pdb_files = glob(os.path.join(pred_root, "*_sample_0.pdb"))
        if pdb_files:
            _bump_lookup_stat(lookup_stats, "pdb_glob")
    if not pdb_files:
        _bump_lookup_stat(lookup_stats, "pdb_missing")
    pred_pdb = pdb_files[0] if pdb_files else None
    return metrics, pred_pdb


def _analysis_worker(args):
    (
        name,
        seq_idx,
        binder_chain,
        design_pdb,
        af2_complex_pdbs,
        af2_monomer_pdbs,
        ptx_pdb,
        ptx_mini_pdb,
    ) = args
    metrics: Dict[str, Any] = {}

    if design_pdb and os.path.isfile(design_pdb):
        try:
            alpha, beta, loop = secondary.cacl_secondary_structure(
                design_pdb, binder_chain
            )
            Rg, ref_ratio = secondary.get_chain_rg(design_pdb, binder_chain)
            metrics.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "loop": loop,
                    "Rg": Rg,
                    "ref_ratio": ref_ratio,
                }
            )
        except Exception:
            pass

    if af2_complex_pdbs:
        rmsds = []
        for pdb_path in af2_complex_pdbs:
            rmsd = None
            if (
                pdb_path
                and os.path.isfile(pdb_path)
                and design_pdb
                and os.path.isfile(design_pdb)
            ):
                try:
                    value = align_and_calculate_rmsd(pdb_path, design_pdb)
                    rmsd = round(value, 2) if value is not None else None
                except Exception:
                    rmsd = None
            rmsds.append(rmsd)
        metrics["af2_complex_pred_design_rmsd"] = rmsds

    if af2_monomer_pdbs:
        binder_rmsd = []
        bound_unbound = []
        for i, monomer_pdb in enumerate(af2_monomer_pdbs):
            complex_pdb = (
                af2_complex_pdbs[i]
                if af2_complex_pdbs and i < len(af2_complex_pdbs)
                else None
            )
            cur_bound = None
            if (
                monomer_pdb
                and complex_pdb
                and os.path.isfile(monomer_pdb)
                and os.path.isfile(complex_pdb)
            ):
                try:
                    value = Binder_align_and_calculate_rmsd(
                        monomer_pdb, complex_pdb, binder_chain
                    )
                    cur_bound = round(value, 2) if value is not None else None
                except Exception:
                    cur_bound = None
            bound_unbound.append(cur_bound)

            cur_binder = None
            if (
                monomer_pdb
                and design_pdb
                and os.path.isfile(monomer_pdb)
                and os.path.isfile(design_pdb)
            ):
                try:
                    value = Binder_align_and_calculate_rmsd(
                        monomer_pdb, design_pdb, binder_chain
                    )
                    cur_binder = round(value, 2) if value is not None else None
                except Exception:
                    cur_binder = None
            binder_rmsd.append(cur_binder)

        metrics["bound_unbound_RMSD"] = bound_unbound
        metrics["af2_binder_pred_design_rmsd"] = binder_rmsd

    if ptx_pdb and design_pdb and os.path.isfile(ptx_pdb) and os.path.isfile(design_pdb):
        try:
            value = permute_generated_min_complex_rmsd(
                ptx_pdb, design_pdb, ptx_pdb
            )
            metrics["ptx_pred_design_rmsd"] = (
                round(value, 2) if value is not None else None
            )
        except Exception:
            pass

    if (
        ptx_mini_pdb
        and design_pdb
        and os.path.isfile(ptx_mini_pdb)
        and os.path.isfile(design_pdb)
    ):
        try:
            value = permute_generated_min_complex_rmsd(
                ptx_mini_pdb, design_pdb, ptx_mini_pdb
            )
            metrics["ptx_mini_pred_design_rmsd"] = (
                round(value, 2) if value is not None else None
            )
        except Exception:
            pass

    return (name, seq_idx, metrics)


def aggregate_binder_eval(
    task_name: str,
    eval_dir: str,
    pdb_dir: str,
    pdb_names: List[str],
    cond_chains: List[str],
    binder_chains: List[str],
    cfg,
    seed: int,
    analysis_workers: int = 1,
    sample_fn: str = "sample_level_output.csv",
    summary_fn: str = "summary_output.json",
) -> Dict[str, str]:
    af2_dir = os.path.join(eval_dir, "af2_pred")
    ptx_dir = os.path.join(eval_dir, "ptx_pred")
    ptx_mini_dir = os.path.join(eval_dir, "ptx_mini_pred")
    seq_dir = os.path.join(eval_dir, "seqs")

    sequences = _load_sequences(seq_dir)
    if not sequences:
        fallback_keys = _scan_sample_keys(af2_dir, ptx_dir, ptx_mini_dir)
        for name, seq_idx in fallback_keys:
            sequences[(name, seq_idx)] = ""

    af2_index = _index_af2_outputs(af2_dir)
    rows: List[Dict[str, Any]] = []
    analysis_jobs = []
    binder_chain = binder_chains[0] if binder_chains else None
    ptx_lookup_stats: Dict[str, int] = {}
    ptx_mini_lookup_stats: Dict[str, int] = {}
    sorted_sequences = sorted(sequences.items())
    total = len(sorted_sequences)
    metrics_start = time.time()

    for i, ((name, seq_idx), sequence) in enumerate(sorted_sequences, start=1):
        length = len(sequence) if isinstance(sequence, str) and sequence else None
        row = {
            "name": name,
            "seq_idx": int(seq_idx),
            "sequence": sequence,
            "length": length,
        }

        af2_metrics, _model_ids, af2_pdbs = _collect_af2_metrics_from_index(
            af2_index, af2_dir, name, seq_idx, monomer=False
        )
        row.update(af2_metrics)

        af2_mono_metrics, _mono_ids, af2_mono_pdbs = (
            _collect_af2_metrics_from_index(
                af2_index, af2_dir, name, seq_idx, monomer=True
            )
        )
        row.update(af2_mono_metrics)

        ptx_metrics, ptx_pdb = _collect_ptx_metrics(
            ptx_dir,
            name,
            seq_idx,
            seed,
            binder_chain,
            lookup_stats=ptx_lookup_stats,
        )
        row.update(ptx_metrics)

        ptx_mini_metrics, ptx_mini_pdb = _collect_ptx_metrics(
            ptx_mini_dir,
            name,
            seq_idx,
            seed,
            binder_chain,
            suffix="_mini",
            lookup_stats=ptx_mini_lookup_stats,
        )
        row.update(ptx_mini_metrics)

        rows.append(row)

        analysis_jobs.append(
            (
                name,
                seq_idx,
                binder_chain,
                os.path.join(pdb_dir, f"{name}.pdb") if pdb_dir else None,
                af2_pdbs,
                af2_mono_pdbs,
                ptx_pdb,
                ptx_mini_pdb,
            )
        )
        if i == total or i % 500 == 0:
            logger.info(
                "[aggregate] metrics collection task=%s done=%d total=%d elapsed_s=%.1f",
                task_name,
                i,
                total,
                time.time() - metrics_start,
            )

    logger.info(
        "[aggregate] PTX lookup stats task=%s stats=%s mini_stats=%s",
        task_name,
        dict(sorted(ptx_lookup_stats.items())),
        dict(sorted(ptx_mini_lookup_stats.items())),
    )

    metrics_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if analysis_jobs:
        analysis_start = time.time()
        logger.info(
            "[aggregate] analysis start task=%s jobs=%d analysis_workers=%d",
            task_name,
            len(analysis_jobs),
            int(analysis_workers or 0),
        )
        if analysis_workers and analysis_workers > 1:
            with ProcessPoolExecutor(max_workers=analysis_workers) as executor:
                for name, seq_idx, metrics in executor.map(
                    _analysis_worker, analysis_jobs
                ):
                    metrics_map[(name, int(seq_idx))] = metrics
        else:
            for job in analysis_jobs:
                name, seq_idx, metrics = _analysis_worker(job)
                metrics_map[(name, int(seq_idx))] = metrics
        logger.info(
            "[aggregate] analysis end task=%s jobs=%d elapsed_s=%.1f",
            task_name,
            len(analysis_jobs),
            time.time() - analysis_start,
        )

    for row in rows:
        key = (row["name"], int(row["seq_idx"]))
        row.update(metrics_map.get(key, {}))

    sample_df = pd.DataFrame(rows)
    if not sample_df.empty:
        sample_df = sample_df.sort_values(by=["name", "seq_idx"])

    div = -1
    if getattr(cfg, "eval_diversity", False) and pdb_dir and binder_chain:
        pdb_paths = [os.path.join(pdb_dir, f"{name}.pdb") for name in pdb_names]
        div = diversity.compute_diversity(pdb_paths, binder_chain)

    BaseTask.compute_success_rate(cfg.filters, sample_df)
    summary_dict = {"task": "binder", "name": task_name}
    summary_dict.update(
        BaseTask.summary_from_df(sample_df, other_metrics={"diversity": div})
    )
    sample_save_path, summary_save_path = save_eval_results(
        sample_df, summary_dict, eval_dir, sample_fn, summary_fn
    )
    logger.info(
        "[aggregate] results written task=%s sample_csv=%s summary_json=%s",
        task_name,
        sample_save_path,
        summary_save_path,
    )
    return {
        "sample_save_path": sample_save_path,
        "summary_save_path": summary_save_path,
    }
