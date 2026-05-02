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
import errno
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
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
from pxdbench.utils import (
    convert_cif_to_pdb,
    convert_cifs_to_pdbs,
    find_binder_chains,
    find_cond_chains,
    str2bool,
)

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

EVAL_TOOL_ORDER = ("af2_complex", "af2_monomer", "ptx_mini", "ptx")
EVAL_TOOL_GROUP = {
    "af2_complex": "af2_eval",
    "af2_monomer": "af2_eval",
    "ptx_mini": "protenix_eval",
    "ptx": "protenix_eval",
}


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


def _expected_pdb_names(task_name: str, expected_total: int) -> list[str]:
    total = max(int(expected_total or 0), 0)
    return [f"{task_name}_sample_{i:06d}" for i in range(total)]


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


def _is_valid_cached_pdb(path: str, *, parse_check: bool = False) -> bool:
    if not _is_nonempty_file(path):
        return False
    if not parse_check:
        return True
    try:
        with open(path, "r") as f:
            for _ in range(1024):
                line = f.readline()
                if not line:
                    break
                if line.startswith(("ATOM", "HETATM")):
                    return True
    except Exception:
        return False
    return False


_PERSISTENT_WRITE_ERRNOS = {errno.EROFS, errno.EACCES, errno.EPERM}


def _ensure_writable_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    probe = os.path.join(
        path,
        f".pxdesign_write_probe_{int(os.getpid())}_{int(time.time_ns())}",
    )
    try:
        with open(probe, "w") as f:
            f.write("ok")
        os.unlink(probe)
    except Exception:
        try:
            if os.path.exists(probe):
                os.unlink(probe)
        except Exception:
            pass
        raise


def _overlay_to_rw_path(path: str) -> Optional[str]:
    if not path:
        return None
    norm = os.path.normpath(str(path))
    root = os.path.normpath("/root/pxdesign-work")
    if norm == root or not norm.startswith(root + os.sep):
        return None
    rel = norm[len(root) + 1 :]
    parts = [part for part in rel.split(os.sep) if part]
    if not parts:
        return None
    project = parts[0]
    if project.startswith("_rw_"):
        return norm
    return os.path.join(root, f"_rw_{project}", *parts[1:])


def _rw_project_root_for_path(path: str) -> Optional[str]:
    rw_path = _overlay_to_rw_path(path)
    if not rw_path:
        return None
    root = os.path.normpath("/root/pxdesign-work")
    rel = rw_path[len(root) + 1 :]
    parts = [part for part in rel.split(os.sep) if part]
    if not parts:
        return None
    return os.path.join(root, parts[0])


def _marker_norm_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _marker_int(value: Any) -> Optional[int]:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _marker_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "y", "on", "enabled"}:
            return True
        if raw in {"0", "false", "no", "n", "off", "disabled"}:
            return False
    return None


def _marker_task_names(data: dict) -> set[str]:
    tasks = data.get("tasks")
    names: set[str] = set()
    if isinstance(tasks, dict):
        names.update(str(k) for k in tasks.keys())
    elif isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, dict):
                for key in ("task", "task_name", "name"):
                    value = item.get(key)
                    if value:
                        names.add(str(value))
                        break
            elif item is not None:
                names.add(str(item))
    elif tasks:
        names.add(str(tasks))
    return names


def _marker_task_payloads(data: dict, task_name: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for key in (
        "tasks",
        "task_counts",
        "task_status",
        "task_metadata",
        "tasks_by_name",
        "per_task",
        "by_task",
    ):
        value = data.get(key)
        if isinstance(value, dict):
            item = value.get(task_name)
            if item is None:
                item = value.get(str(task_name))
            if isinstance(item, dict):
                payloads.append(item)
        elif isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    continue
                item_name = item.get("task") or item.get("task_name") or item.get("name")
                if str(item_name or "") == str(task_name):
                    payloads.append(item)
    return payloads


def _marker_sections(
    data: dict,
    task_name: str,
    section_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    roots = [data] + _marker_task_payloads(data, task_name)
    sections: list[dict[str, Any]] = []
    for root in roots:
        if not isinstance(root, dict):
            continue
        for key in section_names:
            value = root.get(key)
            if isinstance(value, dict):
                sections.append(value)
                task_value = value.get(task_name) or value.get(str(task_name))
                if isinstance(task_value, dict):
                    sections.append(task_value)
            elif isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    item_name = (
                        item.get("task") or item.get("task_name") or item.get("name")
                    )
                    if item_name is None or str(item_name) == str(task_name):
                        sections.append(item)
        sections.append(root)
    return sections


def _marker_first_value(sections: list[dict[str, Any]], aliases: tuple[str, ...]) -> Any:
    wanted = {_marker_norm_key(alias) for alias in aliases}
    for section in sections:
        for key, value in section.items():
            if _marker_norm_key(key) in wanted:
                return value
    return None


def _marker_first_int(sections: list[dict[str, Any]], aliases: tuple[str, ...]) -> Optional[int]:
    return _marker_int(_marker_first_value(sections, aliases))


def _marker_aliases(*aliases: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for alias in aliases:
        expanded.extend(
            [
                alias,
                f"expected_{alias}",
                f"{alias}_expected",
                f"{alias}_expected_count",
                f"expected_{alias}_count",
            ]
        )
    return tuple(expanded)


def _marker_tool_name(name: Any) -> Optional[str]:
    key = _marker_norm_key(name)
    if key in {"af2complex", "complex", "evalcomplex", "alphafoldcomplex"}:
        return "af2_complex"
    if key in {
        "af2monomer",
        "monomer",
        "bindermonomer",
        "evalbindermonomer",
        "alphafoldmonomer",
    }:
        return "af2_monomer"
    if key in {"ptx", "protenix", "evalprotenix"}:
        return "ptx"
    if key in {"ptxmini", "protenixmini", "evalprotenixmini"}:
        return "ptx_mini"
    return None


def _marker_enabled_tools(data: dict, task_name: str) -> Optional[dict[str, bool]]:
    roots = [data] + _marker_task_payloads(data, task_name)
    sections = _marker_sections(
        data,
        task_name,
        (
            "enabled_tools",
            "enabled_eval_tools",
            "eval_tools",
            "tools_enabled",
            "tool_enabled",
        ),
    )
    sections.extend(
        section
        for section in _marker_sections(data, task_name, ("tools", "eval", "config"))
        if isinstance(section, dict)
    )

    tools: dict[str, bool] = {}

    def _set_tool(raw_name: Any, raw_value: Any = True) -> None:
        canonical = _marker_tool_name(raw_name)
        if canonical is None:
            return
        enabled = raw_value
        if isinstance(raw_value, dict):
            enabled = raw_value.get("enabled", raw_value.get("active", True))
        as_bool = _marker_bool(enabled)
        if as_bool is not None:
            tools[canonical] = bool(as_bool)

    for section in sections:
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("tool") or item.get("id")
                    _set_tool(name, item.get("enabled", True))
                else:
                    _set_tool(item, True)
            continue
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if isinstance(value, dict) and _marker_norm_key(key) in {"af2", "alphafold"}:
                for sub_key, sub_value in value.items():
                    _set_tool(f"af2_{sub_key}", sub_value)
                continue
            if isinstance(value, list) and _marker_norm_key(key) in {
                "enabledtools",
                "enabledevaltools",
                "evaltools",
            }:
                for item in value:
                    _set_tool(item, True)
                continue
            _set_tool(key, value)

    for root in roots:
        if not isinstance(root, dict):
            continue
        for key in (
            "enabled_tools",
            "enabled_eval_tools",
            "eval_tools",
            "tools_enabled",
            "tool_enabled",
        ):
            value = root.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("tool") or item.get("id")
                        _set_tool(name, item.get("enabled", True))
                    else:
                        _set_tool(item, True)

    return tools if tools else None


def _eval_tool_flags(eval_cfg) -> dict[str, bool]:
    return {
        "af2_complex": bool(getattr(eval_cfg, "eval_complex", False)),
        "af2_monomer": bool(getattr(eval_cfg, "eval_binder_monomer", False)),
        "ptx_mini": bool(getattr(eval_cfg, "eval_protenix_mini", False)),
        "ptx": bool(getattr(eval_cfg, "eval_protenix", False)),
    }


def _check_marker_count(
    *,
    status: dict[str, Any],
    label: str,
    expected: int,
    expected_sections: list[dict[str, Any]],
    count_sections: list[dict[str, Any]],
    aliases: tuple[str, ...],
    expected_aliases: Optional[tuple[str, ...]] = None,
    required: bool = True,
) -> bool:
    expected_aliases = expected_aliases or aliases
    marker_expected = _marker_first_int(
        expected_sections,
        _marker_aliases(*expected_aliases),
    )
    observed = _marker_first_int(count_sections, aliases)
    status.setdefault("counts", {})[label] = {
        "expected": int(expected),
        "marker_expected": marker_expected,
        "observed": observed,
    }
    if marker_expected is None:
        if required:
            status["reason"] = f"missing_expected_count:{label}"
            return False
    elif int(marker_expected) != int(expected):
        status["reason"] = (
            f"expected_count_mismatch:{label}:"
            f"marker={int(marker_expected)}:config={int(expected)}"
        )
        return False
    if expected <= 0:
        return True
    if observed is None:
        if required:
            status["reason"] = f"missing_observed_count:{label}"
            return False
        return True
    if int(observed) < int(expected):
        status["reason"] = (
            f"observed_count_insufficient:{label}:"
            f"observed={int(observed)}:expected={int(expected)}"
        )
        return False
    return True


def _marker_has_count(
    expected_sections: list[dict[str, Any]],
    count_sections: list[dict[str, Any]],
    aliases: tuple[str, ...],
) -> bool:
    return (
        _marker_first_int(expected_sections, _marker_aliases(*aliases)) is not None
        or _marker_first_int(count_sections, aliases) is not None
    )


def _aggregation_seed_marker_status(
    path: str,
    *,
    run_dir: str,
    task_name: str,
    eval_cfg=None,
    expected_total: Optional[int] = None,
    pdb_names: Optional[list[str]] = None,
    run_seed: Optional[int] = None,
) -> dict[str, Any]:
    rw_root = _rw_project_root_for_path(path)
    marker_path = (
        os.path.join(rw_root, "output", ".aggregation_seed", "complete.json")
        if rw_root
        else ""
    )
    expected_designs = int(expected_total or 0)
    current_pdb_digest = _canonical_hash(pdb_names or [])
    status: dict[str, Any] = {
        "valid": False,
        "usable_for_completeness": False,
        "usable_for_legacy_scan_bypass": False,
        "usable_for_path_preference": False,
        "mode": "none",
        "reason": "missing_rw_root" if not rw_root else "missing_marker",
        "marker_path": marker_path,
        "rw_root": rw_root or "",
        "expected_designs": expected_designs,
        "current_pdb_names_digest": current_pdb_digest,
        "marker_pdb_names_digest": "",
        "counts": {},
    }
    if not rw_root:
        return status
    data = _read_json_obj(marker_path)
    if not isinstance(data, dict):
        return status

    try:
        version = data.get("version")
        if version is not None and int(version) < 1:
            status["reason"] = "unsupported_marker_version"
            return status
    except Exception:
        status["reason"] = "invalid_marker_version"
        return status

    if not bool(data.get("validated", True)):
        status["reason"] = "marker_not_validated"
        return status
    if str(data.get("run_dir") or "") != str(run_dir):
        status["reason"] = (
            f"run_dir_mismatch:marker={data.get('run_dir')}:current={run_dir}"
        )
        return status
    if run_seed is not None and data.get("run_seed") is not None:
        try:
            if int(data.get("run_seed")) != int(run_seed):
                status["reason"] = "run_seed_mismatch"
                return status
        except Exception:
            status["reason"] = "invalid_run_seed"
            return status
    if str(task_name) not in _marker_task_names(data):
        status["reason"] = "task_missing"
        return status

    status["valid"] = True
    status["usable_for_path_preference"] = True
    status["mode"] = "path_preference"
    status["reason"] = "basic_valid"

    if eval_cfg is None or expected_designs <= 0:
        return status

    count_sections = _marker_sections(
        data,
        task_name,
        (
            "counts",
            "observed_counts",
            "actual_counts",
            "validation_counts",
            "artifact_counts",
            "outputs",
        ),
    )
    expected_sections = _marker_sections(
        data,
        task_name,
        (
            "expected_counts",
            "expected",
            "expected_outputs",
            "required_counts",
            "requirements",
        ),
    )

    expected_tools = _eval_tool_flags(eval_cfg)
    marker_tools = _marker_enabled_tools(data, task_name)
    status["expected_tools"] = dict(expected_tools)
    status["marker_tools"] = dict(marker_tools or {})
    if marker_tools is None:
        status["reason"] = "missing_enabled_tools"
        return status
    for tool_name, enabled in expected_tools.items():
        if bool(marker_tools.get(tool_name)) != bool(enabled):
            status["reason"] = (
                f"enabled_tool_mismatch:{tool_name}:"
                f"marker={bool(marker_tools.get(tool_name))}:config={bool(enabled)}"
            )
            return status

    num_seqs = int(getattr(eval_cfg, "num_seqs", 1) or 1)
    expected_name_seq = int(expected_designs * max(num_seqs, 1))
    status["expected_name_seq"] = expected_name_seq

    local_count = len(pdb_names) if pdb_names is not None else expected_designs
    if int(local_count) < expected_designs:
        status["reason"] = (
            f"local_design_count_insufficient:local={int(local_count)}:"
            f"expected={expected_designs}"
        )
        return status

    if not _check_marker_count(
        status=status,
        label="diffusion_cif",
        expected=expected_designs,
        expected_sections=expected_sections,
        count_sections=count_sections,
        aliases=(
            "diffusion_cif_count",
            "diffusion_cif",
            "diffusion_cifs",
            "diffusion_count",
            "cif_count",
            "cif",
            "cifs",
            "structures",
            "structure_cifs",
        ),
        expected_aliases=(
            "diffusion_cif_count",
            "diffusion_cif",
            "diffusion_cifs",
            "diffusion_count",
            "designs",
            "num_designs",
            "n_designs",
            "samples",
            "n_samples",
            "cif_count",
            "cif",
            "cifs",
            "structures",
            "structure_cifs",
        ),
    ):
        return status
    if not _check_marker_count(
        status=status,
        label="seq_txt",
        expected=expected_name_seq,
        expected_sections=expected_sections,
        count_sections=count_sections,
        aliases=(
            "sequence_txt_count",
            "sequence_txt",
            "seq_txt_count",
            "seq_txt",
            "seqs_txt_count",
            "sequence_count",
            "seq_count",
            "seqs",
        ),
        expected_aliases=(
            "sequence_txt_count",
            "sequence_txt",
            "seq_txt_count",
            "seq_txt",
            "seqs_txt_count",
            "sequence_count",
            "seq_outputs",
            "sequence_outputs",
            "seq_count",
            "seqs",
        ),
    ):
        return status

    af2_model_count = max(len(_model_ids_from_cfg(eval_cfg)), 1)
    af2_expected = expected_name_seq * af2_model_count * int(
        bool(expected_tools["af2_complex"]) + bool(expected_tools["af2_monomer"])
    )
    if af2_expected:
        af2_json_aliases = (
            "af2_json",
            "af2_json_count",
            "af2_jsons",
            "af2_summary_json",
            "af2_summary_json_count",
            "alphafold_json",
            "alphafold_json_count",
        )
        af2_pdb_aliases = (
            "af2_pdb",
            "af2_pdb_count",
            "af2_pdbs",
            "alphafold_pdb",
            "alphafold_pdb_count",
        )
        combined_aliases = (
            "af2_count",
            "af2_outputs",
            "af2_name_seq_count",
            "alphafold_count",
        )
        if _marker_has_count(expected_sections, count_sections, af2_json_aliases) or _marker_has_count(
            expected_sections, count_sections, af2_pdb_aliases
        ):
            if not _check_marker_count(
                status=status,
                label="af2_json",
                expected=af2_expected,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=af2_json_aliases,
                expected_aliases=af2_json_aliases + combined_aliases,
            ):
                return status
            if not _check_marker_count(
                status=status,
                label="af2_pdb",
                expected=af2_expected,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=af2_pdb_aliases,
                expected_aliases=af2_pdb_aliases + combined_aliases,
            ):
                return status
        elif _marker_first_int(expected_sections, _marker_aliases(*combined_aliases)) is not None:
            if not _check_marker_count(
                status=status,
                label="af2_total",
                expected=af2_expected,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=combined_aliases,
            ):
                return status
        else:
            if expected_tools["af2_complex"] and not _check_marker_count(
                status=status,
                label="af2_complex",
                expected=expected_name_seq * af2_model_count,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=(
                    "af2_complex",
                    "af2_complex_count",
                    "af2_complex_outputs",
                    "af2_complex_json",
                    "af2_complex_json_count",
                    "af2_complex_pdb",
                    "af2_complex_pdb_count",
                    "complex_af2_count",
                    "eval_complex_count",
                ),
            ):
                return status
            if expected_tools["af2_monomer"] and not _check_marker_count(
                status=status,
                label="af2_monomer",
                expected=expected_name_seq * af2_model_count,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=(
                    "af2_monomer",
                    "af2_monomer_count",
                    "af2_monomer_outputs",
                    "af2_monomer_json",
                    "af2_monomer_json_count",
                    "af2_monomer_pdb",
                    "af2_monomer_pdb_count",
                    "binder_monomer_count",
                    "eval_binder_monomer_count",
                ),
            ):
                return status

    if expected_tools["ptx"]:
        ptx_json_aliases = (
            "ptx_json",
            "ptx_json_count",
            "ptx_summary_json",
            "ptx_summary_json_count",
            "protenix_json",
            "protenix_json_count",
        )
        ptx_pdb_aliases = (
            "ptx_pdb",
            "ptx_pdb_count",
            "ptx_pdbs",
            "protenix_pdb",
            "protenix_pdb_count",
        )
        ptx_combined_aliases = (
            "ptx_count",
            "ptx_outputs",
            "protenix_count",
            "protenix_outputs",
        )
        if _marker_has_count(expected_sections, count_sections, ptx_json_aliases) or _marker_has_count(
            expected_sections, count_sections, ptx_pdb_aliases
        ):
            if not _check_marker_count(
                status=status,
                label="ptx_json",
                expected=expected_name_seq,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=ptx_json_aliases,
                expected_aliases=ptx_json_aliases + ptx_combined_aliases,
            ):
                return status
            if not _check_marker_count(
                status=status,
                label="ptx_pdb",
                expected=expected_name_seq,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=ptx_pdb_aliases,
                expected_aliases=ptx_pdb_aliases + ptx_combined_aliases,
            ):
                return status
        elif not _check_marker_count(
            status=status,
            label="ptx",
            expected=expected_name_seq,
            expected_sections=expected_sections,
            count_sections=count_sections,
            aliases=ptx_combined_aliases,
        ):
            return status
    if expected_tools["ptx_mini"]:
        ptx_mini_json_aliases = (
            "ptx_mini_json",
            "ptx_mini_json_count",
            "ptx_mini_summary_json",
            "ptx_mini_summary_json_count",
            "protenix_mini_json",
            "protenix_mini_json_count",
        )
        ptx_mini_pdb_aliases = (
            "ptx_mini_pdb",
            "ptx_mini_pdb_count",
            "ptx_mini_pdbs",
            "protenix_mini_pdb",
            "protenix_mini_pdb_count",
        )
        ptx_mini_combined_aliases = (
            "ptx_mini_count",
            "ptx_mini_outputs",
            "protenix_mini_count",
            "protenix_mini_outputs",
        )
        if _marker_has_count(
            expected_sections, count_sections, ptx_mini_json_aliases
        ) or _marker_has_count(expected_sections, count_sections, ptx_mini_pdb_aliases):
            if not _check_marker_count(
                status=status,
                label="ptx_mini_json",
                expected=expected_name_seq,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=ptx_mini_json_aliases,
                expected_aliases=ptx_mini_json_aliases + ptx_mini_combined_aliases,
            ):
                return status
            if not _check_marker_count(
                status=status,
                label="ptx_mini_pdb",
                expected=expected_name_seq,
                expected_sections=expected_sections,
                count_sections=count_sections,
                aliases=ptx_mini_pdb_aliases,
                expected_aliases=ptx_mini_pdb_aliases + ptx_mini_combined_aliases,
            ):
                return status
        elif not _check_marker_count(
            status=status,
            label="ptx_mini",
            expected=expected_name_seq,
            expected_sections=expected_sections,
            count_sections=count_sections,
            aliases=ptx_mini_combined_aliases,
        ):
            return status
    if not expected_tools["ptx_mini"]:
        mini_expected = _marker_first_int(
            expected_sections,
            _marker_aliases(
                "ptx_mini",
                "ptx_mini_count",
                "ptx_mini_outputs",
                "ptx_mini_json",
                "ptx_mini_pdb",
                "protenix_mini_count",
                "protenix_mini_outputs",
            ),
        )
        if mini_expected not in (None, 0):
            status["reason"] = (
                f"disabled_tool_expected_count_nonzero:ptx_mini:{mini_expected}"
            )
            return status

    marker_digest = _marker_first_value(
        expected_sections + count_sections,
        (
            "pdb_names_digest",
            "pdb_name_digest",
            "names_digest",
            "expected_pdb_names_digest",
        ),
    )
    if isinstance(marker_digest, str):
        status["marker_pdb_names_digest"] = marker_digest

    strict_ok = False
    strict_reasons: list[str] = []
    marker_num_seqs = _marker_first_int(
        expected_sections + count_sections,
        ("num_seqs", "num_sequences", "n_seq", "n_seqs"),
    )
    if marker_num_seqs is None or int(marker_num_seqs) != int(num_seqs):
        strict_reasons.append("num_seqs")
    marker_model_ids = _marker_first_value(
        expected_sections + count_sections,
        ("model_ids", "af2_model_ids", "af2_models"),
    )
    if isinstance(marker_model_ids, list):
        try:
            marker_model_ids_norm = sorted(int(x) for x in marker_model_ids)
            current_model_ids = sorted(int(x) for x in _model_ids_from_cfg(eval_cfg))
            if marker_model_ids_norm != current_model_ids:
                strict_reasons.append("model_ids")
        except Exception:
            strict_reasons.append("model_ids")
    else:
        strict_reasons.append("model_ids")
    if not marker_digest or str(marker_digest) != str(current_pdb_digest):
        strict_reasons.append("pdb_names_digest")
    if not strict_reasons:
        strict_ok = True

    if strict_ok:
        status["usable_for_completeness"] = True
        status["usable_for_legacy_scan_bypass"] = True
        status["mode"] = "strict_manifest"
        status["reason"] = "strict_manifest_complete"
    else:
        status["usable_for_legacy_scan_bypass"] = True
        status["mode"] = "legacy_counts_scan_bypass"
        status["reason"] = "legacy_counts_complete:missing_" + ",".join(
            sorted(set(strict_reasons))
        )
    return status


def _marker_allows_eval_scan_bypass(marker_status: Optional[dict[str, Any]]) -> bool:
    if not isinstance(marker_status, dict):
        return False
    return bool(
        marker_status.get("usable_for_completeness")
        or marker_status.get("usable_for_legacy_scan_bypass")
    )


def _aggregation_seed_marker_matches(path: str, *, run_dir: str, task_name: str) -> bool:
    rw_root = _rw_project_root_for_path(path)
    if not rw_root:
        return False
    status = _aggregation_seed_marker_status(
        path,
        run_dir=run_dir,
        task_name=task_name,
    )
    return bool(status.get("usable_for_path_preference"))


def _count_local_files(path: str, pattern: str, *, recursive: bool = False) -> int:
    try:
        base = Path(path)
        if not base.is_dir():
            return 0
        iterator = base.rglob(pattern) if recursive else base.glob(pattern)
        return sum(1 for fp in iterator if fp.is_file() and fp.stat().st_size > 0)
    except Exception:
        return 0


def _local_struct_dir_has_expected_outputs(struct_dir: str, pdb_names: list[str]) -> bool:
    if not struct_dir or not pdb_names or not os.path.isdir(struct_dir):
        return False
    for name in pdb_names:
        if not _is_nonempty_file(os.path.join(struct_dir, f"{name}.cif")):
            return False
    return True


def _local_pdb_cache_has_expected_outputs(pdb_dir: str, pdb_names: list[str]) -> bool:
    if not pdb_dir or not pdb_names or not os.path.isdir(pdb_dir):
        return False
    for name in pdb_names:
        if not _is_valid_cached_pdb(os.path.join(pdb_dir, f"{name}.pdb")):
            return False
    return True


def _local_eval_dir_has_expected_outputs(
    eval_dir: str,
    pdb_names: list[str],
    eval_cfg,
    seed: int,
) -> bool:
    if not eval_dir or not pdb_names or not os.path.isdir(eval_dir):
        return False
    if (
        _count_local_files(
            os.path.join(eval_dir, "attempts"),
            "shard_*_inputs.json",
            recursive=True,
        )
        <= 0
    ):
        return False
    num_seqs = int(getattr(eval_cfg, "num_seqs", 1) or 1)
    expected_name_seq = int(len(pdb_names) * max(num_seqs, 1))
    if _count_local_files(os.path.join(eval_dir, "seqs"), "*.txt") < expected_name_seq:
        return False
    if _pending_pdb_names(pdb_names, eval_dir, eval_cfg, seed):
        return False
    if bool(getattr(eval_cfg, "eval_protenix", False)):
        ptx_pdb_count = _count_local_files(
            os.path.join(eval_dir, "ptx_pred"), "*.pdb", recursive=True
        )
        if ptx_pdb_count < expected_name_seq:
            return False
    if bool(getattr(eval_cfg, "eval_protenix_mini", False)):
        ptx_mini_pdb_count = _count_local_files(
            os.path.join(eval_dir, "ptx_mini_pred"), "*.pdb", recursive=True
        )
        if ptx_mini_pdb_count < expected_name_seq:
            return False
    return True


def _select_rw_overlay_path(
    path: str,
    *,
    run_dir: str,
    task_name: str,
    evidence_ok: bool = False,
    marker_status: Optional[dict[str, Any]] = None,
    allow_marker_path: bool = False,
) -> tuple[str, str]:
    del run_dir, task_name
    rw_path = _overlay_to_rw_path(path)
    if not rw_path:
        return path, "non_overlay"
    if os.path.normpath(rw_path) == os.path.normpath(str(path)):
        return path, "rw"
    if (
        allow_marker_path
        and isinstance(marker_status, dict)
        and bool(marker_status.get("usable_for_path_preference"))
        and os.path.exists(rw_path)
    ):
        mode = str(marker_status.get("mode") or "marker")
        return rw_path, f"marker:{mode}"
    if evidence_ok and os.path.exists(rw_path):
        return rw_path, "evidence"
    return path, "fallback"


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _is_persistent_write_error(exc: BaseException) -> bool:
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, OSError) and getattr(cur, "errno", None) in _PERSISTENT_WRITE_ERRNOS:
            return True
        msg = str(cur).lower()
        if (
            "read-only file system" in msg
            or "[errno 30]" in msg
            or "permission denied" in msg
            or "operation not permitted" in msg
        ):
            return True
    return False


def _exception_mentions_any_path(exc: BaseException, paths: list[str]) -> bool:
    norm_paths = [os.path.normpath(p) for p in paths if p]
    if not norm_paths:
        return False
    for cur in _iter_exception_chain(exc):
        for attr in ("filename", "filename2"):
            value = getattr(cur, attr, None)
            if not value:
                continue
            try:
                candidate = os.path.normpath(str(value))
            except Exception:
                continue
            if any(candidate == p or candidate.startswith(p + os.sep) for p in norm_paths):
                return True
        msg = str(cur)
        if any(p in msg for p in norm_paths):
            return True
    return False


def _resolve_aggregate_cache_root(
    *,
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
) -> tuple[str, str]:
    configured_root = str(os.environ.get("PXDESIGN_AGG_CACHE_ROOT", "") or "").strip()
    scope_leaf = _attempt_dir_name(
        f"task={task_name}|run={int(run_id)}|seed={int(run_seed)}"
    )
    default_root = os.path.join(task_eval_dir, "_cache", "cif_to_pdb", "aggregate")
    fallback_root = os.path.join(
        tempfile.gettempdir(),
        "pxdesign_aggregate_cache",
        scope_leaf,
    )
    candidates: list[tuple[str, str]] = []
    if configured_root:
        configured_base = (
            configured_root
            if os.path.isabs(configured_root)
            else os.path.abspath(os.path.join(task_eval_dir, configured_root))
        )
        candidates.append(("configured", os.path.join(configured_base, scope_leaf)))
    candidates.append(("eval_cache", default_root))
    candidates.append(("tmp_fallback", fallback_root))

    last_error: Exception | None = None
    for source, path in candidates:
        try:
            _ensure_writable_dir(path)
            return path, source
        except Exception as e:
            last_error = e
            logger.warning(
                "[pipeline] aggregate cache root unavailable source=%s path=%s reason=%s",
                source,
                path,
                e,
            )
    raise RuntimeError(
        f"Unable to create writable aggregate cache root for task={task_name}: {last_error}"
    )


def _prune_aggregate_attempt_dirs(agg_root: str) -> None:
    keep_attempts = _clamp_env_int(
        "PXDESIGN_AGG_CACHE_KEEP_ATTEMPTS",
        8,
        1,
        2000,
    )
    try:
        entries = [
            p
            for p in Path(agg_root).iterdir()
            if p.is_dir() and p.name.startswith("attempt_")
        ]
    except Exception:
        return
    if len(entries) <= int(keep_attempts):
        return
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in entries[int(keep_attempts) :]:
        try:
            shutil.rmtree(str(stale))
        except Exception:
            pass


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


def _start_eval_heartbeat_keepalive(
    hb: Optional[HeartbeatReporter],
    *,
    interval_s: float,
    task_name: str,
    task_eval_dir: str,
    pdb_names: list[str],
    eval_cfg,
    seed: int,
    step: str,
    scan_complete: bool = False,
    marker_status: Optional[Dict[str, Any]] = None,
) -> Optional[tuple[threading.Event, threading.Thread]]:
    if hb is None or interval_s <= 0:
        return None

    stop = threading.Event()

    def _loop():
        while not stop.wait(interval_s):
            try:
                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=eval_cfg,
                    seed=seed,
                    step=step,
                    scan_complete=scan_complete,
                    marker_status=marker_status,
                )
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
    step: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    scan_complete: bool = False,
    marker_status: Optional[Dict[str, Any]] = None,
) -> None:
    if hb is None:
        return
    rank = int(DIST_WRAPPER.rank)
    world_size = max(int(DIST_WRAPPER.world_size), 1)
    owned_names = pdb_names[rank::world_size]
    if not owned_names:
        return
    owned_total = int(len(owned_names))
    if scan_complete:
        pending_owned: list[str] = []
        owned_done = int(owned_total)
    else:
        pending_owned = _pending_pdb_names(owned_names, task_eval_dir, eval_cfg, seed)
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
        if scan_complete:
            return int(owned_total)
        if not (eval_monomer if monomer else eval_complex):
            return 0
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
        if scan_complete:
            return int(owned_total)
        if not enabled:
            return 0
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

    def _tool_entry(enabled: bool, done: int) -> dict[str, Any]:
        if not enabled:
            return {"enabled": False, "done": 0, "total": 0}
        return {"enabled": True, "done": int(done), "total": int(owned_total)}

    tool_progress = {
        "af2_complex": _tool_entry(eval_complex, _count_af2_done(monomer=False)),
        "af2_monomer": _tool_entry(eval_monomer, _count_af2_done(monomer=True)),
        "ptx_mini": _tool_entry(
            eval_ptx_mini, _count_ptx_done(ptx_mini_dir, eval_ptx_mini)
        ),
        "ptx": _tool_entry(eval_ptx, _count_ptx_done(ptx_dir, eval_ptx)),
    }

    active_tool = next(
        (
            tool
            for tool in EVAL_TOOL_ORDER
            if bool(tool_progress[tool].get("enabled"))
            and int(tool_progress[tool].get("done", 0) or 0)
            < int(tool_progress[tool].get("total", 0) or 0)
        ),
        None,
    )
    active_group = EVAL_TOOL_GROUP.get(active_tool) if active_tool else None
    eval_extra: Dict[str, Any] = {
        "task": task_name,
        "owned_total": owned_total,
        "owned_done": owned_done,
        "owned_pending": int(len(pending_owned)),
        "global_total": int(len(pdb_names)),
        "expected_outputs": expected_outputs,
        "tool_progress": tool_progress,
        "active_tool": active_tool,
        "active_group": active_group,
    }
    if step:
        eval_extra["step"] = str(step)
    if active_tool:
        active_entry = tool_progress[active_tool]
        eval_extra["active_done"] = int(active_entry.get("done", 0) or 0)
        eval_extra["active_total"] = int(active_entry.get("total", 0) or 0)
    if metrics:
        eval_extra["metrics"] = metrics
    if isinstance(marker_status, dict):
        eval_extra["marker_valid"] = bool(marker_status.get("valid"))
        eval_extra["marker_mode"] = str(marker_status.get("mode") or "")
        eval_extra["marker_reason"] = str(marker_status.get("reason") or "")
        eval_extra["marker_path"] = str(marker_status.get("marker_path") or "")
    if scan_complete:
        eval_extra["phase"] = "marker_complete"

    hb.update(
        produced_total=owned_done,
        expected_total=owned_total,
        primary_counter="eval_designs",
        extra={"eval": eval_extra},
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


def _pick_canonical_decision_task(
    *,
    active_tasks: set[str] | None,
    last_orig_seqs: dict[str, Any] | None,
    fallback_tasks: list[str],
) -> Optional[str]:
    """Pick a deterministic task used for target-template decision."""
    if active_tasks:
        candidates = sorted({str(t) for t in active_tasks if str(t).strip()})
        if candidates:
            return str(candidates[0])
    if isinstance(last_orig_seqs, dict) and last_orig_seqs:
        candidates = sorted({str(t) for t in last_orig_seqs.keys() if str(t).strip()})
        if candidates:
            return str(candidates[0])
    if fallback_tasks:
        candidates = sorted({str(t) for t in fallback_tasks if str(t).strip()})
        if candidates:
            return str(candidates[0])
    return None


def _is_readable_json_file(path: str) -> bool:
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except Exception:
        return False


def _inspect_target_pred_artifacts(
    *,
    target_pred_dir: str,
    task_name: str,
    run_seed: int,
) -> dict[str, Any]:
    """
    Inspect target_pred artifacts for sample_0.

    Returns:
      {
        "complete": bool,
        "reason": str,
        "seed_dir": str,
        "pred_dir": str,
        "pred_cif_path": str | None,
        "summary_json_path": str | None,
        "has_success_file": bool,
      }
    """
    seed_dir = os.path.join(target_pred_dir, str(task_name), f"seed_{int(run_seed)}")
    pred_dir = os.path.join(seed_dir, "predictions")
    info: dict[str, Any] = {
        "complete": False,
        "reason": "",
        "seed_dir": seed_dir,
        "pred_dir": pred_dir,
        "pred_cif_path": None,
        "summary_json_path": None,
        "has_success_file": bool(os.path.isfile(os.path.join(seed_dir, "SUCCESS_FILE"))),
    }
    if not os.path.isdir(pred_dir):
        info["reason"] = f"missing predictions dir: {pred_dir}"
        return info

    cif_candidates = sorted(Path(pred_dir).glob("*_sample_0.cif"))
    nonempty_cifs = [str(fp) for fp in cif_candidates if _is_nonempty_file(str(fp))]
    if not nonempty_cifs:
        info["reason"] = "missing non-empty sample_0 cif"
        return info

    pred_cif_path = str(nonempty_cifs[0])
    info["pred_cif_path"] = pred_cif_path

    pred_prefix = Path(pred_cif_path).name.removesuffix("_sample_0.cif")
    expected_summary = os.path.join(
        pred_dir,
        f"{pred_prefix}_summary_confidence_sample_0.json",
    )
    if not _is_readable_json_file(expected_summary):
        info["reason"] = (
            "missing readable summary_confidence_sample_0 json for matched cif prefix"
        )
        return info

    info["summary_json_path"] = expected_summary
    info["complete"] = True
    return info


def _state_allows_target_template_reuse(
    *,
    state_obj: Optional[dict],
    run_id: int,
    run_seed: int,
    decision_task: str,
    rmsd_threshold: float,
) -> bool:
    if not isinstance(state_obj, dict):
        return False
    try:
        if int(state_obj.get("run_id", -1)) != int(run_id):
            return False
        if int(state_obj.get("run_seed", -1)) != int(run_seed):
            return False
        if str(state_obj.get("decision_task", "")) != str(decision_task):
            return False
        if "use_target_template" not in state_obj:
            return False
        state_threshold = float(state_obj.get("target_template_rmsd_thres"))
        if not math.isfinite(state_threshold):
            return False
        if abs(float(state_threshold) - float(rmsd_threshold)) > 1e-9:
            return False
    except Exception:
        return False
    return True


def _derive_target_template_from_existing_artifacts(
    *,
    gt_cif_path: str,
    pred_cif_path: str,
    rmsd_threshold: float,
) -> tuple[Optional[bool], Optional[float], str]:
    """
    Derive use_target_template from existing target_pred artifacts.

    This helper is read-only with respect to target_pred paths:
    temporary outputs are written under a temp directory.
    """
    if not _is_nonempty_file(gt_cif_path):
        return None, None, f"missing/non-empty gt cif: {gt_cif_path}"
    if not _is_nonempty_file(pred_cif_path):
        return None, None, f"missing/non-empty pred cif: {pred_cif_path}"

    try:
        from pxdbench.metrics.Kalign import align_and_calculate_target_rmsd
        from pxdbench.permutation import permute_generated_min_complex_rmsd
        from pxdbench.utils import convert_cif_to_pdb
        from pxdesign.runner.helpers import keep_target_chains
    except Exception as e:
        return None, None, f"import error: {e}"

    tmp_parent = os.path.join(tempfile.gettempdir(), "pxdesign_target_template_rmsd")
    os.makedirs(tmp_parent, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rmsd_", dir=tmp_parent) as tmp_dir:
            gt_pdb_path = os.path.join(tmp_dir, "gt_target.pdb")
            pred_pdb_path = os.path.join(tmp_dir, "pred_target.pdb")

            convert_cif_to_pdb(gt_cif_path, gt_pdb_path)
            keep_target_chains(gt_pdb_path, gt_pdb_path)
            convert_cif_to_pdb(pred_cif_path, pred_pdb_path)
            permute_generated_min_complex_rmsd(pred_pdb_path, gt_pdb_path, pred_pdb_path)
            rmsd = float(align_and_calculate_target_rmsd(pred_pdb_path, gt_pdb_path))
            if not math.isfinite(rmsd):
                return None, None, f"non-finite rmsd: {rmsd}"
    except Exception as e:
        return None, None, f"rmsd derivation error: {e}"

    use_target_template = bool(float(rmsd) >= float(rmsd_threshold))
    return use_target_template, float(rmsd), ""


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


def _has_any_pending_eval_work(
    *,
    dump_dir: str,
    run_id: int,
    active_tasks: set[str],
    expected_total: int,
    eval_cfg,
    run_seed: int,
) -> bool:
    """Return True when at least one active task still has pending eval outputs."""
    for task_name in sorted(active_tasks):
        struct_dir = _diffusion_struct_dir(dump_dir, run_id, task_name)
        done = _existing_indices(struct_dir, task_name)
        done = {i for i in done if 0 <= i < int(expected_total)}
        pdb_names = [f"{task_name}_sample_{int(i):06d}" for i in sorted(done)]
        if not pdb_names:
            continue
        task_eval_dir = _eval_task_dir(dump_dir, run_id, task_name)
        pending_names = _pending_pdb_names(
            pdb_names,
            task_eval_dir,
            eval_cfg,
            run_seed,
        )
        if pending_names:
            return True
    return False


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


def _marker_state_path(task_eval_dir: str) -> str:
    return os.path.join(task_eval_dir, "marker_state.json")


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


def _chain_ids_for_hint_compare(value: Any) -> list[str]:
    ids = _normalize_chain_ids(value)
    canonical: set[str] = set()
    for chain in ids:
        chain_s = str(chain).strip()
        if not chain_s:
            continue
        if chain_s.endswith("0") and chain_s[:-1].isalpha():
            chain_s = chain_s[:-1]
        # CIF->PDB conversion trims chain IDs to one character.
        chain_s = chain_s[0]
        canonical.add(chain_s)
    return sorted(canonical)


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


def _map_chain_hints_to_observed_ids(
    *,
    hints: list[str],
    observed_chain_ids: list[str],
    task_name: str,
    hint_label: str,
) -> list[str]:
    """
    Map user-facing chain hints (e.g., A/B/C) onto observed CIF chain IDs
    (e.g., A0/B0/C0) deterministically.
    """
    normalized_hints = _normalize_chain_ids(hints)
    if not normalized_hints:
        return []

    observed = _normalize_chain_ids(observed_chain_ids)
    if not observed:
        raise RuntimeError(
            f"Cannot map {hint_label} chain hints for {task_name}: no observed chains."
        )

    by_key: dict[str, list[str]] = {}
    for chain in observed:
        keys = _chain_ids_for_hint_compare([chain])
        key = keys[0] if keys else str(chain).strip()[:1]
        by_key.setdefault(key, []).append(chain)
    for key in by_key:
        by_key[key] = sorted(set(by_key[key]))

    resolved: list[str] = []
    missing: list[str] = []
    for hint in normalized_hints:
        if hint in observed:
            resolved.append(hint)
            continue

        keys = _chain_ids_for_hint_compare([hint])
        key = keys[0] if keys else str(hint).strip()[:1]
        matches = list(by_key.get(key, []))
        if not matches:
            missing.append(hint)
            continue
        if len(matches) == 1:
            resolved.append(matches[0])
            continue

        hint_s = str(hint).strip()
        preferred = next((m for m in matches if m == f"{hint_s}0"), None)
        if preferred is None:
            preferred = next((m for m in matches if m.startswith(hint_s)), None)
        resolved.append(preferred or matches[0])

    if missing:
        raise RuntimeError(
            f"Unmatched {hint_label} chain hints for {task_name}: hints={sorted(missing)} "
            f"observed={observed}"
        )

    return sorted(set(resolved))


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

    probe_names = [str(x) for x in (probe_names or []) if str(x)]
    if not probe_names:
        raise RuntimeError(
            f"Cannot resolve chain authority for {task_name}: no probe names available."
        )

    canonical_prefix = f"{task_name}_sample_"
    canonical_names = sorted(
        {
            name
            for name in probe_names
            if str(name).strip() and str(name).startswith(canonical_prefix)
        }
    )
    if not canonical_names:
        canonical_names = sorted(set(probe_names))
    max_primary_candidates = _clamp_env_int(
        "PXDESIGN_CHAIN_PROBE_MAX_PRIMARY_CANDIDATES",
        1,
        1,
        100000,
    )
    probe_candidates = canonical_names[: max(1, min(len(canonical_names), max_primary_candidates))]

    def _cond_hint_variants(base_hint: list[str]) -> list[list[str]]:
        normalized = _normalize_chain_ids(base_hint)
        variants: list[list[str]] = [normalized]
        with_suffix = []
        for chain in normalized:
            chain_s = str(chain).strip()
            if chain_s and not any(ch.isdigit() for ch in chain_s):
                with_suffix.append(f"{chain_s}0")
            else:
                with_suffix.append(chain_s)
        with_suffix = _normalize_chain_ids(with_suffix)
        if with_suffix != normalized:
            variants.append(with_suffix)
        without_suffix = []
        for chain in normalized:
            chain_s = str(chain).strip()
            if chain_s.endswith("0") and chain_s[:-1].isalpha():
                without_suffix.append(chain_s[:-1])
            else:
                without_suffix.append(chain_s)
        without_suffix = _normalize_chain_ids(without_suffix)
        if without_suffix and without_suffix not in variants:
            variants.append(without_suffix)
        return variants

    def _infer_chains_from_cif(
        *,
        name: str,
        condition_hint: list[str] | None,
        stage_timeout_s: int,
    ) -> tuple[list[str], list[str]]:
        cif_path = os.path.join(struct_dir, f"{name}.cif")
        deadline = time.time() + max(int(stage_timeout_s), 1)
        while True:
            if not _is_nonempty_file(cif_path):
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Cannot resolve chain authority for {task_name}: failed reading {cif_path} "
                        f"within timeout_s={int(stage_timeout_s)} (missing or empty file)"
                    )
                time.sleep(max(1, int(poll_s)))
                continue

            try:
                if condition_hint:
                    inferred_cond_local = _normalize_chain_ids(condition_hint)
                else:
                    inferred_cond_local = _normalize_chain_ids(find_cond_chains(cif_path))
                inferred_binder_local = _normalize_chain_ids(
                    find_binder_chains(cif_path, inferred_cond_local)
                )
                return inferred_cond_local, inferred_binder_local
            except Exception as e:
                # Hinted-chain mismatches are deterministic for an existing CIF.
                # Fail fast so caller can try the next hint variant (e.g. A -> A0)
                # instead of waiting the full mount timeout.
                if condition_hint:
                    raise RuntimeError(
                        f"Cannot resolve chain authority for {task_name}: "
                        f"condition hint {sorted(_normalize_chain_ids(condition_hint))} "
                        f"does not match {cif_path} ({e})"
                    ) from e
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"Cannot resolve chain authority for {task_name}: failed reading {cif_path} "
                        f"within timeout_s={int(stage_timeout_s)} ({e})"
                    ) from e
                time.sleep(max(1, int(poll_s)))

    hint_variants = _cond_hint_variants(cond_hint) if cond_hint else [[]]
    selected_cond_hint: list[str] = list(cond_hint)
    inferred_cond: list[str] = []
    inferred_binder: list[str] = []
    selected_probe_name: Optional[str] = None
    primary_candidate = probe_candidates[0]
    probe_deadline = time.time() + max(int(timeout_s), 1)
    per_probe_timeout = max(1, int(timeout_s) // max(len(probe_candidates), 1))
    last_error: Exception | None = None
    for probe_candidate in probe_candidates:
        remaining = int(probe_deadline - time.time())
        if remaining <= 0:
            break
        candidate_timeout = max(1, min(remaining, max(per_probe_timeout, 5)))
        for hint_variant in hint_variants:
            try:
                inferred_cond, inferred_binder = _infer_chains_from_cif(
                    name=probe_candidate,
                    condition_hint=hint_variant or None,
                    stage_timeout_s=candidate_timeout,
                )
                selected_probe_name = probe_candidate
                selected_cond_hint = list(hint_variant)
                last_error = None
                break
            except Exception as e:
                last_error = e
        if selected_probe_name is not None:
            break
    if selected_probe_name is None:
        if last_error is not None:
            raise RuntimeError(
                f"Cannot resolve chain authority for {task_name}: no readable probe CIF found "
                f"within timeout_s={int(timeout_s)} candidates={int(len(probe_candidates))} "
                f"({last_error})"
            ) from last_error
        raise RuntimeError(
            f"Cannot resolve chain authority for {task_name}: no readable probe CIF found."
        )

    if selected_probe_name != primary_candidate:
        logger.warning(
            "[pipeline] chain probe fallback task=%s preferred_primary=%s selected_primary=%s",
            task_name,
            primary_candidate,
            selected_probe_name,
        )

    sanity_candidates = [name for name in canonical_names if name > selected_probe_name]
    if not sanity_candidates:
        sanity_candidates = [name for name in canonical_names if name != selected_probe_name]
    sanity_probe_name = sanity_candidates[0] if sanity_candidates else None

    selected_probe_cif = os.path.join(struct_dir, f"{selected_probe_name}.cif")
    observed_cond = _normalize_chain_ids(find_cond_chains(selected_probe_cif))
    observed_binder = _normalize_chain_ids(
        find_binder_chains(selected_probe_cif, observed_cond)
    )
    observed_all = sorted(set(observed_cond + observed_binder))
    cond_hint = _map_chain_hints_to_observed_ids(
        hints=_normalize_chain_ids(selected_cond_hint),
        observed_chain_ids=observed_all,
        task_name=task_name,
        hint_label="condition",
    )
    binder_hint = _map_chain_hints_to_observed_ids(
        hints=binder_hint,
        observed_chain_ids=observed_all,
        task_name=task_name,
        hint_label="binder",
    )

    if cond_hint and inferred_cond and _canonical_hash(
        _chain_ids_for_hint_compare(cond_hint)
    ) != _canonical_hash(_chain_ids_for_hint_compare(inferred_cond)):
        raise RuntimeError(
            f"Condition chain mismatch for {task_name}: input={cond_hint} inferred={inferred_cond}"
        )
    if binder_hint and inferred_binder and _canonical_hash(
        _chain_ids_for_hint_compare(binder_hint)
    ) != _canonical_hash(_chain_ids_for_hint_compare(inferred_binder)):
        raise RuntimeError(
            f"Binder chain mismatch for {task_name}: input={binder_hint} inferred={inferred_binder}"
        )

    cond_final = cond_hint if cond_hint else inferred_cond
    binder_final = binder_hint if binder_hint else inferred_binder
    if not cond_final and not binder_final:
        raise RuntimeError(
            f"Failed to resolve non-empty chain authority for {task_name}."
        )

    sanity_status = "skipped_no_second_sample"
    if sanity_probe_name is not None:
        sanity_cond, sanity_binder = _infer_chains_from_cif(
            name=sanity_probe_name,
            condition_hint=cond_final,
            stage_timeout_s=timeout_s,
        )
        if _canonical_hash(_chain_ids_for_hint_compare(cond_final)) != _canonical_hash(
            _chain_ids_for_hint_compare(sanity_cond)
        ) or _canonical_hash(_chain_ids_for_hint_compare(binder_final)) != _canonical_hash(
            _chain_ids_for_hint_compare(sanity_binder)
        ):
            raise RuntimeError(
                "Chain sanity-check mismatch for "
                f"{task_name}: expected cond={sorted(cond_final)} binder={sorted(binder_final)}; "
                f"observed cond={sorted(sanity_cond)} binder={sorted(sanity_binder)} "
                f"on sample={sanity_probe_name}"
            )
        sanity_status = "passed"

    payload = _chain_payload(
        cond_final,
        binder_final,
        probe_metadata={
            "chain_probe_mode": "single_probe_one_sanity",
            "chain_probe_primary_sample": selected_probe_name,
            "chain_probe_sanity_sample": sanity_probe_name,
            "chain_probe_sanity_status": sanity_status,
        },
    )
    logger.info(
        "[pipeline] chain authority resolved task=%s mode=%s primary=%s sanity=%s sanity_status=%s",
        task_name,
        payload.get("chain_probe_mode"),
        payload.get("chain_probe_primary_sample"),
        payload.get("chain_probe_sanity_sample"),
        payload.get("chain_probe_sanity_status"),
    )
    return payload


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


def _write_marker_state(
    *,
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    world_size: int,
    pdb_names_digest: str,
    marker_status: dict[str, Any],
) -> None:
    payload = {
        "task": str(task_name),
        "run_id": int(run_id),
        "run_seed": int(run_seed),
        "world_size": int(world_size),
        "pdb_names_digest": str(pdb_names_digest),
        "marker_status": marker_status,
        "process_start_ns": int(_PROCESS_START_NS),
        "updated_ns": int(time.time_ns()),
        "updated_at": _iso_now(),
        "version": 1,
    }
    _atomic_write_json(_marker_state_path(task_eval_dir), payload)


def _wait_for_marker_state(
    *,
    task_eval_dir: str,
    task_name: str,
    run_id: int,
    run_seed: int,
    world_size: int,
    pdb_names_digest: str,
    timeout_s: int,
    poll_s: int,
) -> dict[str, Any]:
    deadline = time.time() + max(int(timeout_s), 1)
    poll_s = max(int(poll_s), 1)
    path = _marker_state_path(task_eval_dir)
    while True:
        data = _read_json_obj(path)
        if (
            isinstance(data, dict)
            and str(data.get("task") or "") == str(task_name)
            and int(data.get("run_id", -1)) == int(run_id)
            and int(data.get("run_seed", -1)) == int(run_seed)
            and int(data.get("world_size", -1)) == int(world_size)
            and str(data.get("pdb_names_digest", "")) == str(pdb_names_digest)
            and int(data.get("updated_ns", -1)) >= int(_PROCESS_START_NS)
            and isinstance(data.get("marker_status"), dict)
        ):
            return dict(data.get("marker_status") or {})
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timeout waiting for marker state for {task_name}: path={path}"
            )
        time.sleep(poll_s)


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


def _shared_cache_root(task_eval_dir: str, task_name: str) -> str:
    return os.path.join(
        task_eval_dir,
        "_cache",
        "cif_to_pdb",
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
    binder_chains: list[str] | None,
    timeout_s: int,
    poll_s: int,
) -> tuple[str, list[str], list[str], list[str], dict[str, Any]]:
    cache_dir = _shared_cache_root(task_eval_dir, task_name)
    owned_names = sorted(set(owned_names))
    try:
        _ensure_writable_dir(cache_dir)
    except Exception as e:
        logger.error(
            "[pipeline] rank cache root unavailable source=eval_cache task=%s rank=%d path=%s reason=%s",
            task_name,
            int(rank),
            cache_dir,
            e,
        )
        raise RuntimeError(
            f"Rank cache root is not writable for {task_name} rank {rank}: {cache_dir} ({e})"
        ) from e

    incremental_enabled = _is_enabled("PXDESIGN_PDB_CACHE_INCREMENTAL", True)
    cache_mode = "incremental" if incremental_enabled else "rebuild_owned"
    parse_check = _is_enabled("PXDESIGN_PDB_CACHE_PARSE_CHECK", True)
    reused_names: list[str] = []
    missing_names: list[str] = []
    if incremental_enabled:
        validation_start = time.time()
        last_validation_log = validation_start
        logger.info(
            "[pipeline] rank cache validation start task=%s rank=%d owned=%d parse_check=%s cache_dir=%s",
            task_name,
            int(rank),
            int(len(owned_names)),
            str(bool(parse_check)).lower(),
            cache_dir,
        )
        for idx, name in enumerate(owned_names, start=1):
            dst = os.path.join(cache_dir, f"{name}.pdb")
            if _is_valid_cached_pdb(dst, parse_check=parse_check):
                reused_names.append(name)
            else:
                missing_names.append(name)
            now = time.time()
            if idx == len(owned_names) or idx % 1000 == 0 or now - last_validation_log >= 30:
                logger.info(
                    "[pipeline] rank cache validation progress task=%s rank=%d checked=%d reused=%d missing=%d owned=%d cache_dir=%s",
                    task_name,
                    int(rank),
                    int(idx),
                    int(len(reused_names)),
                    int(len(missing_names)),
                    int(len(owned_names)),
                    cache_dir,
                )
                last_validation_log = now
    else:
        missing_names = list(owned_names)

    logger.info(
        "[pipeline] rank cache start task=%s rank=%d owned=%d reused=%d missing=%d parse_check=%s cache_source=eval_cache cache_dir=%s",
        task_name,
        int(rank),
        int(len(owned_names)),
        int(len(reused_names)),
        int(len(missing_names)),
        str(bool(parse_check)).lower(),
        cache_dir,
    )

    converted_names: list[str] = []
    new_cond_chains = _normalize_chain_ids(condition_chains or [])
    new_binder_chains = _normalize_chain_ids(binder_chains or [])

    converted_cond_reference: list[str] = []
    converted_binder_reference: list[str] = []
    if missing_names:
        last_convert_log = time.time()
        for name in missing_names:
            src_cif = os.path.join(struct_dir, f"{name}.cif")
            dst_pdb = os.path.join(cache_dir, f"{name}.pdb")
            tmp_pdb = dst_pdb + ".tmp"
            deadline = time.time() + max(int(timeout_s), 1)
            binder_for_convert = list(new_binder_chains)
            allow_binder_fallback = True
            while True:
                try:
                    if not _is_nonempty_file(src_cif):
                        raise FileNotFoundError(src_cif)
                    if os.path.exists(tmp_pdb):
                        os.unlink(tmp_pdb)
                    if not binder_for_convert:
                        cond_for_inference = list(new_cond_chains)
                        if not cond_for_inference:
                            cond_for_inference = _normalize_chain_ids(
                                find_cond_chains(src_cif)
                            )
                        binder_for_convert = _normalize_chain_ids(
                            find_binder_chains(src_cif, cond_for_inference)
                        )
                    inferred_cond, inferred_binder = convert_cif_to_pdb(
                        src_cif,
                        tmp_pdb,
                        binder_chains=binder_for_convert or None,
                    )
                    if not _is_valid_cached_pdb(tmp_pdb, parse_check=parse_check):
                        raise RuntimeError(
                            f"Converted PDB missing or invalid for {task_name} rank {rank}: {name}"
                        )
                    os.replace(tmp_pdb, dst_pdb)
                    converted_names.append(name)
                    now = time.time()
                    if len(converted_names) == len(missing_names) or len(converted_names) % 100 == 0 or now - last_convert_log >= 30:
                        logger.info(
                            "[pipeline] rank cache progress task=%s rank=%d mode=%s reused=%d converted=%d missing=%d owned=%d cache_dir=%s",
                            task_name,
                            int(rank),
                            cache_mode,
                            int(len(reused_names)),
                            int(len(converted_names)),
                            int(len(missing_names) - len(converted_names)),
                            int(len(owned_names)),
                            cache_dir,
                        )
                        last_convert_log = now

                    cur_cond = _normalize_chain_ids(inferred_cond)
                    cur_binder = _normalize_chain_ids(inferred_binder)
                    if not converted_cond_reference:
                        converted_cond_reference = list(cur_cond)
                    elif _canonical_hash(_chain_ids_for_hint_compare(converted_cond_reference)) != _canonical_hash(
                        _chain_ids_for_hint_compare(cur_cond)
                    ):
                        raise RuntimeError(
                            f"Condition chain mismatch across converted samples for {task_name}: "
                            f"{converted_cond_reference} vs {cur_cond} (name={name})"
                        )
                    if not converted_binder_reference:
                        converted_binder_reference = list(cur_binder)
                    elif _canonical_hash(_chain_ids_for_hint_compare(converted_binder_reference)) != _canonical_hash(
                        _chain_ids_for_hint_compare(cur_binder)
                    ):
                        raise RuntimeError(
                            f"Binder chain mismatch across converted samples for {task_name}: "
                            f"{converted_binder_reference} vs {cur_binder} (name={name})"
                        )
                    if not new_binder_chains:
                        new_binder_chains = _normalize_chain_ids(binder_for_convert)
                    break
                except Exception as e:
                    if (
                        allow_binder_fallback
                        and binder_for_convert
                        and isinstance(e, ValueError)
                    ):
                        binder_for_convert = []
                        new_binder_chains = []
                        allow_binder_fallback = False
                        continue
                    if _is_persistent_write_error(e) and _exception_mentions_any_path(
                        e,
                        [cache_dir, tmp_pdb, dst_pdb],
                    ):
                        logger.error(
                            "[pipeline] rank cache write failed task=%s rank=%d path=%s reason=%s",
                            task_name,
                            int(rank),
                            dst_pdb,
                            e,
                        )
                        raise RuntimeError(
                            f"Persistent rank cache write error for {task_name} rank {rank}: {dst_pdb} ({e})"
                        ) from e
                    if os.path.exists(tmp_pdb):
                        try:
                            os.unlink(tmp_pdb)
                        except Exception:
                            pass
                    if time.time() >= deadline:
                        raise RuntimeError(
                            f"Failed to convert source CIF {src_cif} for {task_name} rank {rank} "
                            f"within timeout_s={int(timeout_s)} ({e})"
                        ) from e
                    time.sleep(max(1, int(poll_s)))

        converted_names = sorted(set(converted_names))
        if converted_names != missing_names:
            raise RuntimeError(
                f"Converted CIF count mismatch for {task_name} rank {rank}: "
                f"expected={missing_names} got={converted_names}"
            )

    if converted_cond_reference:
        if new_cond_chains and _canonical_hash(
            _chain_ids_for_hint_compare(new_cond_chains)
        ) != _canonical_hash(_chain_ids_for_hint_compare(converted_cond_reference)):
            raise RuntimeError(
                f"Condition chain mismatch for {task_name} rank {rank}: "
                f"hint={new_cond_chains} converted={converted_cond_reference}"
            )
        new_cond_chains = list(converted_cond_reference)
    if converted_binder_reference:
        if new_binder_chains and _canonical_hash(
            _chain_ids_for_hint_compare(new_binder_chains)
        ) != _canonical_hash(_chain_ids_for_hint_compare(converted_binder_reference)):
            raise RuntimeError(
                f"Binder chain mismatch for {task_name} rank {rank}: "
                f"hint={new_binder_chains} converted={converted_binder_reference}"
            )
        new_binder_chains = list(converted_binder_reference)

    ready_names: list[str] = []
    for name in owned_names:
        dst = os.path.join(cache_dir, f"{name}.pdb")
        if not _is_valid_cached_pdb(dst, parse_check=parse_check):
            raise RuntimeError(
                f"Rank cache missing/invalid PDB for {task_name} rank {rank}: {name}"
            )
        ready_names.append(name)

    logger.info(
        "[pipeline] rank cache prepared task=%s rank=%d mode=%s pdb_reused_count=%d pdb_converted_count=%d owned_count=%d",
        task_name,
        int(rank),
        cache_mode,
        int(len(reused_names)),
        int(len(converted_names)),
        int(len(owned_names)),
    )

    cache_stats = {
        "pdb_cache_mode": cache_mode,
        "pdb_reused_count": int(len(reused_names)),
        "pdb_converted_count": int(len(converted_names)),
        "owned_count": int(len(owned_names)),
        "incremental_enabled": bool(incremental_enabled),
    }
    return (
        cache_dir,
        ready_names,
        list(new_cond_chains),
        list(new_binder_chains),
        cache_stats,
    )


def _chain_payload(
    cond_chains: list[str],
    binder_chains: list[str],
    probe_metadata: Optional[dict] = None,
) -> dict:
    cond = sorted(cond_chains)
    binder = sorted(binder_chains)
    payload = {
        "cond_chains": cond,
        "binder_chains": binder,
        "chain_digest": _canonical_hash(cond + binder),
        "chain_count": int(len(cond) + len(binder)),
    }
    if isinstance(probe_metadata, dict):
        for key in (
            "chain_probe_mode",
            "chain_probe_primary_sample",
            "chain_probe_sanity_sample",
            "chain_probe_sanity_status",
        ):
            if key in probe_metadata:
                payload[key] = probe_metadata.get(key)
    return payload


def _validate_chain_payload(
    payload: dict, expected_chain_payload: Optional[dict]
) -> bool:
    if expected_chain_payload is None:
        return True
    return (
        _canonical_hash(_chain_ids_for_hint_compare(payload.get("cond_chains", [])))
        == _canonical_hash(
            _chain_ids_for_hint_compare(expected_chain_payload.get("cond_chains", []))
        )
        and _canonical_hash(_chain_ids_for_hint_compare(payload.get("binder_chains", [])))
        == _canonical_hash(
            _chain_ids_for_hint_compare(expected_chain_payload.get("binder_chains", []))
        )
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
    attempt_token: str,
) -> tuple[str, dict[str, int]]:
    del attempt_token
    parse_check = _is_enabled("PXDESIGN_PDB_CACHE_PARSE_CHECK", True)
    shared_pdb_dir = os.path.join(task_eval_dir, "_cache", "cif_to_pdb", task_name)
    os.makedirs(shared_pdb_dir, exist_ok=True)

    source_counts = {
        "owner_rank_cache": 0,
        "shared_cache": 0,
        "other_rank_cache": 0,
        "aggregate_fallback_converted_count": 0,
    }
    unresolved_names: list[str] = []
    for name in sorted(all_pdb_names):
        shared_path = os.path.join(shared_pdb_dir, f"{name}.pdb")
        if _is_valid_cached_pdb(shared_path, parse_check=parse_check):
            source_counts["shared_cache"] = int(source_counts["shared_cache"]) + 1
        else:
            unresolved_names.append(name)

    if unresolved_names:
        try:
            _ensure_writable_dir(shared_pdb_dir)
        except Exception as e:
            logger.error(
                "[pipeline] aggregate shared cache unavailable task=%s path=%s unresolved=%d reason=%s",
                task_name,
                shared_pdb_dir,
                int(len(unresolved_names)),
                e,
            )
            raise RuntimeError(
                f"Aggregate shared cache is not writable for {task_name}: {shared_pdb_dir} ({e})"
            ) from e
        rehydrate_timeout = _clamp_env_int(
            "PXDESIGN_STAGEIN_SOURCE_TIMEOUT_S", 900, 30, 7200
        )
        rehydrate_poll = _clamp_env_int(
            "PXDESIGN_STAGEIN_SOURCE_POLL_S", 10, 2, 60
        )
        rehydrate_tmp = tempfile.mkdtemp(
            prefix=f"aggregate_rehydrate_{task_name}_",
            dir=shared_pdb_dir,
        )
        rehydrate_cif_dir = os.path.join(rehydrate_tmp, "cifs")
        rehydrate_pdb_dir = os.path.join(rehydrate_tmp, "pdbs")
        os.makedirs(rehydrate_cif_dir, exist_ok=True)
        try:
            for name in unresolved_names:
                src_cif = os.path.join(struct_dir, f"{name}.cif")
                dst_cif = os.path.join(rehydrate_cif_dir, f"{name}.cif")
                if not _copy_with_retry(
                    src_cif,
                    dst_cif,
                    timeout_s=rehydrate_timeout,
                    poll_s=rehydrate_poll,
                ):
                    raise RuntimeError(
                        f"Missing required aggregate source CIF for {task_name}: {src_cif}"
                    )
            _, converted_names, _, _ = convert_cifs_to_pdbs(
                rehydrate_cif_dir,
                out_pdb_dir=rehydrate_pdb_dir,
                condition_chains=_normalize_chain_ids((chain_payload or {}).get("cond_chains")) or None,
            )
            converted_set = set(converted_names or [])
            source_counts["aggregate_fallback_converted_count"] = int(len(converted_set))
            for name in unresolved_names:
                if name not in converted_set:
                    raise RuntimeError(
                        f"Failed to rehydrate missing aggregate PDB for {task_name}: {name}"
                    )
                src = os.path.join(rehydrate_pdb_dir, f"{name}.pdb")
                if not _is_valid_cached_pdb(src, parse_check=parse_check):
                    raise RuntimeError(
                        f"Rehydrated aggregate PDB missing or invalid for {task_name}: {src}"
                    )
                dst = os.path.join(shared_pdb_dir, f"{name}.pdb")
                dst_tmp = f"{dst}.rehydrate_tmp_{int(time.time_ns())}"
                try:
                    if os.path.exists(dst_tmp):
                        os.unlink(dst_tmp)
                    try:
                        os.link(src, dst_tmp)
                    except Exception:
                        shutil.copy2(src, dst_tmp)
                    os.replace(dst_tmp, dst)
                finally:
                    if os.path.exists(dst_tmp):
                        try:
                            os.unlink(dst_tmp)
                        except Exception:
                            pass
                if not _is_valid_cached_pdb(dst, parse_check=parse_check):
                    raise RuntimeError(
                        f"Failed to install rehydrated aggregate PDB for {task_name}: {dst}"
                    )
        finally:
            shutil.rmtree(rehydrate_tmp, ignore_errors=True)

    for name in sorted(all_pdb_names):
        src = os.path.join(shared_pdb_dir, f"{name}.pdb")
        if not _is_valid_cached_pdb(src, parse_check=parse_check):
            raise RuntimeError(
                f"Missing required aggregate source {name} in shared cache for {task_name}"
            )

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
        "aggregate_pdb_dir": shared_pdb_dir,
        "source_counts": dict(source_counts),
        "updated_at": _iso_now(),
    }
    aggregate_inputs_path = os.path.join(task_eval_dir, "aggregate_inputs.json")
    try:
        _atomic_write_json(aggregate_inputs_path, aggregate_inputs)
    except Exception as e:
        fallback_inputs_path = os.path.join(shared_pdb_dir, "aggregate_inputs.json")
        logger.warning(
            "[pipeline] aggregate_inputs write fallback task=%s primary=%s fallback=%s reason=%s",
            task_name,
            aggregate_inputs_path,
            fallback_inputs_path,
            e,
        )
        _atomic_write_json(fallback_inputs_path, aggregate_inputs)

    logger.info(
        "[pipeline] aggregate inputs task=%s root_source=%s owner_rank_cache=%d shared_cache=%d other_rank_cache=%d aggregate_fallback_converted_count=%d",
        task_name,
        "shared_cache",
        int(source_counts["owner_rank_cache"]),
        int(source_counts["shared_cache"]),
        int(source_counts["other_rank_cache"]),
        int(source_counts["aggregate_fallback_converted_count"]),
    )
    return shared_pdb_dir, dict(source_counts)


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
                diffusion_marker_status = _aggregation_seed_marker_status(
                    _eval_task_dir(configs.dump_dir, run_id, t),
                    run_dir=f"runs/run_{int(run_id):03d}",
                    task_name=t,
                    eval_cfg=configs.eval.binder,
                    expected_total=expected_total,
                    pdb_names=_expected_pdb_names(t, expected_total),
                    run_seed=run_seed,
                )
                if _marker_allows_eval_scan_bypass(diffusion_marker_status):
                    present_count = int(expected_total)
                    logger.info(
                        "[pipeline] diffusion_state count scan bypassed task=%s run=%d seed=%d marker_mode=%s reason=%s",
                        t,
                        int(run_id),
                        int(run_seed),
                        str(diffusion_marker_status.get("mode")),
                        str(diffusion_marker_status.get("reason")),
                    )
                else:
                    done = _existing_indices(struct_dir, t)
                    done = {i for i in done if 0 <= i < expected_total}
                    present_count = int(len(done))
                task_states[t] = {
                    "expected_total": expected_total,
                    "present": present_count,
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
                decision_source = "emergency_fallback"
                decision_task = ""
                try:
                    target_template_rmsd_thres = float(
                        p.get("target_template_rmsd_thres", 2.0)
                    )
                except Exception:
                    target_template_rmsd_thres = 2.0
                target_template_rmsd: Optional[float] = None

                expected_total_for_eval = int(
                    getattr(configs.sample_diffusion, "N_sample", 0) or 0
                )
                marker_start = time.time()
                marker_status_by_task: dict[str, dict[str, Any]] = {}
                marker_complete_all = bool(active_tasks)
                for marker_task in sorted(active_tasks):
                    marker_pdb_names = _expected_pdb_names(
                        marker_task,
                        expected_total_for_eval,
                    )
                    status = _aggregation_seed_marker_status(
                        _eval_task_dir(configs.dump_dir, run_id, marker_task),
                        run_dir=f"runs/run_{int(run_id):03d}",
                        task_name=marker_task,
                        eval_cfg=configs.eval.binder,
                        expected_total=expected_total_for_eval,
                        pdb_names=marker_pdb_names,
                        run_seed=run_seed,
                    )
                    marker_status_by_task[marker_task] = status
                    if not _marker_allows_eval_scan_bypass(status):
                        marker_complete_all = False
                if marker_complete_all:
                    has_pending_eval = False
                    logger.info(
                        "[pipeline] pending eval scan bypassed for target-template decision run=%d seed=%d tasks=%d elapsed_s=%.2f marker_modes=%s",
                        int(run_id),
                        int(run_seed),
                        int(len(marker_status_by_task)),
                        time.time() - marker_start,
                        {
                            task: status.get("mode")
                            for task, status in marker_status_by_task.items()
                        },
                    )
                else:
                    logger.info(
                        "[pipeline] pending eval scan start for target-template decision run=%d seed=%d marker_elapsed_s=%.2f marker_reasons=%s",
                        int(run_id),
                        int(run_seed),
                        time.time() - marker_start,
                        {
                            task: status.get("reason")
                            for task, status in marker_status_by_task.items()
                        },
                    )
                    pending_scan_start = time.time()
                    has_pending_eval = _has_any_pending_eval_work(
                        dump_dir=configs.dump_dir,
                        run_id=run_id,
                        active_tasks=set(active_tasks),
                        expected_total=expected_total_for_eval,
                        eval_cfg=configs.eval.binder,
                        run_seed=run_seed,
                    )
                    logger.info(
                        "[pipeline] pending eval scan end for target-template decision run=%d seed=%d has_pending=%s elapsed_s=%.2f",
                        int(run_id),
                        int(run_seed),
                        str(bool(has_pending_eval)).lower(),
                        time.time() - pending_scan_start,
                    )

                if not has_pending_eval:
                    decision_source = "no_pending_eval"
                    logger.info(
                        "[pipeline] no pending eval names; skipping target-template decision for run=%d seed=%d",
                        int(run_id),
                        int(run_seed),
                    )
                else:
                    decision_task = _pick_canonical_decision_task(
                        active_tasks=active_tasks,
                        last_orig_seqs=last_orig_seqs,
                        fallback_tasks=task_names,
                    )

                if has_pending_eval and decision_task:
                    gt_cif_path = os.path.join(
                        _diffusion_struct_dir(configs.dump_dir, run_id, decision_task),
                        f"{decision_task}_sample_{0:06d}.cif",
                    )
                    target_pred_dir = os.path.join(
                        _eval_task_dir(configs.dump_dir, run_id, decision_task),
                        "target_pred",
                    )
                    artifact_info = _inspect_target_pred_artifacts(
                        target_pred_dir=target_pred_dir,
                        task_name=decision_task,
                        run_seed=run_seed,
                    )
                    prev_state = _read_json_obj(target_template_state)

                    if bool(artifact_info.get("complete", False)):
                        logger.info(
                            "[pipeline] reuse existing target_pred artifacts: task=%s seed=%d success_file=%s",
                            decision_task,
                            int(run_seed),
                            str(bool(artifact_info.get("has_success_file", False))).lower(),
                        )
                        if _state_allows_target_template_reuse(
                            state_obj=prev_state,
                            run_id=run_id,
                            run_seed=run_seed,
                            decision_task=decision_task,
                            rmsd_threshold=target_template_rmsd_thres,
                        ):
                            use_target_template = bool(
                                prev_state.get("use_target_template", False)
                            )
                            decision_source = "reused_state"
                            logger.info(
                                "[pipeline] target_pred decision reused from trusted state: task=%s seed=%d use_target_template=%s",
                                decision_task,
                                int(run_seed),
                                str(bool(use_target_template)).lower(),
                            )
                        else:
                            logger.info(
                                "[pipeline] target_pred complete, state untrusted; deriving decision from artifacts: task=%s seed=%d",
                                decision_task,
                                int(run_seed),
                            )
                            derived_use, derived_rmsd, derive_error = (
                                _derive_target_template_from_existing_artifacts(
                                    gt_cif_path=gt_cif_path,
                                    pred_cif_path=str(
                                        artifact_info.get("pred_cif_path") or ""
                                    ),
                                    rmsd_threshold=target_template_rmsd_thres,
                                )
                            )
                            if derived_use is not None:
                                use_target_template = bool(derived_use)
                                decision_source = "artifact_derived"
                                target_template_rmsd = (
                                    float(derived_rmsd)
                                    if derived_rmsd is not None
                                    else None
                                )
                                rmsd_for_log = (
                                    float(target_template_rmsd)
                                    if target_template_rmsd is not None
                                    else float("nan")
                                )
                                logger.info(
                                    "[pipeline] target_pred artifact-derived decision: task=%s seed=%d rmsd=%.4f threshold=%.4f use_target_template=%s",
                                    decision_task,
                                    int(run_seed),
                                    rmsd_for_log,
                                    float(target_template_rmsd_thres),
                                    str(bool(use_target_template)).lower(),
                                )
                            else:
                                use_target_template = False
                                decision_source = "emergency_fallback"
                                logger.warning(
                                    "[pipeline] target_pred complete, artifact derivation failed; using emergency fallback use_target_template=False: task=%s seed=%d error=%s",
                                    decision_task,
                                    int(run_seed),
                                    str(derive_error),
                                )
                    else:
                        logger.info(
                            "[pipeline] target_pred incomplete, recomputing: task=%s seed=%d reason=%s",
                            decision_task,
                            int(run_seed),
                            str(artifact_info.get("reason", "unknown")),
                        )
                        use_target_template = bool(
                            use_target_template_or_not(
                                configs.eval,
                                p,
                                gt_cif_path,
                                (last_orig_seqs or {}).get(decision_task),
                                decision_task,
                                target_pred_dir,
                                device="cuda:0",
                                seed=run_seed,
                            )
                        )
                        decision_source = "recomputed"
                        logger.info(
                            "[pipeline] target_pred decision recomputed: task=%s seed=%d use_target_template=%s",
                            decision_task,
                            int(run_seed),
                            str(bool(use_target_template)).lower(),
                        )
                elif has_pending_eval:
                    logger.warning(
                        "[pipeline] target template decision task unavailable; using emergency fallback use_target_template=False"
                    )
                template_attempt_token = _make_attempt_token(
                    run_id=run_id,
                    task_name="target_template",
                    run_seed=run_seed,
                    world_size=int(DIST_WRAPPER.world_size),
                    attempt_ns=int(time.time_ns()),
                )
                target_template_state_obj: dict[str, Any] = {
                    "run_id": int(run_id),
                    "run_seed": int(run_seed),
                    "world_size": int(DIST_WRAPPER.world_size),
                    "attempt_token": str(template_attempt_token),
                    "process_start_ns": int(_PROCESS_START_NS),
                    "decision_task": str(decision_task or ""),
                    "target_template_rmsd_thres": float(target_template_rmsd_thres),
                    "decision_source": str(decision_source),
                    "use_target_template": bool(use_target_template),
                    "updated_ns": int(time.time_ns()),
                    "updated_at": _iso_now(),
                }
                if target_template_rmsd is not None:
                    target_template_state_obj["target_template_rmsd"] = float(
                        target_template_rmsd
                    )
                _atomic_write_json(
                    target_template_state,
                    target_template_state_obj,
                )
                last_use_target_template = bool(use_target_template)
            else:
                wait_timeout_s = _clamp_env_int(
                    "PXDESIGN_TARGET_TEMPLATE_STATE_TIMEOUT_S", 300, 30, 7200
                )
                wait_poll_ms = _clamp_env_int(
                    "PXDESIGN_TARGET_TEMPLATE_STATE_POLL_MS", 200, 50, 5000
                )
                deadline = time.time() + float(wait_timeout_s)
                poll_s = max(float(wait_poll_ms) / 1000.0, 0.05)
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
                    time.sleep(poll_s)
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
            task_eval_dir = _eval_task_dir(configs.dump_dir, run_id, task_name)
            os.makedirs(task_eval_dir, exist_ok=True)

            world_size = int(DIST_WRAPPER.world_size)

            stagein_timeout = _clamp_env_int(
                "PXDESIGN_STAGEIN_SOURCE_TIMEOUT_S", 900, 30, 7200
            )
            stagein_poll = _clamp_env_int(
                "PXDESIGN_STAGEIN_SOURCE_POLL_S", 10, 2, 60
            )

            expected_pdb_names = _expected_pdb_names(task_name, expected_total)
            marker_pdb_names_digest = _canonical_hash(expected_pdb_names)
            if DIST_WRAPPER.rank == 0:
                marker_check_start = time.time()
                marker_status = _aggregation_seed_marker_status(
                    task_eval_dir,
                    run_dir=f"runs/run_{int(run_id):03d}",
                    task_name=task_name,
                    eval_cfg=configs.eval.binder,
                    expected_total=expected_total,
                    pdb_names=expected_pdb_names,
                    run_seed=run_seed,
                )
                _write_marker_state(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    run_id=run_id,
                    run_seed=run_seed,
                    world_size=world_size,
                    pdb_names_digest=marker_pdb_names_digest,
                    marker_status=marker_status,
                )
                logger.info(
                    "[pipeline] marker check task=%s run=%d seed=%d mode=%s usable_scan_bypass=%s reason=%s elapsed_s=%.2f counts=%s",
                    task_name,
                    int(run_id),
                    int(run_seed),
                    str(marker_status.get("mode")),
                    str(_marker_allows_eval_scan_bypass(marker_status)).lower(),
                    str(marker_status.get("reason")),
                    time.time() - marker_check_start,
                    marker_status.get("counts", {}),
                )
            else:
                marker_status = _wait_for_marker_state(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    run_id=run_id,
                    run_seed=run_seed,
                    world_size=world_size,
                    pdb_names_digest=marker_pdb_names_digest,
                    timeout_s=stagein_timeout,
                    poll_s=stagein_poll,
                )

            marker_scan_bypass = _marker_allows_eval_scan_bypass(marker_status)
            if marker_scan_bypass:
                pdb_names = list(expected_pdb_names)
                diffusion_count = int(expected_total)
                rw_struct_dir = _overlay_to_rw_path(struct_dir)
                if rw_struct_dir and os.path.isdir(rw_struct_dir):
                    struct_dir = rw_struct_dir
                elif rw_struct_dir and DIST_WRAPPER.rank == 0:
                    logger.info(
                        "[pipeline] marker-complete rw struct dir unavailable; using original struct dir task=%s rw_struct_dir=%s",
                        task_name,
                        rw_struct_dir,
                    )
            else:
                if not os.path.isdir(struct_dir):
                    if DIST_WRAPPER.rank == 0:
                        logger.warning(f"No diffusion directory for {task_name}: {struct_dir}")
                    continue
                done = _existing_indices(struct_dir, task_name)
                done = {i for i in done if 0 <= i < expected_total}
                diffusion_count = int(len(done))
                pdb_names = [f"{task_name}_sample_{int(i):06d}" for i in sorted(done)]

            if not pdb_names:
                if DIST_WRAPPER.rank == 0:
                    logger.info(
                        f"[pipeline] No matching designs for {task_name} in index range "
                        f"0..{expected_total-1}. Skipping eval."
                    )
                continue

            pdb_names_digest = _canonical_hash(pdb_names)
            pending_scan_start = time.time()
            if marker_scan_bypass:
                pending_names = []
                if DIST_WRAPPER.rank == 0:
                    logger.info(
                        "[pipeline] main eval pending scan bypassed task=%s run=%d seed=%d designs=%d marker_mode=%s reason=%s elapsed_s=%.2f",
                        task_name,
                        int(run_id),
                        int(run_seed),
                        int(len(pdb_names)),
                        str(marker_status.get("mode")),
                        str(marker_status.get("reason")),
                        time.time() - pending_scan_start,
                    )
            else:
                if DIST_WRAPPER.rank == 0:
                    logger.info(
                        "[pipeline] main eval pending scan start task=%s run=%d seed=%d designs=%d marker_reason=%s",
                        task_name,
                        int(run_id),
                        int(run_seed),
                        int(len(pdb_names)),
                        str(marker_status.get("reason")),
                    )
                pending_names = _pending_pdb_names(
                    pdb_names,
                    task_eval_dir,
                    configs.eval.binder,
                    run_seed,
                )
                if DIST_WRAPPER.rank == 0:
                    logger.info(
                        "[pipeline] main eval pending scan end task=%s run=%d seed=%d pending=%d elapsed_s=%.2f",
                        task_name,
                        int(run_id),
                        int(run_seed),
                        int(len(pending_names)),
                        time.time() - pending_scan_start,
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

            if DIST_WRAPPER.rank == 0:
                authoritative_chain_payload = _resolve_authoritative_chain_payload_rank0(
                    task_eval_dir=task_eval_dir,
                    task_name=task_name,
                    struct_dir=struct_dir,
                    probe_names=pdb_names,
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
                if hb is not None:
                    _update_eval_heartbeat(
                        hb,
                        task_name=task_name,
                        task_eval_dir=task_eval_dir,
                        pdb_names=pdb_names,
                        eval_cfg=configs.eval.binder,
                        seed=run_seed,
                        step="chain_probe",
                        scan_complete=marker_scan_bypass,
                        marker_status=marker_status,
                        metrics={
                            "chain_probe_mode": authoritative_chain_payload.get(
                                "chain_probe_mode"
                            ),
                            "chain_probe_primary_sample": authoritative_chain_payload.get(
                                "chain_probe_primary_sample"
                            ),
                            "chain_probe_sanity_sample": authoritative_chain_payload.get(
                                "chain_probe_sanity_sample"
                            ),
                            "chain_probe_sanity_status": authoritative_chain_payload.get(
                                "chain_probe_sanity_status"
                            ),
                        },
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

            (
                pdb_dir,
                ready_pdb_names,
                local_cond_chains,
                local_binder_chains,
                rank_cache_stats,
            ) = _prepare_rank_cache(
                task_eval_dir=task_eval_dir,
                task_name=task_name,
                rank=int(DIST_WRAPPER.rank),
                owned_names=my_owned_names,
                struct_dir=struct_dir,
                condition_chains=cond_chains or None,
                binder_chains=binder_chains or None,
                timeout_s=stagein_timeout,
                poll_s=stagein_poll,
            )
            if hb is not None:
                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    step="pdb_cache",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
                    metrics={
                        "rank": int(DIST_WRAPPER.rank),
                        "pdb_cache_mode": rank_cache_stats.get("pdb_cache_mode"),
                        "pdb_reused_count": int(
                            rank_cache_stats.get("pdb_reused_count", 0)
                        ),
                        "pdb_converted_count": int(
                            rank_cache_stats.get("pdb_converted_count", 0)
                        ),
                    },
                )

            local_chain_payload = _chain_payload(local_cond_chains, local_binder_chains)
            if my_owned_names:
                if not _validate_chain_payload(
                    local_chain_payload, authoritative_chain_payload
                ):
                    raise RuntimeError(
                        f"[pipeline] Rank {int(DIST_WRAPPER.rank)} chain mismatch for task {task_name}"
                    )

            if sorted(ready_pdb_names) != sorted(my_owned_names):
                raise RuntimeError(
                    f"[pipeline] Rank {int(DIST_WRAPPER.rank)} cache readiness mismatch for task {task_name}"
                )

            _update_eval_heartbeat(
                hb,
                task_name=task_name,
                task_eval_dir=task_eval_dir,
                pdb_names=pdb_names,
                eval_cfg=configs.eval.binder,
                seed=run_seed,
                step="pre_run_task",
                scan_complete=marker_scan_bypass,
                marker_status=marker_status,
            )

            my_pdb_names = list(my_owned_names)

            if my_pdb_names:
                eval_cond_chains = _chain_ids_for_hint_compare(cond_chains)
                eval_binder_chains = _chain_ids_for_hint_compare(binder_chains)
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
                    "cond_chains": eval_cond_chains,
                    "binder_chains": eval_binder_chains,
                    "out_dir": task_eval_dir,
                    "orig_seqs": last_orig_seqs.get(task_name),
                    "pred_only": True,
                    "reuse_persisted_sequences": True,
                }
                eval_hb_interval = float(
                    os.environ.get("PXDESIGN_EVAL_HEARTBEAT_INTERVAL", "30") or 30
                )
                keepalive = _start_eval_heartbeat_keepalive(
                    hb,
                    interval_s=eval_hb_interval,
                    task_name=task_name,
                    task_eval_dir=task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    step="run_task",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
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
                    step="run_task_complete",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
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
                    "marker_status": marker_status,
                    "marker_scan_bypass": bool(marker_scan_bypass),
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
                marker_status = dict(meta.get("marker_status") or {})
                marker_scan_bypass = bool(meta.get("marker_scan_bypass", False))
                run_rel_dir = f"runs/run_{int(run_id):03d}"
                rw_task_eval_dir = _overlay_to_rw_path(task_eval_dir) or ""
                rw_struct_dir = _overlay_to_rw_path(struct_dir) or ""
                path_select_start = time.time()
                if marker_scan_bypass:
                    eval_evidence_ok = False
                    struct_evidence_ok = False
                    logger.info(
                        "[pipeline] aggregate evidence scans bypassed task=%s marker_mode=%s reason=%s",
                        task_name,
                        str(marker_status.get("mode")),
                        str(marker_status.get("reason")),
                    )
                else:
                    eval_evidence_ok = _local_eval_dir_has_expected_outputs(
                        rw_task_eval_dir,
                        pdb_names,
                        configs.eval.binder,
                        run_seed,
                    )
                    struct_evidence_ok = _local_struct_dir_has_expected_outputs(
                        rw_struct_dir,
                        pdb_names,
                    )
                aggregate_task_eval_dir, eval_source_reason = _select_rw_overlay_path(
                    task_eval_dir,
                    run_dir=run_rel_dir,
                    task_name=task_name,
                    evidence_ok=eval_evidence_ok,
                    marker_status=marker_status,
                    allow_marker_path=marker_scan_bypass,
                )
                aggregate_struct_dir, struct_source_reason = _select_rw_overlay_path(
                    struct_dir,
                    run_dir=run_rel_dir,
                    task_name=task_name,
                    evidence_ok=struct_evidence_ok,
                    marker_status=marker_status,
                    allow_marker_path=marker_scan_bypass,
                )
                logger.info(
                    "[pipeline] aggregate path selection prepared task=%s marker_used=%s marker_mode=%s elapsed_s=%.2f",
                    task_name,
                    str(bool(marker_scan_bypass)).lower(),
                    str(marker_status.get("mode") or ""),
                    time.time() - path_select_start,
                )
                agg_timeout = _clamp_env_int(
                    "PXDESIGN_AGG_READY_TIMEOUT_S", 1800, 60, 21600
                )
                agg_poll = _clamp_env_int(
                    "PXDESIGN_AGG_READY_POLL_S", 30, 5, 120
                )

                shard_wait_start = time.time()
                logger.info(
                    "[pipeline] shard manifest wait start task=%s marker_mode=%s timeout_s=%d poll_s=%d",
                    task_name,
                    str(marker_status.get("mode") or ""),
                    int(agg_timeout),
                    int(agg_poll),
                )
                ready_manifests = _wait_for_shards_ready(
                    task_eval_dir=aggregate_task_eval_dir,
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
                logger.info(
                    "[pipeline] shard manifest wait end task=%s manifests=%d elapsed_s=%.2f",
                    task_name,
                    int(len(ready_manifests)),
                    time.time() - shard_wait_start,
                )

                aggregate_inputs_start = time.time()
                logger.info(
                    "[pipeline] aggregate input build start task=%s marker_mode=%s",
                    task_name,
                    str(marker_status.get("mode") or ""),
                )
                aggregate_pdb_dir, aggregate_source_counts = _build_aggregate_inputs(
                    task_eval_dir=aggregate_task_eval_dir,
                    task_name=task_name,
                    run_id=run_id,
                    run_seed=run_seed,
                    world_size=int(DIST_WRAPPER.world_size),
                    all_pdb_names=pdb_names,
                    all_output_manifests=ready_manifests,
                    chain_payload=chain_payload,
                    struct_dir=aggregate_struct_dir,
                    attempt_token=attempt_token,
                )
                logger.info(
                    "[pipeline] aggregate input build end task=%s pdb_dir=%s elapsed_s=%.2f",
                    task_name,
                    aggregate_pdb_dir,
                    time.time() - aggregate_inputs_start,
                )
                rw_aggregate_pdb_dir = _overlay_to_rw_path(aggregate_pdb_dir) or ""
                pdb_evidence_ok = _local_pdb_cache_has_expected_outputs(
                    rw_aggregate_pdb_dir,
                    pdb_names,
                )
                aggregate_pdb_dir, pdb_source_reason = _select_rw_overlay_path(
                    aggregate_pdb_dir,
                    run_dir=run_rel_dir,
                    task_name=task_name,
                    evidence_ok=pdb_evidence_ok,
                    marker_status=marker_status,
                    allow_marker_path=False,
                )
                logger.info(
                    "[pipeline] aggregate path selection task=%s eval_dir=%s "
                    "struct_dir=%s pdb_dir=%s eval_source=%s struct_source=%s "
                    "pdb_source=%s marker_mode=%s",
                    task_name,
                    aggregate_task_eval_dir,
                    aggregate_struct_dir,
                    aggregate_pdb_dir,
                    eval_source_reason,
                    struct_source_reason,
                    pdb_source_reason,
                    str(marker_status.get("mode") or ""),
                )
                if hb is not None:
                    _update_eval_heartbeat(
                        hb,
                        task_name=task_name,
                        task_eval_dir=aggregate_task_eval_dir,
                        pdb_names=pdb_names,
                        eval_cfg=configs.eval.binder,
                        seed=run_seed,
                        step="aggregate_inputs",
                        scan_complete=marker_scan_bypass,
                        marker_status=marker_status,
                        metrics={
                            "aggregate_fallback_converted_count": int(
                                aggregate_source_counts.get(
                                    "aggregate_fallback_converted_count", 0
                                )
                            ),
                            "aggregate_owner_rank_cache_count": int(
                                aggregate_source_counts.get("owner_rank_cache", 0)
                            ),
                            "aggregate_shared_cache_count": int(
                                aggregate_source_counts.get("shared_cache", 0)
                            ),
                            "aggregate_other_rank_cache_count": int(
                                aggregate_source_counts.get("other_rank_cache", 0)
                            ),
                        },
                    )

                if not pdb_names:
                    continue

                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=aggregate_task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    step="pre_aggregate",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
                )

                os.environ["PXDESIGN_TASK_NAME"] = str(task_name)
                eval_hb_interval = float(
                    os.environ.get("PXDESIGN_EVAL_HEARTBEAT_INTERVAL", "30") or 30
                )
                keepalive = _start_eval_heartbeat_keepalive(
                    hb,
                    interval_s=eval_hb_interval,
                    task_name=task_name,
                    task_eval_dir=aggregate_task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    step="aggregate",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
                )
                aggregate_cond_chains = _chain_ids_for_hint_compare(
                    chain_payload.get("cond_chains", [])
                )
                aggregate_binder_chains = _chain_ids_for_hint_compare(
                    chain_payload.get("binder_chains", [])
                )
                aggregate_start = time.time()
                logger.info(
                    "[pipeline] aggregate_binder_eval start task=%s designs=%d marker_mode=%s eval_dir=%s pdb_dir=%s",
                    task_name,
                    int(len(pdb_names)),
                    str(marker_status.get("mode") or ""),
                    aggregate_task_eval_dir,
                    aggregate_pdb_dir,
                )
                try:
                    aggregate_binder_eval(
                        task_name=task_name,
                        eval_dir=aggregate_task_eval_dir,
                        pdb_dir=aggregate_pdb_dir,
                        pdb_names=pdb_names,
                        cond_chains=aggregate_cond_chains,
                        binder_chains=aggregate_binder_chains,
                        cfg=configs.eval.binder,
                        seed=run_seed,
                        analysis_workers=int(p.get("analysis_workers")),
                    )
                finally:
                    if keepalive is not None:
                        stop_event, thread = keepalive
                        stop_event.set()
                        thread.join(timeout=1.0)
                    logger.info(
                        "[pipeline] aggregate_binder_eval end task=%s elapsed_s=%.2f",
                        task_name,
                        time.time() - aggregate_start,
                    )

                _update_eval_heartbeat(
                    hb,
                    task_name=task_name,
                    task_eval_dir=aggregate_task_eval_dir,
                    pdb_names=pdb_names,
                    eval_cfg=configs.eval.binder,
                    seed=run_seed,
                    step="aggregate_complete",
                    scan_complete=marker_scan_bypass,
                    marker_status=marker_status,
                )

                csv_path = os.path.join(
                    aggregate_task_eval_dir,
                    "sample_level_output.csv",
                )
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
