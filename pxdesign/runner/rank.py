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

"""Ranking-only entrypoint for PXDesign pipeline v2.

This command re-runs the *derived* ranking/selection stage without re-running
(diffusion, eval). It assumes a v2 dump directory produced by `pxdesign pipeline`.

It reads:
- per-run eval CSVs under: <dump_dir>/runs/run_*/eval/<task>/sample_level_output.csv
- orig_seqs cached under: <dump_dir>/runs/run_XXX/diffusion/orig_seqs.json

Then it writes:
- final summaries under: <dump_dir>/runs/run_XXX/final/
- versioned results snapshots under: <dump_dir>/results/, results_v2/, ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from protenix.utils.distributed import DIST_WRAPPER

from pxdesign.runner.helpers import save_top_designs
from pxdesign.runner.pipeline import allocate_results_dir, parse_pipeline_args
from pxdesign.utils.heartbeat import HeartbeatReporter
from pxdesign.utils.infer import get_configs
from pxdesign.utils.pipeline import check_tool_weights


def _detect_latest_run_id(dump_dir: str) -> int:
    runs_dir = Path(dump_dir) / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"No runs directory found: {runs_dir}")

    ids: list[int] = []
    for p in runs_dir.glob("run_*"):
        name = p.name
        if not name.startswith("run_"):
            continue
        suffix = name.replace("run_", "", 1)
        if suffix.isdigit():
            ids.append(int(suffix))
    if not ids:
        raise FileNotFoundError(f"No run_* folders found under: {runs_dir}")
    return max(ids)


def main(argv=None) -> None:
    # Parse rank-specific args first.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--run_id", type=int, default=None)
    rank_args, remaining = parser.parse_known_args(argv)

    dump_dir = str(rank_args.dump_dir)
    run_id = int(rank_args.run_id) if rank_args.run_id is not None else _detect_latest_run_id(dump_dir)

    # Parse pipeline-level knobs (min_total_return, weights, etc.) from remaining args.
    pipeline_args_ns, remaining_cfg_args = parse_pipeline_args(remaining)
    p = vars(pipeline_args_ns)

    # Build configs from the same parser used by pipeline.
    # `input_json_path` is required by configs; for ranking-only we point it at the
    # cached pipeline_input.json.
    pipeline_input = os.path.join(dump_dir, "pipeline_input.json")
    cfg_argv = [
        "--dump_dir",
        dump_dir,
        "--input_json_path",
        pipeline_input,
        *list(remaining_cfg_args),
    ]
    configs = get_configs(cfg_argv)

    # Keep eval tool dtype / deepspeed flags consistent with pipeline.
    for tool_name in ["ptx_mini", "ptx"]:
        configs["eval"]["binder"]["tools"][tool_name].update(
            {
                "dtype": configs.dtype,
                "use_deepspeed_evo_attention": configs.use_deepspeed_evo_attention,
            }
        )

    # Sanity-check tool weights (needed if extended mode triggers PTX reruns).
    check_tool_weights()

    # Ranking should only run on rank 0 if launched in distributed mode.
    if int(getattr(DIST_WRAPPER, "rank", 0)) != 0:
        return

    os.environ.setdefault("PXDESIGN_STATUS_DIR", str(dump_dir))
    os.environ["PXDESIGN_STAGE"] = "ranking"
    os.environ["PXDESIGN_GLOBAL_RUN"] = str(int(run_id))

    hb = HeartbeatReporter.from_env()
    if hb is not None:
        hb.update(
            produced_total=0,
            expected_total=1,
            extra={"stage_transition": "ranking_only", "run_id": int(run_id)},
            force=True,
        )

    # Load cached orig_seqs (optional but recommended for extended mode).
    orig_seqs_path = os.path.join(
        dump_dir, "runs", f"run_{int(run_id):03d}", "diffusion", "orig_seqs.json"
    )
    orig_seqs: dict = {}
    if os.path.exists(orig_seqs_path):
        try:
            d = json.loads(Path(orig_seqs_path).read_text())
            orig_seqs = d.get("orig_seqs", {}) if isinstance(d, dict) else {}
        except Exception:
            orig_seqs = {}

    final_dir = os.path.join(dump_dir, "runs", f"run_{int(run_id):03d}", "final")
    os.makedirs(final_dir, exist_ok=True)

    results_dir = allocate_results_dir(dump_dir)

    # We do not attempt to infer use_template here; it only affects the UI fig.
    save_top_designs(
        p,
        configs,
        orig_seqs,
        use_template=False,
        final_dir=final_dir,
        results_dir=results_dir,
    )

    if hb is not None:
        hb.complete(extra={"message": "ranking-only complete"})


if __name__ == "__main__":
    main()
