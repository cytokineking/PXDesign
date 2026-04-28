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

import glob
import json
import logging
import os
import shutil

import pandas as pd

from pxdbench.tasks.base import BaseTask
from pxdbench.tools.protmpnn.main_mpnn import get_gt_sequence
from pxdbench.tools.protmpnn.mpnn_predictor import MPNNPredictor
from pxdbench.utils import save_eval_results

from .registry import register_task

logger = logging.getLogger(__name__)

_CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@register_task("binder")
class BinderTask(BaseTask):
    def __init__(self, input_data, cfg, device_id: int, seed: int):
        """
        Initialize a BinderTask instance.

        Args:
            input_data (dict): Task input parameters including PDB paths and chain specifications.
            cfg (dict): Configuration dictionary with task settings.
            device_id (int): GPU device ID (-1 for CPU).
            seed (int): Random seed for reproducibility.

        Validates:
            - Exactly one binder chain is specified (multiple binder chains not supported).
        """
        self.task_type = "binder"
        self.task_name = input_data.get("name", "binder")
        assert "cond_chains" in input_data
        assert "binder_chains" in input_data
        self.cond_chains = input_data["cond_chains"]
        self.binder_chains = input_data["binder_chains"]
        self.pdb_name_to_binder_seq_list = input_data.get(
            "pdb_name_to_binder_seq_list", None
        )
        if input_data.get("orig_seqs_json", None) is not None:
            with open(input_data["orig_seqs_json"], "r") as f:
                self.orig_seqs = json.load(f)
        elif input_data.get("orig_seqs", None) is not None:
            self.orig_seqs = input_data["orig_seqs"]
        else:
            self.orig_seqs = None
        self.reuse_persisted_sequences = bool(
            input_data.get("reuse_persisted_sequences", False)
        )
        self._sequence_overwrite_keys: set[tuple[str, int]] = set()

        # Default values
        self.use_binder_seq_list = cfg.get("use_binder_seq_list", False)
        self.eval_diversity = cfg.get("eval_diversity", False)
        self.eval_binder_monomer = cfg.get("eval_binder_monomer", True)
        self.eval_complex = cfg.get("eval_complex", True)
        self.eval_protenix_mini = cfg.get("eval_protenix_mini", True)
        self.eval_protenix = cfg.get("eval_protenix", False)

        # Check values
        assert (
            len(self.binder_chains) == 1
        ), f"Get {len(self.binder_chains)} binder chains, but only 1 is allowed."

        super().__init__(input_data, cfg, device_id, seed)

    def prepare_data_from_seq_list(self):
        datas = []
        for name in self.pdb_names:
            binder_seq_list = self.pdb_name_to_binder_seq_list[name]
            for i, seq in enumerate(binder_seq_list):
                data = {"name": name, "seq_idx": i, "sequence": seq}
                datas.append(data)
        return datas

    def _seq_cache_path(self, name: str, seq_idx: int) -> str:
        return os.path.join(self.out_dir, "seqs", f"{name}_seq{int(seq_idx)}.txt")

    @staticmethod
    def _read_cached_sequence(path: str) -> tuple[str, str | None, str | None]:
        if not os.path.exists(path):
            return "missing", None, None
        if not os.path.isfile(path):
            return "corrupt", None, "not a regular file"
        try:
            with open(path, "r") as f:
                seq = f.read().strip()
        except OSError as exc:
            return "corrupt", None, f"read failed: {exc}"
        if not seq:
            return "corrupt", None, "empty sequence"
        if seq != seq.upper():
            return "corrupt", None, "sequence is not uppercase"
        invalid = sorted({aa for aa in seq if aa not in _CANONICAL_AA})
        if invalid:
            return "corrupt", None, f"invalid residues: {''.join(invalid)}"
        return "valid", seq, None

    def _downstream_artifact_paths(self, name: str, seq_idx: int) -> list[str]:
        design_name = f"{name}_seq{int(seq_idx)}"
        af2_dir = os.path.join(self.out_dir, "af2_pred")
        paths = []
        for pattern in (
            os.path.join(af2_dir, f"{design_name}_model*.json"),
            os.path.join(af2_dir, f"{design_name}_model*.pdb"),
            os.path.join(af2_dir, f"{design_name}_MONOMER_ONLY_model*.json"),
            os.path.join(af2_dir, f"{design_name}_MONOMER_ONLY_model*.pdb"),
        ):
            paths.extend(glob.glob(pattern))
        for dirname in ("ptx_mini_pred", "ptx_pred"):
            pred_dir = os.path.join(self.out_dir, dirname, design_name)
            if os.path.exists(pred_dir):
                paths.append(pred_dir)
        return sorted(set(paths))

    @staticmethod
    def _remove_visible_artifact(path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to invalidate stale downstream artifact {path}: {exc}"
            ) from exc
        if os.path.exists(path):
            raise RuntimeError(
                f"Failed to invalidate stale downstream artifact {path}: "
                "artifact remains visible after removal"
            )

    def _invalidate_downstream_outputs(self, name: str, seq_indices) -> int:
        invalidated = 0
        for seq_idx in seq_indices:
            for path in self._downstream_artifact_paths(name, int(seq_idx)):
                self._remove_visible_artifact(path)
                invalidated += 1
        return invalidated

    def _run_mpnn(self, pdb_names: list[str], verbose=True):
        mpnn_predictor = MPNNPredictor(
            self.cfg.tools.mpnn,
            device_id=self.device_id,
            verbose=verbose,
            seed=self.seed,
        )
        return mpnn_predictor.design_binder(
            self.pdb_dir,
            pdb_names,
            self.num_seqs,
            binder_chains=self.binder_chains,
            cond_chains=self.cond_chains,
        )

    def _design_sequence_with_cache(self, verbose=True):
        num_seqs = int(self.num_seqs)
        cached_results: dict[tuple[str, int], dict] = {}
        generate_names: list[str] = []
        overwrite_keys: set[tuple[str, int]] = set()
        corrupt_count = 0
        invalidated_count = 0

        for name in self.pdb_names:
            status_rows = []
            for seq_idx in range(num_seqs):
                path = self._seq_cache_path(name, seq_idx)
                status, seq, reason = self._read_cached_sequence(path)
                status_rows.append((seq_idx, status, seq, reason, path))

            if num_seqs == 1:
                seq_idx, status, seq, reason, path = status_rows[0]
                key = (name, seq_idx)
                if status == "valid":
                    cached_results[key] = {
                        "name": name,
                        "seq_idx": seq_idx,
                        "sequence": seq,
                    }
                    continue
                if status == "corrupt":
                    logger.warning(
                        "Recovering corrupt MPNN sequence cache task=%s name=%s "
                        "seq_idx=%d path=%s reason=%s",
                        self.task_name,
                        name,
                        seq_idx,
                        path,
                        reason,
                    )
                    invalidated_count += self._invalidate_downstream_outputs(
                        name, [seq_idx]
                    )
                    overwrite_keys.add(key)
                    corrupt_count += 1
                else:
                    invalidated_count += self._invalidate_downstream_outputs(
                        name, [seq_idx]
                    )
                generate_names.append(name)
                continue

            statuses = [row[1] for row in status_rows]
            if all(status == "valid" for status in statuses):
                for seq_idx, _, seq, _, _ in status_rows:
                    cached_results[(name, seq_idx)] = {
                        "name": name,
                        "seq_idx": seq_idx,
                        "sequence": seq,
                    }
                continue
            if all(status == "missing" for status in statuses):
                invalidated_count += self._invalidate_downstream_outputs(
                    name, range(num_seqs)
                )
                generate_names.append(name)
                continue
            if any(status == "corrupt" for status in statuses):
                reasons = [
                    f"seq{seq_idx}:{reason}"
                    for seq_idx, status, _, reason, _ in status_rows
                    if status == "corrupt"
                ]
                logger.warning(
                    "Recovering corrupt multi-sequence MPNN cache task=%s name=%s "
                    "num_seqs=%d reasons=%s",
                    self.task_name,
                    name,
                    num_seqs,
                    ",".join(reasons),
                )
                seq_indices = list(range(num_seqs))
                invalidated_count += self._invalidate_downstream_outputs(
                    name, seq_indices
                )
                for seq_idx in seq_indices:
                    overwrite_keys.add((name, seq_idx))
                corrupt_count += sum(status == "corrupt" for status in statuses)
                generate_names.append(name)
                continue

            details = ", ".join(
                f"seq{seq_idx}:{status}"
                for seq_idx, status, _, _, _ in status_rows
            )
            raise RuntimeError(
                "Unsupported partial MPNN sequence cache for "
                f"task={self.task_name} name={name} num_seqs={num_seqs}: {details}"
            )

        generated_results = (
            self._run_mpnn(generate_names, verbose=verbose) if generate_names else []
        )
        result_by_key = dict(cached_results)
        for item in generated_results:
            key = (str(item["name"]), int(item["seq_idx"]))
            if key in result_by_key:
                raise RuntimeError(
                    f"Duplicate MPNN sequence result for task={self.task_name} "
                    f"name={key[0]} seq_idx={key[1]}"
                )
            result_by_key[key] = item

        ordered_results = []
        for name in self.pdb_names:
            for seq_idx in range(num_seqs):
                key = (name, seq_idx)
                if key not in result_by_key:
                    raise RuntimeError(
                        f"Missing MPNN sequence result for task={self.task_name} "
                        f"name={name} seq_idx={seq_idx}"
                    )
                ordered_results.append(result_by_key[key])

        self._sequence_overwrite_keys = overwrite_keys
        logger.info(
            "mpnn sequence cache task=%s cached_name_seq=%d generated_names=%d "
            "num_seqs=%d corrupt_name_seq=%d invalidated_artifacts=%d seq_dir=%s",
            self.task_name,
            len(cached_results),
            len(generate_names),
            num_seqs,
            corrupt_count,
            invalidated_count,
            os.path.join(self.out_dir, "seqs"),
        )
        return ordered_results

    def design_sequence(self, verbose=True):
        """
        Generates binder sequences based on task configuration.

        Supports three modes:
        1. Use pre-provided sequence lists (self.use_binder_seq_list)
        2. Use ground truth sequences from PDB files (self.use_gt_seq)
        3. De novo design using MPNN (default)

        Args:
            verbose (bool, optional): Whether to print detailed progress. Defaults to True.

        Returns:
            list[dict]: List of design results with keys "name", "seq_idx", and "sequence".
        """
        self._sequence_overwrite_keys = set()
        if self.use_binder_seq_list:
            results = self.prepare_data_from_seq_list()
        elif self.use_gt_seq:
            results = get_gt_sequence(
                self.pdb_dir, self.pdb_names, self.binder_chains[0]
            )
        elif self.reuse_persisted_sequences:
            results = self._design_sequence_with_cache(verbose=verbose)
        else:
            results = self._run_mpnn(self.pdb_names, verbose=verbose)
        return results

    def run(self):
        """
        Executes the complete binder design evaluation workflow.

        Workflow steps:
        1. Designs sequences via design_sequence()
        2. Runs structure predictions (AF2 complex/monomer, Protenix) based on config
        3. Calculates secondary structure and diversity metrics
        4. Saves sample-level results to CSV and summary metrics to JSON

        Returns:
            dict: Dictionary with task metadata and output file paths.
        """
        results = self.design_sequence()
        self.check_results(results)
        self.persist_sequences(results, overwrite_keys=self._sequence_overwrite_keys)
        binder_chain = self.binder_chains[0]

        af2_pred_path = os.path.join(self.out_dir, "af2_pred")
        if self.eval_complex:
            self.af2_complex_predict(results, af2_pred_path)

        if self.eval_binder_monomer:
            self.af2_monomer_predict(results, af2_pred_path)

        if self.eval_protenix_mini:
            self.protenix_predict(results, orig_seqs=self.orig_seqs)

        if self.eval_protenix:
            self.protenix_predict(results, orig_seqs=self.orig_seqs, is_large=True)

        if self.pred_only:
            return {
                "task": self.task_type,
                "name": self.task_name,
                "pred_only": True,
                "out_dir": self.out_dir,
            }

        self.cal_secondary(results, binder_chain)
        div = self.cal_diversity()
        sample_df = pd.DataFrame(results)
        sample_df = sample_df.sort_values(by=["name", "seq_idx"])
        self.compute_success_rate(self.cfg.filters, sample_df)
        summary_dict = {"task": self.task_type, "name": self.task_name}
        summary_dict.update(
            self.summary_from_df(sample_df, other_metrics={"diversity": div})
        )
        sample_save_path, summary_save_path = save_eval_results(
            sample_df, summary_dict, self.out_dir, self.sample_fn, self.summary_fn
        )
        print(
            f"Eval done! Results are saved in {sample_save_path} and {summary_save_path}"
        )
        return {
            "task": self.task_type,
            "name": self.task_name,
            "sample_save_path": sample_save_path,
            "summary_save_path": summary_save_path,
        }

    def check_results(self, results):
        """
        Validates design results for consistency and correctness.

        Checks:
        1. No duplicate entries (by structure name + sequence index)
        2. Correct number of sequences per structure (when not using pre-provided lists)

        Args:
            results (list[dict]): List of design results from design_sequence()

        Raises:
            ValueError: If duplicates are found or sequence count is incorrect.
        """
        result_names = [
            result["name"] + f"_seq{result['seq_idx']}" for result in results
        ]
        if len(result_names) != len(set(result_names)):
            raise ValueError(f"Found duplicate names in results: {result_names}.")
        if self.use_binder_seq_list or self.use_gt_seq:
            pass
        elif len(result_names) != len(self.pdb_names) * self.num_seqs:
            raise ValueError(
                f"Found {len(result_names)} results, but {len(self.pdb_names)} pdb_names, each with {self.num_seqs} seqs are provided."
            )
        return
