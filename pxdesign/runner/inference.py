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
import time
import traceback
from glob import glob
from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from protenix.config import save_config
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.seed import seed_everything
from protenix.utils.torch_utils import to_device

from pxdesign.data.infer_data_pipeline import InferenceDataset, get_inference_dataloader
from pxdesign.model.pxdesign import ProtenixDesign
from pxdesign.runner.dumper import DataDumper
from pxdesign.runner.dumper import save_structure_cif
from pxdesign.model import generator as diffusion_generator
from pxdesign.utils.infer import (
    configure_runtime_env,
    convert_to_bioassembly_dict,
    derive_seed,
    download_inference_cache,
    get_configs,
)
from pxdesign.utils.inputs import process_input_file

logger = logging.getLogger(__name__)


class InferenceRunner(object):
    def __init__(self, configs: Any) -> None:
        self.configs = configs
        self.init_env()
        self.init_basics()
        self.init_model()
        self.load_checkpoint()
        self.init_dumper()
        self.init_data()

    def init_env(self) -> None:
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            + f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            self.device = torch.device("cuda:{}".format(DIST_WRAPPER.local_rank))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        if DIST_WRAPPER.world_size > 1:
            dist.init_process_group(backend="nccl")

        configure_runtime_env(
            use_fast_ln=self.configs.use_fast_ln,
            use_deepspeed_evo=self.configs.use_deepspeed_evo_attention,
        )
        logging.info("Finished init ENV.")

    def init_basics(self) -> None:
        self.dump_dir = self.configs.dump_dir
        self.error_dir = os.path.join(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        # Default heartbeat location (can be overridden externally).
        # NOTE: Use configs.dump_dir (root output dir) rather than self.dump_dir,
        # because DesignPipeline rebinds self.dump_dir to per-global-run subdirs.
        os.environ.setdefault("PXDESIGN_STATUS_DIR", str(self.configs.dump_dir))

    def init_model(self) -> None:
        self.model = ProtenixDesign(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        checkpoint_path = os.path.join(
            self.configs.load_checkpoint_dir, f"{self.configs.model_name}.pt"
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Given checkpoint path does not exist [{checkpoint_path}]"
            )
        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(checkpoint_path, self.device)

        sample_key = [k for k in checkpoint["model"].keys()][0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print(f"Finish loading checkpoint.")

    def init_dumper(self):
        self.dumper = DataDumper(base_dir=self.dump_dir)

    def init_data(self):
        self.print(f"Input JSON: {self.configs.input_json_path}")
        self.dataset = InferenceDataset(
            input_json_path=self.configs.input_json_path,
            use_msa=self.configs.use_msa,
        )
        self.design_test_dl = get_inference_dataloader(
            configs=self.configs,
            distributed_tasks=bool(getattr(self.configs, "distributed_tasks", True)),
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[self.configs.dtype]

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if torch.cuda.is_available()
            else nullcontext()
        )

        data = to_device(data, self.device)
        with enable_amp:
            prediction = self.model(
                input_feature_dict=data["input_feature_dict"],
                mode="inference",
            )
        return prediction

    @torch.no_grad()
    def _inference(self, run_seed: int):
        num_data = len(self.dataset)
        orig_seqs = {}
        # v2 run context (set by pipeline; default run_000)
        run_id = int(getattr(self, "run_id", 0) or 0)
        run_dir = str(
            getattr(
                self,
                "run_dir",
                os.path.join(self.configs.dump_dir, "runs", f"run_{run_id:03d}"),
            )
        )
        active_tasks = getattr(self, "active_tasks", None)
        active_tasks = set(active_tasks) if active_tasks is not None else None

        # Surface context to lower-level code (diffusion heartbeat / logs).
        os.environ["PXDESIGN_STAGE"] = "diffusion"
        os.environ["PXDESIGN_SEED"] = str(run_seed)
        os.environ["PXDESIGN_GLOBAL_RUN"] = str(run_id)
        for batch in self.design_test_dl:
            data, atom_array, data_error_message = batch[0]
            try:
                if data_error_message:
                    logger.info(data_error_message)
                    continue
                sample = str(data["sample_name"])
                if active_tasks is not None and sample not in active_tasks:
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] Skip inactive task={sample}"
                    )
                    continue

                os.environ["PXDESIGN_TASK_NAME"] = sample
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample}: "
                    f"N_asym={data['N_asym'].item()}, N_token={data['N_token'].item()}, "
                    f"N_atom={data['N_atom'].item()}, N_msa={data['N_msa'].item()}"
                )
                if sample not in orig_seqs:
                    # Copy to avoid mutating the batch object.
                    seqs = json.loads(json.dumps(data.get("sequences", [])))
                    if seqs:
                        seqs.pop(-1)  # generated binder chain
                    for seq_idx, seq in enumerate(seqs):
                        ent_k = list(seq.keys())[0]
                        label_asym_id = f"{chr(ord('A') + seq_idx)}0"
                        assert seq[ent_k]["count"] == 1
                        seq[ent_k]["label_asym_id"] = [label_asym_id]
                    orig_seqs[sample] = seqs

                stream_dump = os.environ.get("PXDESIGN_STREAM_DUMP", "").lower() in {
                    "1",
                    "true",
                    "yes",
                }

                # v2 diffusion output directory for this run/task:
                #   <dump_dir>/runs/run_XXX/diffusion/structures/<task_name>/*.cif
                struct_dir = os.path.join(
                    run_dir, "diffusion", "structures", str(sample)
                )
                os.makedirs(struct_dir, exist_ok=True)

                def _existing_indices(struct_dir: str, sample_name: str) -> set[int]:
                    out: set[int] = set()
                    for fp in glob(
                        os.path.join(struct_dir, f"{sample_name}_sample_*.cif")
                    ):
                        base = os.path.basename(fp)
                        # "<sample>_sample_<idx>.cif"
                        parts = base.rsplit("_sample_", 1)
                        if len(parts) != 2:
                            continue
                        idx_str = parts[1].removesuffix(".cif")
                        if idx_str.isdigit():
                            out.add(int(idx_str))
                    return out

                expected_total = int(
                    getattr(self.configs.sample_diffusion, "N_sample", 0) or 0
                )
                if expected_total <= 0:
                    logger.info(f"[Rank {DIST_WRAPPER.rank}] {sample}: N_sample=0, skip.")
                    continue

                world_size = max(int(DIST_WRAPPER.world_size), 1)
                rank = int(DIST_WRAPPER.rank)

                # Compute existing + missing design IDs (global view).
                done_all = _existing_indices(struct_dir, sample)
                done_all = {i for i in done_all if 0 <= i < expected_total}
                missing_all = [i for i in range(expected_total) if i not in done_all]

                # Partition by design_id ownership (world-size agnostic).
                my_missing = [i for i in missing_all if (i % world_size) == rank]

                # Heartbeat accounting: expected/progress is per-rank ownership.
                # This makes rank-0 aggregation meaningful (sums to global totals).
                if expected_total - 1 >= rank:
                    owned_total = ((expected_total - 1 - rank) // world_size) + 1
                else:
                    owned_total = 0
                owned_done = sum(1 for i in done_all if (i % world_size) == rank)

                os.environ["PXDESIGN_EXPECTED_SAMPLES"] = str(int(owned_total))
                os.environ["PXDESIGN_COMPLETED_BASE"] = str(int(owned_done))

                if not my_missing:
                    logger.info(
                        f"[Rank {rank}] {sample}: "
                        f"owned_done={owned_done}/{owned_total} (global_done={len(done_all)}/{expected_total}). "
                        "Nothing to generate."
                    )
                    continue

                logger.info(
                    f"[Rank {rank}] {sample}: "
                    f"global_done={len(done_all)}/{expected_total}, "
                    f"rank_owned_done={owned_done}/{owned_total}, "
                    f"generating_missing={len(my_missing)}"
                )

                # Prepare atom_array annotations once (matches DataDumper._save_structure)
                atom_array.set_annotation(
                    "b_factor", np.round(np.zeros(len(atom_array)).astype(float), 2)
                )
                if "occupancy" not in atom_array._annot:
                    atom_array.set_annotation(
                        "occupancy", np.round(np.ones(len(atom_array)), 2)
                    )

                entity_poly_type = data["entity_poly_type"]

                def _out_path(out_idx: int) -> str:
                    return os.path.join(
                        struct_dir, f"{sample}_sample_{int(out_idx):06d}.cif"
                    )

                def _chunk_cb(chunk_coords: torch.Tensor, indices: list[int]) -> None:
                    # chunk_coords: [chunk, N_atom, 3]
                    for local_i, out_idx in enumerate(indices):
                        out_path = _out_path(int(out_idx))
                        if os.path.exists(out_path):
                            continue
                        save_structure_cif(
                            atom_array=atom_array,
                            pred_coordinate=chunk_coords[local_i],
                            output_fpath=out_path,
                            entity_poly_type=entity_poly_type,
                            pdb_id=sample,
                        )

                # Temporarily reduce N_sample to this rank's missing work.
                orig_n_sample = int(expected_total)
                self.configs.sample_diffusion.N_sample = int(len(my_missing))
                try:
                    if stream_dump:
                        diffusion_generator.set_stream_chunk_callback(
                            _chunk_cb,
                            sample_indices=my_missing,
                        )
                        _ = self.predict(data)
                    else:
                        pred = self.predict(data)
                        coords = pred.get("coordinate", None)
                        if coords is None:
                            raise RuntimeError("Model prediction missing 'coordinate'.")
                        for local_i, out_idx in enumerate(my_missing):
                            save_structure_cif(
                                atom_array=atom_array,
                                pred_coordinate=coords[local_i],
                                output_fpath=_out_path(int(out_idx)),
                                entity_poly_type=entity_poly_type,
                                pdb_id=sample,
                            )
                finally:
                    if stream_dump:
                        diffusion_generator.clear_stream_chunk_callback()
                    self.configs.sample_diffusion.N_sample = orig_n_sample

                # Verify that this rank's owned IDs are fully present.
                done2 = _existing_indices(struct_dir, sample)
                done2 = {i for i in done2 if 0 <= i < expected_total}
                owned_done2 = sum(1 for i in done2 if (i % world_size) == rank)
                if owned_done2 != int(owned_total):
                    raise RuntimeError(
                        f"v2 streaming dump incomplete for rank-owned IDs: "
                        f"task={sample} rank={rank} owned_done={owned_done2}/{owned_total} "
                        f"(global_done={len(done2)}/{expected_total})"
                    )
                logger.info(
                    f"[Rank {rank}] {sample} diffusion complete for owned IDs. "
                    f"owned_done={owned_done2}/{owned_total} (global_done={len(done2)}/{expected_total})"
                )
            except Exception as e:
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank}] {sample} {e}:\n{traceback.format_exc()}"
                )
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
        return orig_seqs

    def print(self, msg: str):
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def local_print(self, msg: str):
        msg = f"[Rank {DIST_WRAPPER.local_rank}] {msg}"
        logging.info(msg)


def main(argv=None):
    configs = get_configs(argv)
    os.environ.setdefault("PXDESIGN_STATUS_DIR", str(configs.dump_dir))
    os.environ["PXDESIGN_STAGE"] = "startup"
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = process_input_file(
        configs.input_json_path, out_dir=configs.dump_dir
    )
    download_inference_cache(configs)

    # convert cif / pdb to bioassembly dict
    input_tasks_path = os.path.join(configs.dump_dir, "input_tasks.json")
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            convert_to_bioassembly_dict(x, configs.dump_dir)
        tmp = input_tasks_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(orig_inputs, f, indent=4)
        os.replace(tmp, input_tasks_path)
    else:
        # Best-effort wait for rank-0 to finish writing (no dist barrier yet).
        while not os.path.exists(input_tasks_path):
            time.sleep(0.2)

    configs.input_json_path = input_tasks_path

    # v2: replicate tasks across ranks; partition design_id instead.
    setattr(configs, "distributed_tasks", False)

    runner = InferenceRunner(configs)

    logger.info(f"Loading data from\n{configs.input_json_path}")
    if len(runner.dataset) == 0:
        logger.info("Nothing to infer. Bye!")
        return

    seeds = [int(time.time_ns() % (2**31 - 1))] if not configs.seeds else list(configs.seeds)
    for run_id, run_seed in enumerate(seeds):
        print(f"----------Infer run {run_id} (run_seed={run_seed})----------")

        # v2 run context
        runner.run_id = int(run_id)
        runner.run_dir = os.path.join(
            configs.dump_dir, "runs", f"run_{int(run_id):03d}"
        )

        os.environ["PXDESIGN_STAGE"] = "diffusion"
        os.environ["PXDESIGN_GLOBAL_RUN"] = str(int(run_id))
        os.environ["PXDESIGN_SEED"] = str(int(run_seed))

        # Per-rank RNG seed derived from run_seed
        rank_seed = int(derive_seed(int(run_seed), int(DIST_WRAPPER.rank), digits=9))
        seed_everything(seed=rank_seed, deterministic=True)
        runner._inference(int(run_seed))

        if DIST_WRAPPER.world_size > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
