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
        self.design_test_dl = get_inference_dataloader(configs=self.configs)

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
    def _inference(self, seed: int):
        num_data = len(self.dataset)
        orig_seqs = {}
        for batch in self.design_test_dl:
            data, atom_array, data_error_message = batch[0]
            try:
                if data_error_message:
                    logger.info(data_error_message)
                    continue
                sample = data["sample_name"]
                # Surface context to lower-level code (diffusion heartbeat / logs).
                os.environ["PXDESIGN_STAGE"] = "diffusion"
                os.environ["PXDESIGN_TASK_NAME"] = str(sample)
                os.environ["PXDESIGN_SEED"] = str(seed)
                if hasattr(self, "global_run"):
                    os.environ["PXDESIGN_GLOBAL_RUN"] = str(getattr(self, "global_run"))
                logger.info(
                    f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] {sample}: "
                    f"N_asym={data['N_asym'].item()}, N_token={data['N_token'].item()}, "
                    f"N_atom={data['N_atom'].item()}, N_msa={data['N_msa'].item()}"
                )
                if sample not in orig_seqs:
                    data["sequences"].pop(-1)  # generated binder chain
                    for seq_idx, seq in enumerate(data["sequences"]):
                        ent_k = list(seq.keys())[0]
                        label_asym_id = f"{chr(ord('A') + seq_idx)}0"
                        assert seq[ent_k]["count"] == 1
                        seq[ent_k]["label_asym_id"] = [label_asym_id]
                    orig_seqs[data["sample_name"]] = data["sequences"]

                if self.dumper.check_completion("", sample, seed):
                    self.print(f"Skip sample={sample}: already dumped.")
                    continue

                stream_dump = os.environ.get("PXDESIGN_STREAM_DUMP", "").lower() in {
                    "1",
                    "true",
                    "yes",
                }

                # Where outputs go for this task/seed
                dump_dir = self.dumper._get_dump_dir("", sample, seed)
                pred_dir = os.path.join(dump_dir, "predictions")
                os.makedirs(pred_dir, exist_ok=True)

                def _existing_indices(pred_dir: str, sample_name: str) -> set[int]:
                    out: set[int] = set()
                    for fp in glob(os.path.join(pred_dir, f"{sample_name}_sample_*.cif")):
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

                if stream_dump and expected_total > 0:
                    done = _existing_indices(pred_dir, sample)
                    missing = [i for i in range(expected_total) if i not in done]

                    if not missing:
                        # Ensure completion marker exists (job may have died after dumping)
                        self.dumper._mark_task_complete(dump_dir)
                        logger.info(
                            f"[Rank {DIST_WRAPPER.rank}] {sample} already has "
                            f"{len(done)}/{expected_total} CIFs. Marked complete."
                        )
                        continue

                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample} streaming dump enabled. "
                        f"Found {len(done)}/{expected_total} CIFs, generating {len(missing)} missing."
                    )

                    # Resume-aware progress reporting for heartbeat / logs
                    os.environ["PXDESIGN_EXPECTED_SAMPLES"] = str(expected_total)
                    os.environ["PXDESIGN_COMPLETED_BASE"] = str(len(done))

                    # Prepare atom_array annotations once (matches DataDumper._save_structure)
                    atom_array.set_annotation(
                        "b_factor", np.round(np.zeros(len(atom_array)).astype(float), 2)
                    )
                    if "occupancy" not in atom_array._annot:
                        atom_array.set_annotation(
                            "occupancy", np.round(np.ones(len(atom_array)), 2)
                        )

                    entity_poly_type = data["entity_poly_type"]

                    def _chunk_cb(chunk_coords: torch.Tensor, indices: list[int]) -> None:
                        # chunk_coords: [chunk, N_atom, 3]
                        for local_i, out_idx in enumerate(indices):
                            out_path = os.path.join(
                                pred_dir, f"{sample}_sample_{out_idx}.cif"
                            )
                            if os.path.exists(out_path):
                                continue
                            save_structure_cif(
                                atom_array=atom_array,
                                pred_coordinate=chunk_coords[local_i],
                                output_fpath=out_path,
                                entity_poly_type=entity_poly_type,
                                pdb_id=sample,
                            )

                    # Temporarily reduce N_sample to remaining work and stream results.
                    orig_n_sample = expected_total
                    self.configs.sample_diffusion.N_sample = int(len(missing))
                    diffusion_generator.set_stream_chunk_callback(
                        _chunk_cb,
                        sample_indices=missing,
                    )
                    try:
                        _ = self.predict(data)
                    finally:
                        diffusion_generator.clear_stream_chunk_callback()
                        self.configs.sample_diffusion.N_sample = orig_n_sample

                    done2 = _existing_indices(pred_dir, sample)
                    if len(done2) != expected_total:
                        raise RuntimeError(
                            f"Streaming dump incomplete for {sample}: have {len(done2)}/{expected_total} CIFs"
                        )

                    self.dumper._mark_task_complete(dump_dir)
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample} streaming dump complete. Saved to {dump_dir}"
                    )
                else:
                    pred = self.predict(data)
                    self.dumper.dump(
                        "",
                        sample,
                        seed,
                        pred_dict=pred,
                        atom_array=atom_array,
                        entity_poly_type=data["entity_poly_type"],
                    )
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample} succeeded. Saved to {self.dumper._get_dump_dir('', sample, seed)}"
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
    os.makedirs(configs.dump_dir, exist_ok=True)
    configs.input_json_path = process_input_file(
        configs.input_json_path, out_dir=configs.dump_dir
    )
    download_inference_cache(configs)

    # convert cif / pdb to bioassembly dict
    if DIST_WRAPPER.rank == 0:
        save_config(configs, os.path.join(configs.dump_dir, "config.yaml"))
        with open(configs.input_json_path, "r") as f:
            orig_inputs = json.load(f)
        for x in orig_inputs:
            convert_to_bioassembly_dict(x, configs.dump_dir)
        configs.input_json_path = os.path.join(configs.dump_dir, "input_tasks.json")
        with open(configs.input_json_path, "w") as f:
            json.dump(orig_inputs, f, indent=4)

    runner = InferenceRunner(configs)

    logger.info(f"Loading data from\n{configs.input_json_path}")
    if len(runner.dataset) == 0:
        logger.info("Nothing to infer. Bye!")
        return

    seeds = [derive_seed(time.time_ns())] if not configs.seeds else configs.seeds
    for seed in seeds:
        print(f"----------Infer with seed {seed}----------")
        seed_everything(seed=seed, deterministic=False)
        runner._inference(seed)


if __name__ == "__main__":
    main()
