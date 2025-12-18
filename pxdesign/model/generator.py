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

from typing import Any, Callable, Optional, Sequence, Union

import os
import time

import numpy as np
import torch
from protenix.model.utils import centre_random_augmentation
from protenix.utils.logger import get_logger

from pxdesign.utils.heartbeat import HeartbeatReporter

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Streaming / incremental dump hooks (set by runner)
# -----------------------------------------------------------------------------

_STREAM_CHUNK_CALLBACK: Optional[Callable[[torch.Tensor, list[int]], None]] = None
_STREAM_SAMPLE_INDICES: Optional[list[int]] = None


def set_stream_chunk_callback(
    callback: Callable[[torch.Tensor, list[int]], None],
    *,
    sample_indices: Sequence[int] | None = None,
) -> None:
    """
    Enable streaming mode for diffusion sampling.

    When enabled, `sample_diffusion` will call `callback(chunk_coords, indices)`
    after each diffusion chunk finishes and will NOT accumulate all coordinates
    in memory.

    Args:
        callback: function called once per chunk.
        sample_indices: optional mapping from generated sample order -> output index.
            If provided, must have length equal to the N_sample passed to sampling.
    """
    global _STREAM_CHUNK_CALLBACK, _STREAM_SAMPLE_INDICES
    _STREAM_CHUNK_CALLBACK = callback
    _STREAM_SAMPLE_INDICES = list(sample_indices) if sample_indices is not None else None


def clear_stream_chunk_callback() -> None:
    """Disable streaming mode."""
    global _STREAM_CHUNK_CALLBACK, _STREAM_SAMPLE_INDICES
    _STREAM_CHUNK_CALLBACK = None
    _STREAM_SAMPLE_INDICES = None


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    # step_scale_eta: float = 1.5,
    step_scale_eta: Union[float, dict] = {"type": "const", "min": 1.5, "max": 1.5},
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    n_step = max(int(len(noise_schedule) - 1), 0)
    logger.info("sampling eta schedule: %s", step_scale_eta)

    # Optional heartbeat + progress (best-effort, throttled)
    hb = HeartbeatReporter.from_env()
    t0 = time.time()
    hb_expected_total = int(os.environ.get("PXDESIGN_EXPECTED_SAMPLES", str(N_sample)) or N_sample)
    hb_completed_base = int(os.environ.get("PXDESIGN_COMPLETED_BASE", "0") or 0)
    hb_expected_total = max(hb_expected_total, 1)
    hb_completed_base = max(hb_completed_base, 0)
    if hb is not None:
        hb.start(expected_total=hb_expected_total)
        # If resuming, reflect already-produced designs immediately.
        if hb_completed_base:
            hb.update(
                produced_total=hb_completed_base,
                expected_total=hb_expected_total,
                extra={"resume": {"completed_base": hb_completed_base}},
                force=True,
            )

    progress_interval_s = float(os.environ.get("PXDESIGN_PROGRESS_INTERVAL", "30") or 30)
    step_heartbeat_interval_s = float(
        os.environ.get("PXDESIGN_STEP_HEARTBEAT_INTERVAL", "30") or 30
    )
    last_progress_log_ts = 0.0
    last_step_heartbeat_ts = 0.0
    stream_cb = _STREAM_CHUNK_CALLBACK
    stream_indices = _STREAM_SAMPLE_INDICES
    stream_enabled = stream_cb is not None
    if stream_indices is not None and len(stream_indices) != int(N_sample):
        raise ValueError(
            f"stream sample_indices length ({len(stream_indices)}) "
            f"must equal N_sample ({int(N_sample)})"
        )

    # Streaming requires chunking; if user disabled chunking, pick a small default.
    if stream_enabled and diffusion_chunk_size is None:
        diffusion_chunk_size = int(os.environ.get("PXDESIGN_STREAM_CHUNK_SIZE", "10") or 10)

    def _chunk_sample_diffusion(
        chunk_n_sample: int,
        inplace_safe: bool,
        *,
        chunk_index: int,
        num_chunks: int,
        completed_before_chunk: int,
    ):
        nonlocal last_progress_log_ts, last_step_heartbeat_ts
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training
        T = len(noise_schedule)
        for step_t, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # "Liveness" heartbeat during long diffusion chunks (no design outputs
            # are written until the full chunk completes).
            now = time.time()
            if (
                hb is not None
                and step_heartbeat_interval_s > 0
                and (now - last_step_heartbeat_ts) >= step_heartbeat_interval_s
            ):
                extra = {
                    "diffusion": {
                        "chunk_index": int(chunk_index) + 1,
                        "num_chunks": int(num_chunks),
                        "step": int(step_t) + 1,
                        "num_steps": int(T - 1),
                        "chunk_samples": int(chunk_n_sample),
                        "completed_samples": int(completed_before_chunk),
                        "total_samples": int(N_sample),
                    }
                }
                hb.update(
                    produced_total=int(hb_completed_base + completed_before_chunk),
                    expected_total=int(hb_expected_total),
                    extra=extra,
                )
                last_step_heartbeat_ts = now

            if (
                progress_interval_s > 0
                and (now - last_progress_log_ts) >= progress_interval_s
            ):
                # Note: produced_total does NOT advance until the chunk finishes.
                logger.info(
                    "[diffusion] chunk %d/%d | step %d/%d | completed %d/%d samples",
                    int(chunk_index) + 1,
                    int(num_chunks),
                    int(step_t) + 1,
                    int(T - 1),
                    int(hb_completed_base + completed_before_chunk),
                    int(hb_expected_total),
                )
                last_progress_log_ts = now

            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
            )

            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            if isinstance(step_scale_eta, float):
                eta = step_scale_eta
            elif step_scale_eta["type"] == "const":
                assert step_scale_eta["min"] == step_scale_eta["max"]
                eta = step_scale_eta["min"]
            else:
                eta_min, eta_max = step_scale_eta["min"], step_scale_eta["max"]
                if step_scale_eta["type"] == "linear":
                    eta = eta_min + (eta_max - eta_min) * (step_t / T)
                elif step_scale_eta["type"] == "poly":
                    eta = eta_min + (eta_max - eta_min) * (step_t / T) ** 2
                elif step_scale_eta["type"] == "cos":
                    eta = eta_min + 0.5 * (eta_max - eta_min) * (
                        1 - np.cos(np.pi * step_t / T)
                    )
                elif step_scale_eta["type"] == "piecewise":
                    eta = eta_min if step_t / T < 0.5 else eta_max
                elif step_scale_eta["type"] == "piecewise_65":
                    eta = eta_min if step_t / T < 0.65 else eta_max
                elif step_scale_eta["type"] == "piecewise_70":
                    eta = eta_min if step_t / T < 0.70 else eta_max
                else:
                    raise ValueError("Unsupported eta schedule!")
            x_l = x_noisy + eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(
            int(N_sample),
            inplace_safe=inplace_safe,
            chunk_index=0,
            num_chunks=1,
            completed_before_chunk=0,
        )
        if stream_enabled:
            indices = (
                stream_indices
                if stream_indices is not None
                else list(range(0, int(N_sample)))
            )
            assert stream_cb is not None
            stream_cb(x_l, indices)
            # Return an empty tensor to avoid holding all coordinates in memory.
            x_l = torch.empty((*batch_shape, 0, N_atom, 3), device=device, dtype=dtype)
        if hb is not None:
            hb.update(
                produced_total=int(hb_completed_base + N_sample),
                expected_total=int(hb_expected_total),
                extra={"diffusion": {"num_steps": int(n_step)}},
                force=True,
            )
    else:
        logger.info("diffusion_chunk_size: %s", diffusion_chunk_size)
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        completed = 0
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                int(chunk_n_sample),
                inplace_safe=inplace_safe,
                chunk_index=int(i),
                num_chunks=int(no_chunks),
                completed_before_chunk=int(completed),
            )
            if stream_enabled:
                assert stream_cb is not None
                start = int(completed)
                end = int(completed + chunk_n_sample)
                indices = (
                    stream_indices[start:end]
                    if stream_indices is not None
                    else list(range(start, end))
                )
                stream_cb(chunk_x_l, indices)
            else:
                x_l.append(chunk_x_l)
            completed += int(chunk_n_sample)

            # Chunk-complete progress is the only "ground-truth" progress in terms
            # of finished samples, because outputs are written after sampling.
            if hb is not None:
                hb.update(
                    produced_total=int(hb_completed_base + completed),
                    expected_total=int(hb_expected_total),
                    extra={
                        "diffusion": {
                            "chunk_index": int(i) + 1,
                            "num_chunks": int(no_chunks),
                            "completed_samples": int(hb_completed_base + completed),
                            "total_samples": int(hb_expected_total),
                            "num_steps": int(n_step),
                        }
                    },
                )

            # Human-friendly progress line (throttled)
            now = time.time()
            if (
                progress_interval_s > 0
                and (
                    (now - last_progress_log_ts) >= progress_interval_s
                    or completed == int(N_sample)
                    or i == 0
                )
            ):
                elapsed = max(now - t0, 1e-6)
                rate = float(completed) / elapsed
                eta_s = (float(N_sample - completed) / rate) if rate > 0 else None
                logger.info(
                    "[diffusion] %d/%d samples (%.1f%%) | chunk %d/%d | rate %.2f samples/min | eta %s",
                    int(hb_completed_base + completed),
                    int(hb_expected_total),
                    100.0
                    * (float(hb_completed_base + completed) / max(float(hb_expected_total), 1.0)),
                    int(i) + 1,
                    int(no_chunks),
                    rate * 60.0,
                    f"{eta_s:.0f}s" if isinstance(eta_s, (int, float)) else "?",
                )
                last_progress_log_ts = now

        if stream_enabled:
            # Return an empty tensor to avoid holding all coordinates in memory.
            x_l = torch.empty((*batch_shape, 0, N_atom, 3), device=device, dtype=dtype)
        else:
            x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l
