# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WorldCache: training-free feature caching for transformer block speedup.

Algorithm per forward pass:
  1. Run the first `probe_depth` transformer blocks.
  2. Measure L2 drift vs the probe output cached from the previous timestep.
  3. If drift < threshold: skip remaining blocks and reconstruct output as
     probe_out + prev_residual  (OFA residual extrapolation).
  4. Otherwise: run all blocks, cache probe_out and residual = full_out - probe_out.

State resets automatically when the incoming timestep scalar increases
(= a new denoising sequence / new AR chunk has started).

For CFG inference (cond + uncond), two independent ping-pong buffers are
maintained: branch 0 = cond, branch 1 = uncond.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field


@dataclass
class WorldCacheState:
    # Indexed by branch: 0 = cond, 1 = uncond
    prev_probe: list = field(default_factory=lambda: [None, None])
    prev_residual: list = field(default_factory=lambda: [None, None])
    prev_t_scalar: list = field(default_factory=lambda: [float("inf"), float("inf")])
    total_steps: int = 0
    skipped_steps: int = 0

    def reset(self) -> None:
        self.prev_probe = [None, None]
        self.prev_residual = [None, None]
        self.prev_t_scalar = [float("inf"), float("inf")]

    def log_stats(self, logger) -> None:
        if self.total_steps > 0:
            logger.info(
                f"[WorldCache] steps={self.total_steps}  skipped={self.skipped_steps}  "
                f"hit_rate={100.0 * self.skipped_steps / self.total_steps:.1f}%",
                rank0_only=True,
            )


def _l2_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-token L2 distance between two (B, L, D) tensors."""
    return (a.float() - b.float()).norm(dim=-1).mean().item()


def run_blocks_worldcache(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    kwargs_blocks: dict,
    state: WorldCacheState,
    is_uncond: bool,
    t_scalar: float,
    probe_depth: int = 4,
    drift_threshold: float = 0.10,
) -> torch.Tensor:
    """Run transformer blocks with WorldCache caching.

    Args:
        x: Input hidden states (B, L, D).
        blocks: Full list of transformer blocks.
        kwargs_blocks: Shared keyword arguments forwarded to every block.
        state: Mutable WorldCacheState shared across timesteps.
        is_uncond: True for the unconditional CFG branch.
        t_scalar: Current denoising timestep as a plain float.
        probe_depth: Number of blocks always executed (probe head).
        drift_threshold: Skip remaining blocks when drift < this value.

    Returns:
        Output hidden states (B, L, D).
    """
    branch = 1 if is_uncond else 0

    # Detect new denoising sequence: timestep jumped back up.
    if t_scalar > state.prev_t_scalar[branch]:
        state.prev_probe[branch] = None
        state.prev_residual[branch] = None

    state.prev_t_scalar[branch] = t_scalar
    state.total_steps += 1

    # Always run probe blocks.
    x_probe = x
    for block in blocks[:probe_depth]:
        x_probe = block(x_probe, **kwargs_blocks)

    # Try to skip remaining blocks when cache is warm and drift is low.
    if state.prev_probe[branch] is not None:
        drift = _l2_drift(x_probe, state.prev_probe[branch])
        if drift < drift_threshold:
            state.skipped_steps += 1
            state.prev_probe[branch] = x_probe.detach()
            # OFA residual: approximate full output from cached residual.
            return x_probe + state.prev_residual[branch]

    # Cache miss: run all remaining blocks.
    x_full = x_probe
    for block in blocks[probe_depth:]:
        x_full = block(x_full, **kwargs_blocks)

    state.prev_probe[branch] = x_probe.detach()
    state.prev_residual[branch] = (x_full - x_probe).detach()

    return x_full
