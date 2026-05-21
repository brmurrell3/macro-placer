"""DREAMPlace's Nesterov + Barzilai-Borwein line-search optimizer.

Extracted from `submit_deps/dreamplace_install/dreamplace/
NesterovAcceleratedGradientOptimizer.py` (step_bb path, which is the
default in DP). Pure PyTorch; no C++ ops or DP database dependencies.

Used by E116 to give the challenge-proxy descent the same optimizer
the top-2 leaderboard team uses.

Reference: Lu et al., "ePlace-MS: Electrostatics-Based Placement for
Mixed-Size Circuits", IEEE TCAD 2015. DP follows their algorithm 2.

Key features vs Adam:
1. Look-ahead update: v_{k+1} = u_{k+1} + ((a_k - 1) / a_{k+1}) * (u_{k+1} - u_k)
2. Step size from Barzilai-Borwein quotient: ||s_k|| / ||y_k||
   with `s_k = v_k - v_{k-1}` and `y_k = g_k - g_{k-1}`.
3. The momentum coefficient (a_k - 1) / a_{k+1} grows toward 1.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.optim.optimizer import Optimizer


class DreamPlaceNesterov(Optimizer):
    """Nesterov + BB line search, matching DREAMPlace step_bb."""

    def __init__(
        self,
        params,
        lr: float,
        obj_and_grad_fn: Callable[[torch.Tensor], tuple],
        constraint_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            u_k=[], v_k=[], obj_k=[], a_k=[], alpha_k=[],
            v_k_1=[], obj_k_1=[],
            v_kp1=[None],
            obj_eval_count=0,
        )
        super().__init__(params, defaults)
        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        if len(self.param_groups) != 1:
            raise ValueError("Only single-tensor parameter groups supported")

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group["params"]):
                if p.grad is None and not group["u_k"]:
                    # First eval: trigger forward+backward to populate p.grad
                    pass

                # First call init: u_k = p; v_k = p; eval to populate g_k, obj_k.
                if not group["u_k"]:
                    group["u_k"].append(p.data.clone())
                    group["v_k"].append(p)
                u_k = group["u_k"][i]
                v_k = group["v_k"][i]

                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group["obj_k"]:
                    group["obj_k"].append(None)
                group["obj_k"][i] = obj_k.data.clone()

                if not group["a_k"]:
                    group["a_k"].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group["v_k_1"].append(
                        torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                    )
                    group["v_k_1"][i].data.copy_(group["v_k"][i] - group["lr"] * g_k)
                a_k = group["a_k"][i]
                v_k_1 = group["v_k_1"][i]

                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group["obj_k_1"]:
                    group["obj_k_1"].append(None)
                group["obj_k_1"][i] = obj_k_1.data.clone()

                if group["v_kp1"][i] is None:
                    group["v_kp1"][i] = torch.autograd.Variable(
                        torch.zeros_like(v_k), requires_grad=True
                    )
                v_kp1 = group["v_kp1"][i]
                if not group["alpha_k"]:
                    init_alpha = (v_k - v_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)
                    group["alpha_k"].append(init_alpha)
                alpha_k = group["alpha_k"][i]

                # Momentum scalar
                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1

                with torch.no_grad():
                    s_k = (v_k - v_k_1).reshape(-1)
                    y_k = (g_k - g_k_1).reshape(-1)
                    y_norm = y_k.norm(p=2)
                    if y_norm > 1e-10:
                        syk_dot = s_k.dot(y_k)
                        bb_short = (syk_dot / y_k.dot(y_k)).data
                        lip_step = (s_k.norm(p=2) / y_norm).data
                    else:
                        bb_short = alpha_k
                        lip_step = alpha_k
                    step_size = bb_short if bb_short > 0 else torch.minimum(lip_step, alpha_k)

                # One step
                u_kp1 = v_k - step_size * g_k
                v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                if constraint_fn is not None:
                    constraint_fn(v_kp1)
                group["obj_eval_count"] += 1

                # Roll state
                v_k_1.data.copy_(v_k.data)
                alpha_k.data.copy_(step_size.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)

        return loss
