""" Muon optimizer for UNet
Original code copied from:
    https://github.com/toothacher17/Megatron-LM/blob/moonshot/distributedmuon-impl/megatron/core/optimizer/muon.py
Then we apply the following changes:
    * Remove everything related to distributed training (as we run on single GPU)
    * Reshape all weights to 2D, before Newton-Schulz, and then reshape back.
    (ref: https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py)
    * Batch Newton-Schulz iterations across all params of the same shape.
    * Fuse momentum buffer updates via torch._foreach_* ops (2 kernel launches vs 306).
    * @torch.compile on the NS kernel for fused GPU execution.
"""
from typing import List, Tuple
import torch
import math
from collections import defaultdict


@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Accepts G of shape (m, n) or (B, m, n) — batch dimension is handled transparently.
    The quintic coefficients are tuned to maximise slope at zero.
    """
    assert G.ndim == 3, "NS5 expects batched (B, m, n) input"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def adjust_lr_wd_for_muon(lr, matched_adamw_rms, param_shape):
    A, B = param_shape
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    return lr * adjusted_ratio


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Optimisations vs the reference implementation:
    - Momentum buffer updates are fused with torch._foreach_mul_ / _foreach_add_
      (2 GPU kernel launches instead of 2×n_params sequential launches).
    - Newton-Schulz is batched across all params sharing the same 2D shape
      (O(n_shapes) batched matmuls instead of O(n_params) sequential calls).
    - @torch.compile on the NS kernel fuses the inner matmul/elementwise ops.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    """

    def __init__(self,
                 param_groups,
                 lr=2e-2,
                 weight_decay=0.1,
                 matched_adamw_rms=0.2,
                 momentum=0.95,
                 nesterov=True,
                 ns_steps=5,
                 adamw_betas=(0.95, 0.95),
                 adamw_eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        matched_adamw_rms=matched_adamw_rms,
                        momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps)
        super().__init__(param_groups, defaults)

    def step(self):
        # ---- Muon groups ----
        for group in self.param_groups:
            if not group.get('use_muon', False):
                continue
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            matched_adamw_rms = group["matched_adamw_rms"]
            nesterov = group["nesterov"]
            params = group["params"]

            # Initialise momentum buffers on first call
            for p in params:
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p.grad)

            # Fused momentum update: buf = buf * momentum + grad  (2 kernel launches)
            bufs = [self.state[p]["momentum_buffer"] for p in params]
            grads = [p.grad for p in params]
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)

            # Build per-param ns_inputs and group by flattened 2D shape
            # shape_key = (m, n) with m <= n
            shape_to_entries: dict = defaultdict(list)
            for p, buf, g in zip(params, bufs, grads):
                ns_input = g.add(buf, alpha=momentum) if nesterov else buf
                ns_flat = ns_input.reshape(ns_input.shape[0], -1)
                transposed = ns_flat.shape[0] > ns_flat.shape[1]
                if transposed:
                    ns_flat = ns_flat.mT.contiguous()
                shape_key = (ns_flat.shape[0], ns_flat.shape[1])
                shape_to_entries[shape_key].append((p, ns_flat, transposed, g.shape))

            # Batched NS5 per shape group, then fused parameter update
            for shape_key, entries in shape_to_entries.items():
                stacked = torch.stack([e[1] for e in entries], dim=0)  # (B, m, n)
                updates_raw = zeropower_via_newtonschulz5(stacked, steps=ns_steps)

                adjusted_lr = adjust_lr_wd_for_muon(lr, matched_adamw_rms, shape_key)

                p_datas = [e[0].data for e in entries]
                # Build update tensors reshaped back to original param shape
                update_list = []
                for i, (p, ns_flat, transposed, orig_shape) in enumerate(entries):
                    u = updates_raw[i]
                    if transposed:
                        u = u.mT.contiguous()
                    update_list.append(u.view(orig_shape))

                # Cautious weight decay + param update (fused within group)
                if weight_decay != 0.0:
                    for p_data, upd in zip(p_datas, update_list):
                        wd_mask = (upd * p_data).gt(0).to(p_data.dtype)
                        p_data.mul_(1 - lr * weight_decay * wd_mask)

                torch._foreach_add_(p_datas, update_list, alpha=-adjusted_lr)

        # ---- AdamW groups ----
        for group in self.param_groups:
            if group.get('use_muon', False):
                continue
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            step = group['step']
            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if len(state) == 0:
                    state['adamw_exp_avg'] = torch.zeros_like(g)
                    state['adamw_exp_avg_sq'] = torch.zeros_like(g)
                buf1 = state['adamw_exp_avg']
                buf2 = state['adamw_exp_avg_sq']
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                if weight_decay != 0.0:
                    wd_mask = (g * p.data).gt(0).to(p.data.dtype)
                    p.data.mul_(1 - lr * weight_decay * wd_mask)
                p.data.add_(g, alpha=-lr / scale)
