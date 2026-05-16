"""Monkey-patch DPO primitives to be CUDA-safe. Import this BEFORE importing diff_proxy."""
import sys, importlib.util
from pathlib import Path
import torch

_ROOT = Path(__file__).resolve().parents[3]
_DPO = _ROOT / "writeup" / "archive" / "submissions" / "dpo" / "ablation_v2_steps.py"
spec = importlib.util.spec_from_file_location("_dpo_orig", str(_DPO))
dpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dpo)

# Save originals
_orig_rudy = dpo._rudy_congestion
_orig_density = dpo._grid_density

def _rudy_congestion_cuda(positions, net_data, port_base, gamma,
                          cell_x_min, cell_x_max, cell_y_min, cell_y_max,
                          grid_h_routes, grid_v_routes, grid_rows, grid_cols):
    """Same as DPO _rudy_congestion but h_cong/v_cong allocated on positions.device."""
    device = positions.device
    if not isinstance(positions, torch.Tensor) or positions.numel() == 0:
        return torch.tensor(0.0, device=device)
    all_pos = torch.cat([positions, port_base], dim=0)
    pin_macro_idx = net_data.pin_macro_idx
    pin_offsets = net_data.pin_offsets
    mask = net_data.mask
    pin_pos = all_pos[pin_macro_idx] + pin_offsets
    mask_float = mask.float()
    mask_inv = (1.0 - mask_float)
    x = pin_pos[..., 0] * mask_float + (-1e9) * mask_inv
    y = pin_pos[..., 1] * mask_float + (-1e9) * mask_inv
    x_neg = pin_pos[..., 0] * mask_float + 1e9 * mask_inv
    y_neg = pin_pos[..., 1] * mask_float + 1e9 * mask_inv
    bbox_x_max = gamma * torch.logsumexp(x / gamma, dim=1)
    bbox_y_max = gamma * torch.logsumexp(y / gamma, dim=1)
    bbox_x_min = -gamma * torch.logsumexp(-x_neg / gamma, dim=1)
    bbox_y_min = -gamma * torch.logsumexp(-y_neg / gamma, dim=1)
    bbox_w = torch.clamp(bbox_x_max - bbox_x_min, min=1e-3)
    bbox_h = torch.clamp(bbox_y_max - bbox_y_min, min=1e-3)
    num_nets = pin_macro_idx.shape[0]
    w = net_data.weights
    h_coeff = w / (bbox_w * grid_h_routes)
    v_coeff = w / (bbox_h * grid_v_routes)
    h_cong = torch.zeros(grid_rows, grid_cols, device=device)
    v_cong = torch.zeros(grid_rows, grid_cols, device=device)
    batch_size = 2000
    for b_start in range(0, num_nets, batch_size):
        b_end = min(b_start + batch_size, num_nets)
        b = slice(b_start, b_end)
        x_ol = torch.clamp(
            torch.min(bbox_x_max[b].unsqueeze(1), cell_x_max.unsqueeze(0))
            - torch.max(bbox_x_min[b].unsqueeze(1), cell_x_min.unsqueeze(0)),
            min=0,
        )
        y_ol = torch.clamp(
            torch.min(bbox_y_max[b].unsqueeze(1), cell_y_max.unsqueeze(0))
            - torch.max(bbox_y_min[b].unsqueeze(1), cell_y_min.unsqueeze(0)),
            min=0,
        )
        h_cong += (h_coeff[b].unsqueeze(1) * x_ol).T @ y_ol
        v_cong += x_ol.T @ (v_coeff[b].unsqueeze(1) * y_ol)
    abu_5 = max(1, int(h_cong.numel() * 0.05))
    total = (h_cong + v_cong).flatten()
    top, _ = torch.topk(total, abu_5)
    return top.mean()

# Apply patch
dpo._rudy_congestion = _rudy_congestion_cuda
sys.modules['_dpo_orig'] = dpo

# Re-export for diff_proxy import-time bind
import sys
sys.modules['dpo_cuda'] = sys.modules['_dpo_orig']
print(f"[dpo_cuda_patch] _rudy_congestion patched to use positions.device", flush=True)
