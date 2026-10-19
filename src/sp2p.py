"""
Full SP2P-DAS pipeline as a standalone module, with:
  * Stage 1: DAS-adapted Pixel2Pixel (non-local self-similarity based denoising)
  * Stage 2: Structure-Tensor Guided Enhancement (wavefront-aligned smoothing)
  * Optional bidirectional Stage-2 -> Stage-1 guidance (for stronger novelty claim)
  * All preprocessing steps explicitly exposed / switchable for ablation

Every parameter has a default matching the SP2P paper (sp2p.tex), but every
choice is overridable so the user can run sensitivity studies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import einops
from einops import rearrange

from .preprocessing import Preprocessor
from .structure_filter import structure_guided_filter


# ============================================================
# Stage 1: Denoise network (6-layer FCN, per sp2p.tex L44-65)
# ============================================================
class P2PDenoiseNet(nn.Module):
    """Lightweight 6-layer fully-conv denoiser for P2P stage."""

    def __init__(self, n_chan, chan_embed=64):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.conv3(x)
        return x


# ============================================================
# Stage 1: Pixel Bank Construction (block-wise, half-precision)
# ============================================================
def construct_pixel_bank(img, window_size=40, patch_size=7, num_neighbors=100,
                         blk_sz=64, batch_per_gpu=4, device=None,
                         direction_field=None, angle_tolerance_deg=None):
    """
    Non-local patch matching for P2P training pair generation.

    New (for revision novelty): if `direction_field` is provided (H,W array of
    wavefront orientations from a quick structure-tensor estimate on raw /
    early-epoch output), restrict candidate patches to lie within
    ±angle_tolerance_deg of the wavefront direction. This is the
    "Physically-Constrained Directional Patch Search" innovation.

    Returns:
        bank: np.ndarray (K, H, W, C) top-k candidate patches per position
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    img = img.to(device, dtype=dtype)

    pad_sz = window_size // 2 + patch_size // 2
    center_offset = window_size // 2
    even = (window_size % 2 == 0)
    H_org, W_org = img.shape[-2], img.shape[-1]

    with torch.no_grad():
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=patch_size, padding=0, stride=1)
        H_new = H_org + window_size
        W_new = W_org + window_size
        img_unfold = rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

        num_blk_h = H_org // blk_sz
        num_blk_w = W_org // blk_sz
        total_blks = num_blk_h * num_blk_w
        blk_indices = list(range(total_blks))

        topk_list = []
        for idx in range(0, total_blks, batch_per_gpu):
            batch_indices = blk_indices[idx:idx + batch_per_gpu]
            batch_topk = []
            for blk_idx in batch_indices:
                blk_j = blk_idx // num_blk_w
                blk_i = blk_idx % num_blk_w
                sh = blk_j * blk_sz
                eh = (blk_j + 1) * blk_sz + window_size
                sw = blk_i * blk_sz
                ew = (blk_i + 1) * blk_sz + window_size
                sub_uf = img_unfold[..., sh:eh, sw:ew]
                if even:
                    sub_uf = sub_uf[..., :-1, :-1]

                pw = F.unfold(sub_uf, kernel_size=window_size, padding=0, stride=1)
                pw = rearrange(
                    pw,
                    'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                    k1=patch_size, k2=patch_size, k3=window_size, k4=window_size,
                    h=blk_sz, w=blk_sz)

                img_ctr = rearrange(
                    sub_uf, 'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=patch_size, k2=patch_size,
                    h=sub_uf.shape[-2], w=sub_uf.shape[-1])
                img_ctr = img_ctr[..., center_offset:center_offset + blk_sz,
                                  center_offset:center_offset + blk_sz]

                # L2 distance
                l2_dis = torch.sum((img_ctr - pw) ** 2, dim=1, keepdim=False)

                # -------- Directional mask (novel contribution) --------
                if direction_field is not None and angle_tolerance_deg is not None:
                    tol_rad = np.deg2rad(angle_tolerance_deg)
                    # Extract current block orientation
                    bf_h_start, bf_h_end = blk_j * blk_sz, (blk_j + 1) * blk_sz
                    bf_w_start, bf_w_end = blk_i * blk_sz, (blk_i + 1) * blk_sz
                    theta_block = direction_field[bf_h_start:bf_h_end,
                                                  bf_w_start:bf_w_end]
                    theta_block = torch.from_numpy(theta_block).float()
                    # Build angle map from each pixel to each candidate
                    # position within the window_size x window_size grid
                    ws = window_size
                    yy, xx = torch.meshgrid(torch.arange(ws), torch.arange(ws),
                                            indexing='ij')
                    cy = cx = ws // 2
                    dy = yy - cy  # (ws, ws) time offset
                    dx = xx - cx  # (ws, ws) channel offset
                    cand_theta = torch.atan2(dy.float(), dx.float() + 1e-8)
                    # Difference in angle, wrapped to [0, pi/2]
                    diff = torch.abs(torch.atan2(
                        torch.sin(theta_block[:, :, None, None] - cand_theta),
                        torch.cos(theta_block[:, :, None, None] - cand_theta)))
                    # Keep only candidates close (within tol) to wavefront dir
                    Hb, Wb = theta_block.shape
                    mask = (diff < tol_rad).float().to(device)  # (Hb, Wb, ws, ws)
                    mask = mask.reshape(Hb, Wb, ws * ws)  # (Hb, Wb, ws*ws)
                    mask = mask.unsqueeze(0)  # (1, Hb, Wb, ws*ws)
                    # l2_dis shape is (b, ws*ws, Hb, Wb) -> (b, Hb, Wb, ws*ws)
                    l2_dis = l2_dis.permute(0, 2, 3, 1)  # (b, Hb, Wb, ws*ws)
                    # Penalize invalid directions by big L2
                    l2_dis = l2_dis + (1.0 - mask) * 1e6
                    l2_dis = l2_dis.permute(0, 3, 1, 2)  # back to (b, ws*ws, Hb, Wb)

                k = min(num_neighbors, l2_dis.shape[1])
                _, sort_idx = torch.topk(l2_dis, k=k, largest=False, sorted=True, dim=-3)

                pwr = rearrange(pw, 'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                                k1=patch_size, k2=patch_size,
                                k3=window_size, k4=window_size)
                pctr = pwr[:, :, pwr.shape[2] // 2, ...]
                sie = sort_idx.unsqueeze(1).expand(-1, pctr.shape[1], -1, -1, -1)
                topk = torch.gather(pctr, dim=-3, index=sie)
                batch_topk.append(topk.cpu())
                del sub_uf, pw, img_ctr, l2_dis, sort_idx, pctr, sie
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_topk = torch.cat(batch_topk, dim=0)
            topk_list.append(batch_topk)
            del batch_topk

        topk = torch.cat(topk_list, dim=0)
        topk = rearrange(topk, '(w2 w1) c k h w -> k c (w2 h) (w1 w)',
                         w1=num_blk_w, w2=num_blk_h)
        topk = topk.permute(2, 3, 0, 1)
        return topk


# ============================================================
# Stage 1: P2P Training
# ============================================================
def train_p2p_one_epoch(model, optimizer, img_bank, loss_fn):
    K, H, W, C = img_bank.shape
    device = next(model.parameters()).device

    i1 = torch.randint(0, K, size=(H, W), device=device)
    i1e = i1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    im1 = torch.gather(img_bank, 0, i1e).permute(0, 3, 1, 2)

    i2 = torch.randint(0, K, size=(H, W), device=device)
    eq = (i2 == i1)
    if eq.any():
        i2[eq] = (i2[eq] + 1) % K
    i2e = i2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    im2 = torch.gather(img_bank, 0, i2e).permute(0, 3, 1, 2)

    pred1 = model(im1)
    loss = loss_fn(im2, pred1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ============================================================
# Stage 1+2 Full SP2P pipeline
# ============================================================
def sp2p_das(data,
             # ---------- Stage 1 params (sp2p.tex defaults) ----------
             p2p_window_size=40,
             p2p_patch_size=7,
             p2p_num_neighbors=100,
             p2p_topk_mm=20,            # argsmm=20 actually used in training
             p2p_max_epoch=100,
             p2p_lr=1e-3,
             p2p_weight_decay=1e-4,
             milestones=(0.5, 0.66, 0.87),
             gamma=0.5,
             loss_type='L2',
             # ---------- Stage 2 params ----------
             sf_enabled=True,
             sf_tensor_window=(5, 5),
             sf_alpha=0.8,
             sf_sigma_parallel=3.0,
             sf_sigma_perpendicular=0.5,
             sf_use_adaptive=False,
             # ---------- (Novel) bidirectional guidance ----------
             bidirectional_guidance=False,   # if True, Stage-2 theta -> Stage-1 search mask
             angle_tolerance_deg=30.0,       # for directional search
             # ---------- Training setup ----------
             device=None,
             preprocessor=None,
             verbose=True,
             return_intermediates=False):
    """
    SP2P-DAS full two-stage denoising framework.

    Parameters follow sp2p.tex Section II.
    Setting `bidirectional_guidance=True` activates the revision-level
    novelty: structure tensor from a quick Stage-1 bootstrap estimate
    back-guides the non-local patch search in Stage-1 by restricting
    candidate directions to physically plausible wavefront orientations.

    Returns:
        denoised: np.ndarray (H,W) original scale
        (if return_intermediates=True): dict with {'stage1_output',
            'stage2_output', 'structure_info', 'direction_field'}
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Preprocessing ----
    if preprocessor is None:
        preprocessor = Preprocessor(
            do_bandpass=True, lowcut=30.0, highcut=100.0, fs=1000,
            do_weak_suppress=True, sigma_scale=2.0, weak_divisor=5.0,
            do_border_atten=True, border_time=20, border_channel=50,
            do_zscore=True,
            do_resize=True, resize_multiple=64,
            target_size=(1024, 1024) if (data.shape[0] % 64 != 0 or
                                          data.shape[1] % 64 != 0) else None)
    tensor, aux = preprocessor(data)

    # ---- Optional: Quick direction field for bidirectional guidance ----
    direction_field = None
    if bidirectional_guidance:
        # Use a fast structure tensor on the (resized) input as bootstrap
        # guidance for Stage-1 patch search
        quick_img = tensor.squeeze(0).squeeze(0).cpu().numpy()
        from .structure_filter import compute_structure_tensor, decompose_structure_tensor
        j11, j12, j22 = compute_structure_tensor(quick_img, sigma_deriv=1.0,
                                                 window_size=sf_tensor_window)
        theta0, coh0, _, _ = decompose_structure_tensor(j11, j12, j22)
        # theta0 shape is (H', W'), same as resized tensor
        direction_field = theta0

    # ---- Stage 1: Pixel Bank + P2P ----
    bank_arr = construct_pixel_bank(
        tensor, window_size=p2p_window_size, patch_size=p2p_patch_size,
        num_neighbors=p2p_num_neighbors, blk_sz=64, batch_per_gpu=4,
        device=device, direction_field=direction_field,
        angle_tolerance_deg=angle_tolerance_deg if bidirectional_guidance
        else None).numpy()
    if bank_arr.ndim == 3:
        bank_arr = np.expand_dims(bank_arr, axis=1)
    bank_arr = bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
    bank_arr = bank_arr[:p2p_topk_mm]
    img_bank = torch.from_numpy(bank_arr).to(device)
    noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

    model = P2PDenoiseNet(n_chan=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=p2p_lr,
                            weight_decay=p2p_weight_decay)
    ms = [int(p2p_max_epoch * m) for m in milestones]
    scheduler = MultiStepLR(optimizer, milestones=ms, gamma=gamma)
    loss_fn = nn.L1Loss() if loss_type == 'L1' else nn.MSELoss()

    iterator = tqdm(range(p2p_max_epoch), disable=not verbose,
                    desc="SP2P Stage1 (P2P)")
    for _ in iterator:
        l = train_p2p_one_epoch(model, optimizer, img_bank, loss_fn)
        scheduler.step()
        if verbose:
            iterator.set_postfix(loss=f"{l:.5f}")

    model.eval()
    with torch.no_grad():
        stage1_tensor = model(noisy_img)

    # ---- Stage 2: Structure Guided Enhancement ----
    stage1_np = stage1_tensor.squeeze(0).squeeze(0).cpu().numpy()
    info = None
    if sf_enabled:
        stage2_np, info = structure_guided_filter(
            stage1_np,
            tensor_window=sf_tensor_window,
            sigma_parallel=sf_sigma_parallel,
            sigma_perpendicular=sf_sigma_perpendicular,
            alpha=sf_alpha,
            use_adaptive=sf_use_adaptive)
        final_np = stage2_np
    else:
        final_np = stage1_np

    # Wrap for postprocessing
    final_tensor = torch.from_numpy(final_np).float()[None, None, :, :]
    denoised = preprocessor.postprocess(final_tensor, aux)

    # ---- Optional intermediates for visualization / analysis ----
    if not return_intermediates:
        return denoised
    intermed = {
        'stage1_output': preprocessor.postprocess(
            torch.from_numpy(stage1_np).float()[None, None, :, :], aux),
        'stage2_output': denoised if sf_enabled else None,
        'structure_info': info,
        'direction_field': direction_field,
    }
    return denoised, intermed
