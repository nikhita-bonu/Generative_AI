# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:28:10 2025

@author: mkausar
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP crash on Windows

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BRATS Diffusion Anomaly Detection – Full Single File (CPU-only)

Implements:
- 4-channel BRATS dataset (T1, T1ce, T2, FLAIR) + seg-based slice labels
- UNCONDITIONAL DDPM (UNet + time embedding)
- Time-conditioned classifier C(x_t, t) – healthy vs tumor
- Classifier-guided deterministic translation toward healthy class (y=0)
- Anomaly map from SSIM/L2 (structural-weighted L2 on FLAIR)
- Evaluation: slice-wise Dice (Otsu) + pixel-wise AUROC + slice-wise AUROC
- Sweep over (steps, guidance_scale)
- Tkinter GUI for:
    * Train DDPM
    * Train classifier
    * Run anomaly detection on a single FLAIR volume

Patched:
- Removed no_grad around guided steps (fixes autograd error)
- Added CSV saving:
    * Single-case anomaly map CSV
    * Per-slice metrics CSV
    * Summary metrics CSV (Dice / AUROC)
- Updated anomaly scoring to SSIM/L2 (highlights structural abnormalities
  better than L1)
"""

import glob
import argparse
import threading
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim   # <-- NEW
from sklearn.metrics import roc_auc_score

# =========================
# GLOBALS
# =========================

DEVICE = torch.device("cpu")        # CPU only as requested
HEALTHY_CLASS = 0                   # label for healthy slices
IMAGE_SIZE_DEFAULT = 256
DDPM_TIMESTEPS_DEFAULT = 1000


# =========================
# Building blocks: UNet + time embedding
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb_scale = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        emb = self.lin1(emb)
        emb = F.relu(emb)
        emb = self.lin2(emb)
        return emb


class UNetWithTime(nn.Module):
    def __init__(self, in_channels=4, base_ch=64, time_dim=256):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)

        # encoder
        self.inc   = DoubleConv(in_channels, base_ch)        # 4  -> 64
        self.down1 = Down(base_ch, base_ch * 2)              # 64 -> 128
        self.down2 = Down(base_ch * 2, base_ch * 4)          # 128 -> 256

        # decoder (no extra bottleneck level)
        self.up1  = Up(base_ch * 4, base_ch * 2)             # 256 + 128 -> 128
        self.up2  = Up(base_ch * 2, base_ch)                 # 128 + 64  -> 64
        self.outc = nn.Conv2d(base_ch, in_channels, 1)       # 64 -> 4

        # time embedding mapped to 256 channels (same as x3)
        self.t_to_ch = nn.Linear(time_dim, base_ch * 4)      # 256

    def forward(self, x, t):
        # ----- time embedding -----
        t_emb = self.time_mlp(t)               # (B, time_dim)
        t_ch  = self.t_to_ch(t_emb)            # (B, 256)
        t_ch  = t_ch[:, :, None, None]         # (B, 256, 1, 1)

        # ----- encoder -----
        x1 = self.inc(x)                       # (B, 64, 256, 256)
        x2 = self.down1(x1)                    # (B, 128, 128, 128)
        x3 = self.down2(x2)                    # (B, 256, 64, 64)

        # inject time at deepest level
        x3_t = x3 + t_ch                       # (B, 256, 64, 64)

        # ----- decoder -----
        x = self.up1(x3_t, x2)                 # up: 64x64 -> 128x128, concat with x2
        x = self.up2(x,   x1)                  # up: 128x128 -> 256x256, concat with x1
        out = self.outc(x)                     # (B, 4, 256, 256)
        return out


# =========================
# DDPM core
# =========================

class DDPM:
    def __init__(self, model, timesteps=DDPM_TIMESTEPS_DEFAULT,
                 beta_start=1e-4, beta_end=0.02, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(device)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_om * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_noisy, t)
        return F.mse_loss(pred_noise, noise)


# =========================
# Time-conditioned classifier C(x_t, t)
# =========================

class TimeCondClassifier(nn.Module):
    """
    Simple CNN + time embedding. Outputs logits for 2 classes (healthy / diseased).
    """
    def __init__(self, in_channels=4, base_ch=32, time_dim=256, num_classes=2):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.time_to_ch = nn.Linear(time_dim, base_ch)

        self.conv1 = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_ch)

        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(base_ch * 2)

        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(base_ch * 4)

        self.fc = nn.Linear(base_ch * 4, num_classes)

    def forward(self, x, t):
        # x: (B, 4, H, W), t: (B,)
        t_emb = self.time_mlp(t)              # (B, time_dim)
        t_ch = self.time_to_ch(t_emb)         # (B, base_ch)
        t_ch = t_ch[:, :, None, None]         # (B, base_ch, 1, 1)

        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h + t_ch)                  # inject time into first layer

        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        h = h.mean(dim=[2, 3])                # global average pooling
        logits = self.fc(h)
        return logits


# =========================
# BRATS multi-channel dataset
# =========================

class BratsMultiChannelDataset(Dataset):
    """
    root_dir should contain BraTS20_Training_* (or Validation_*) subfolders.

    For each subject we expect:
      *_t1.nii, *_t1ce.nii, *_t2.nii, *_flair.nii, *_seg.nii

    For each slice (axis=2):
      label = 0 if seg slice is all zero (healthy)
      label = 1 if seg slice has any tumor voxels (diseased)
    """
    def __init__(
        self,
        root_dir: str,
        image_size: int = IMAGE_SIZE_DEFAULT,
        slice_axis: int = 2,
        max_subjects: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.slice_axis = slice_axis
        self.max_subjects = max_subjects
        self.samples: List[Tuple[str, int, int]] = []
        self._build_index()

    def _build_index(self):
        pattern_seg = os.path.join(self.root_dir, "**", "*_seg.nii")
        seg_files = glob.glob(pattern_seg, recursive=True)
        if not seg_files:
            raise RuntimeError(
                "No segmentation NIfTI files found.\n"
                f"Tried pattern: {pattern_seg}"
            )

        if self.max_subjects is not None:
            seg_files = seg_files[: self.max_subjects]

        print(f"Found {len(seg_files)} subjects; building slice index...")
        for seg_path in tqdm(seg_files, desc="Indexing subjects"):
            base = seg_path.replace("_seg.nii", "")
            t1_path    = base + "_t1.nii"
            t1ce_path  = base + "_t1ce.nii"
            t2_path    = base + "_t2.nii"
            flair_path = base + "_flair.nii"

            if not (os.path.isfile(t1_path) and os.path.isfile(t1ce_path)
                    and os.path.isfile(t2_path) and os.path.isfile(flair_path)):
                continue  # skip subjects with missing modalities

            seg_vol = nib.load(seg_path).get_fdata().astype(np.float32)
            num_slices = seg_vol.shape[self.slice_axis]
            for s in range(num_slices):
                seg_slice = np.take(seg_vol, indices=s, axis=self.slice_axis)
                label = 1 if np.max(seg_slice) > 0 else 0
                self.samples.append((base, s, label))

        if not self.samples:
            raise RuntimeError("No slices found – check data structure.")
        print(f"Total slices indexed: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _preprocess_slice(self, sl: np.ndarray) -> np.ndarray:
        sl = sl.astype(np.float32)
        sl = (sl - sl.mean()) / (sl.std() + 1e-6)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)  # [0,1]
        sl = sl * 2 - 1  # [-1,1]

        h, w = sl.shape
        size = self.image_size
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        sl = np.pad(sl,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant")
        sl = sl[:size, :size]
        return sl

    def __getitem__(self, idx):
        base, s_idx, label = self.samples[idx]
        t1    = nib.load(base + "_t1.nii").get_fdata().astype(np.float32)
        t1ce  = nib.load(base + "_t1ce.nii").get_fdata().astype(np.float32)
        t2    = nib.load(base + "_t2.nii").get_fdata().astype(np.float32)
        flair = nib.load(base + "_flair.nii").get_fdata().astype(np.float32)

        t1_sl    = np.take(t1,    indices=s_idx, axis=2)
        t1ce_sl  = np.take(t1ce,  indices=s_idx, axis=2)
        t2_sl    = np.take(t2,    indices=s_idx, axis=2)
        flair_sl = np.take(flair, indices=s_idx, axis=2)

        t1_sl    = self._preprocess_slice(t1_sl)
        t1ce_sl  = self._preprocess_slice(t1ce_sl)
        t2_sl    = self._preprocess_slice(t2_sl)
        flair_sl = self._preprocess_slice(flair_sl)

        x = np.stack([t1_sl, t1ce_sl, t2_sl, flair_sl], axis=0)  # (4,H,W)
        x = torch.from_numpy(x)
        y = torch.tensor(label, dtype=torch.long)
        return x, y, base, s_idx


# =========================
# Helper: Dice
# =========================

def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / (denom + 1e-6)


# =========================
# Training: DDPM
# =========================

def train_ddpm(
    data_root: str,
    out_dir: str = "checkpoints",
    image_size: int = IMAGE_SIZE_DEFAULT,
    batch_size: int = 4,
    epochs: int = 1,
    timesteps: int = DDPM_TIMESTEPS_DEFAULT,
    base_channels: int = 64,
    lr: float = 1e-4,
    max_subjects: Optional[int] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    dataset = BratsMultiChannelDataset(
        root_dir=data_root,
        image_size=image_size,
        slice_axis=2,
        max_subjects=max_subjects,
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    model = UNetWithTime(in_channels=4, base_ch=base_channels, time_dim=256)
    ddpm = DDPM(model, timesteps=timesteps, device=DEVICE)
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=lr)

    for epoch in range(epochs):
        ddpm.model.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"DDPM Epoch {epoch + 1}/{epochs}")
        for x, _, _, _ in pbar:
            x = x.to(DEVICE)
            b = x.size(0)
            t = torch.randint(0, timesteps, (b,), device=DEVICE).long()

            loss = ddpm.p_losses(x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * b
            pbar.set_postfix(loss=loss.item())

        avg_loss = running / len(dataset)
        print(f"DDPM Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(out_dir, f"ddpm_brats4ch_epoch{epoch + 1}.pt")
        torch.save(
            {
                "model_state": ddpm.model.state_dict(),
                "timesteps": timesteps,
                "base_channels": base_channels,
                "image_size": image_size,
            },
            ckpt_path,
        )
        print("Saved DDPM checkpoint:", ckpt_path)


def load_ddpm_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    base_channels = ckpt.get("base_channels", 64)
    timesteps = ckpt.get("timesteps", DDPM_TIMESTEPS_DEFAULT)
    image_size = ckpt.get("image_size", IMAGE_SIZE_DEFAULT)

    model = UNetWithTime(in_channels=4, base_ch=base_channels, time_dim=256)
    model.load_state_dict(ckpt["model_state"])
    ddpm = DDPM(model, timesteps=timesteps, device=DEVICE)
    return ddpm, image_size


# =========================
# Training: classifier on noisy x_t
# =========================

def train_classifier(
    data_root: str,
    ddpm_ckpt: str,
    out_dir: str = "checkpoints",
    image_size: int = IMAGE_SIZE_DEFAULT,
    batch_size: int = 8,
    epochs: int = 1,
    timesteps: int = DDPM_TIMESTEPS_DEFAULT,
    lr: float = 1e-4,
    max_subjects: Optional[int] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    ddpm, _ = load_ddpm_from_ckpt(ddpm_ckpt)
    ddpm.model.eval()

    dataset = BratsMultiChannelDataset(
        root_dir=data_root,
        image_size=image_size,
        slice_axis=2,
        max_subjects=max_subjects,
    )
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    clf = TimeCondClassifier(in_channels=4, base_ch=32, time_dim=256, num_classes=2)
    clf.to(DEVICE)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        clf.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}")
        for x, y, _, _ in pbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            b = x.size(0)
            t = torch.randint(0, timesteps, (b,), device=DEVICE).long()

            with torch.no_grad():
                x_t = ddpm.q_sample(x, t)

            logits = clf(x_t, t)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * b
            pbar.set_postfix(loss=loss.item())

        avg_loss = running / len(dataset)
        print(f"Classifier Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(out_dir, f"classifier_brats4ch_epoch{epoch + 1}.pt")
        torch.save(
            {
                "model_state": clf.state_dict(),
                "time_dim": 256,
                "num_classes": 2,
                "timesteps": timesteps,
            },
            ckpt_path,
        )
        print("Saved classifier checkpoint:", ckpt_path)


def load_classifier_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    clf = TimeCondClassifier(
        in_channels=4,
        base_ch=32,
        time_dim=ckpt.get("time_dim", 256),
        num_classes=ckpt.get("num_classes", 2),
    )
    clf.load_state_dict(ckpt["model_state"])
    clf.to(DEVICE)
    clf.eval()
    timesteps = ckpt.get("timesteps", DDPM_TIMESTEPS_DEFAULT)
    return clf, timesteps


# =========================
# Guided translation toward healthy class
# =========================

def guided_translate_to_healthy(
    ddpm: DDPM,
    clf: TimeCondClassifier,
    x0: torch.Tensor,
    steps: int = 50,
    guidance_scale: float = 5.0,
):
    """
    Approximate DDIM-style classifier-guided translation:

    - Deterministic "encoding": x_L = sqrt(alpha_bar_L) * x0  (no noise)
    - Deterministic guided steps back to t=0, using gradient of log p(y=healthy | x_t, t).

    NOTE: This function MUST track gradients inside the loop,
    so do NOT wrap it in torch.no_grad().
    """
    ddpm.model.eval()
    clf.eval()

    b = x0.size(0)
    T = ddpm.timesteps
    t_L = T - 1

    # Encode to large t without noise (deterministic)
    tL = torch.full((b,), t_L, device=DEVICE, dtype=torch.long)
    alpha_bar_L = ddpm.alphas_cumprod[tL].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_L) * x0

    for i in reversed(range(1, steps + 1)):
        t_step = int(i * t_L / steps)
        t = torch.full((b,), t_step, device=DEVICE, dtype=torch.long)
        # enable grad inside the guided step
        with torch.enable_grad():
            x_t = _p_sample_guided(
                ddpm,
                clf,
                x_t,
                t,
                y_target=torch.full((b,), HEALTHY_CLASS, device=DEVICE, dtype=torch.long),
                guidance_scale=guidance_scale,
            )

    return x_t


def _p_sample_guided(
    ddpm: DDPM,
    clf: TimeCondClassifier,
    x: torch.Tensor,
    t: torch.Tensor,
    y_target: torch.Tensor,
    guidance_scale: float = 5.0,
):
    """
    One deterministic backward step with classifier guidance.

    This function assumes gradients are enabled (torch.enable_grad).
    """
    x = x.clone().detach().requires_grad_(True)

    # Predict noise with DDPM model
    eps_pred = ddpm.model(x, t)

    # Classifier guidance
    logits = clf(x, t)
    log_probs = F.log_softmax(logits, dim=1)
    selected = log_probs[torch.arange(x.size(0), device=x.device), y_target]
    grad = torch.autograd.grad(selected.sum(), x)[0]

    sqrt_one_minus_ab = ddpm.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_bar_t = ddpm.alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_bar_prev = ddpm.alphas_cumprod[(t - 1).clamp(min=0)].view(-1, 1, 1, 1)

    eps_guided = eps_pred - guidance_scale * sqrt_one_minus_ab * grad
    x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps_guided) / torch.sqrt(alpha_bar_t)
    x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + torch.sqrt(1 - alpha_bar_prev) * eps_guided
    return x_prev.detach()


# =========================
# Anomaly mapping for one FLAIR volume (single case, GUI)
# =========================

def postprocess_slice(x: np.ndarray) -> np.ndarray:
    x = (x + 1) / 2.0
    x = np.clip(x, 0, 1)
    return x


def ssim_l2_anomaly(orig: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """
    Compute SSIM/L2 anomaly map.

    - orig, rec in [0,1]
    - SSIM map (structural similarity) in [-1,1], convert to (1 - ssim_map)
    - L2 = (orig - rec)^2
    - Final score = (1 - ssim_map) * L2, normalized to [0,1]

    This tends to highlight structural abnormalities better than plain L1.
    """
    # structural similarity map
    _, ssim_map = ssim(orig, rec, data_range=1.0, full=True)
    ssim_anom = 1.0 - ssim_map      # higher where structure differs

    l2_map = (orig - rec) ** 2

    score = ssim_anom * l2_map
    score = score - score.min()
    maxv = score.max()
    if maxv > 0:
        score = score / maxv
    return score


def run_anomaly_single(
    ddpm_ckpt: str,
    clf_ckpt: str,
    flair_nifti_path: str,
    out_dir: str = "outputs_gui",
    steps: int = 50,
    guidance_scale: float = 5.0,
):
    """
    Runs guided translation on the mid-slice of one subject
    and saves:
      - PNG with original / healthy / anomaly
      - CSV with anomaly map values (for that slice)
    """
    os.makedirs(out_dir, exist_ok=True)

    ddpm, image_size = load_ddpm_from_ckpt(ddpm_ckpt)
    clf, _ = load_classifier_from_ckpt(clf_ckpt)
    print("Loaded DDPM:", ddpm_ckpt)
    print("Loaded classifier:", clf_ckpt)

    if not flair_nifti_path.endswith("_flair.nii"):
        raise ValueError("Please select a *_flair.nii file (no .gz).")

    base = flair_nifti_path.replace("_flair.nii", "")
    t1_path    = base + "_t1.nii"
    t1ce_path  = base + "_t1ce.nii"
    t2_path    = base + "_t2.nii"

    for p in [t1_path, t1ce_path, t2_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing modality file: {p}")

    t1    = nib.load(t1_path).get_fdata().astype(np.float32)
    t1ce  = nib.load(t1ce_path).get_fdata().astype(np.float32)
    t2    = nib.load(t2_path).get_fdata().astype(np.float32)
    flair = nib.load(flair_nifti_path).get_fdata().astype(np.float32)

    z_mid = flair.shape[2] // 2

    def prep(vol):
        sl = vol[:, :, z_mid].astype(np.float32)
        sl = (sl - sl.mean()) / (sl.std() + 1e-6)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
        sl = sl * 2 - 1
        h, w = sl.shape
        pad_h = max(0, image_size - h)
        pad_w = max(0, image_size - w)
        sl = np.pad(
            sl,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w //2)),
            mode="constant",
        )
        sl = sl[:image_size, :image_size]
        return sl

    t1_sl    = prep(t1)
    t1ce_sl  = prep(t1ce)
    t2_sl    = prep(t2)
    flair_sl = prep(flair)

    x0_np = np.stack([t1_sl, t1ce_sl, t2_sl, flair_sl], axis=0)
    x0 = torch.from_numpy(x0_np[None, :, :, :]).to(DEVICE)

    # NO torch.no_grad() here because guided_translate needs gradients internally
    x_healthy = guided_translate_to_healthy(
        ddpm, clf, x0, steps=steps, guidance_scale=guidance_scale
    )

    x0_np = x0.squeeze().cpu().numpy()
    xh_np = x_healthy.squeeze().cpu().numpy()

    orig = postprocess_slice(x0_np[3])
    rec  = postprocess_slice(xh_np[3])

    # ---- SSIM/L2 anomaly map ----
    diff = ssim_l2_anomaly(orig, rec)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(orig, cmap="gray")
    axs[0].set_title("Original (FLAIR)")
    axs[1].imshow(rec, cmap="gray")
    axs[1].set_title("Healthy reconstruction (FLAIR)")
    axs[2].imshow(diff, cmap="hot")
    axs[2].set_title("Anomaly (SSIM/L2)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(flair_nifti_path))[0]
    out_png = os.path.join(out_dir, f"{base_name}_anomaly_flair_guided.png")
    plt.savefig(out_png, dpi=150)
    plt.show()

    # Save anomaly map as CSV for this slice
    out_csv = os.path.join(out_dir, f"{base_name}_anomaly_flair_guided.csv")
    np.savetxt(out_csv, diff, delimiter=",")
    print("Saved anomaly PNG:", out_png)
    print("Saved anomaly CSV:", out_csv)

    return out_png


# =========================
# Evaluation on BRATS (Dice + AUROC + CSV)
# =========================

def evaluate_brats(
    data_root: str,
    ddpm_ckpt: str,
    clf_ckpt: str,
    out_dir: str = "eval_results",
    steps: int = 30,
    guidance_scale: float = 5.0,
    max_subjects: Optional[int] = 10,
):
    """
    Evaluates the model on a BRATS set and writes:
      - metrics.txt (mean Dice, pixel AUROC, slice AUROC)
      - slice_metrics.csv (per-slice Dice, mean anomaly, etc.)
      - summary_metrics.csv (single-row aggregated metrics)
    """
    os.makedirs(out_dir, exist_ok=True)

    ddpm, image_size = load_ddpm_from_ckpt(ddpm_ckpt)
    clf, _ = load_classifier_from_ckpt(clf_ckpt)
    ddpm.model.eval()
    clf.eval()

    pattern_seg = os.path.join(data_root, "**", "*_seg.nii")
    seg_files = glob.glob(pattern_seg, recursive=True)
    if not seg_files:
        raise RuntimeError(f"No *_seg.nii files found in {data_root}")

    if max_subjects is not None:
        seg_files = seg_files[:max_subjects]

    print(f"Evaluating on {len(seg_files)} subjects...")

    all_scores = []          # pixel-level anomaly scores
    all_labels = []          # pixel-level GT labels
    dice_list = []           # slice-wise Dice

    slice_rows = []          # for CSV
    slice_scores = []        # scalar per-slice anomaly score
    slice_labels = []        # 0/1 presence of tumor in slice

    for seg_path in tqdm(seg_files, desc="Eval subjects"):
        base = seg_path.replace("_seg.nii", "")
        subj_id = os.path.basename(base)

        t1_path    = base + "_t1.nii"
        t1ce_path  = base + "_t1ce.nii"
        t2_path    = base + "_t2.nii"
        flair_path = base + "_flair.nii"

        if not (os.path.isfile(t1_path) and os.path.isfile(t1ce_path) and
                os.path.isfile(t2_path) and os.path.isfile(flair_path)):
            continue

        seg_vol   = nib.load(seg_path).get_fdata().astype(np.float32)
        t1_vol    = nib.load(t1_path).get_fdata().astype(np.float32)
        t1ce_vol  = nib.load(t1ce_path).get_fdata().astype(np.float32)
        t2_vol    = nib.load(t2_path).get_fdata().astype(np.float32)
        flair_vol = nib.load(flair_path).get_fdata().astype(np.float32)

        num_slices = seg_vol.shape[2]

        for s_idx in range(num_slices):
            seg_slice = seg_vol[:, :, s_idx]
            gt_mask = seg_slice > 0
            has_tumor = int(gt_mask.any())

            def prep_slice(vol):
                sl = vol[:, :, s_idx].astype(np.float32)
                sl = (sl - sl.mean()) / (sl.std() + 1e-6)
                sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
                sl = sl * 2 - 1
                h, w = sl.shape
                pad_h = max(0, image_size - h)
                pad_w = max(0, image_size - w)
                sl = np.pad(
                    sl,
                    ((pad_h // 2, pad_h - pad_h // 2),
                     (pad_w // 2, pad_w - pad_w // 2)),
                    mode="constant",
                )
                sl = sl[:image_size, :image_size]
                return sl

            t1_sl    = prep_slice(t1_vol)
            t1ce_sl  = prep_slice(t1ce_vol)
            t2_sl    = prep_slice(t2_vol)
            flair_sl = prep_slice(flair_vol)

            x0_np = np.stack([t1_sl, t1ce_sl, t2_sl, flair_sl], axis=0)
            x0 = torch.from_numpy(x0_np[None, :, :, :]).to(DEVICE)

            # Guided translation with gradients enabled inside
            x_healthy = guided_translate_to_healthy(
                ddpm, clf, x0,
                steps=steps,
                guidance_scale=guidance_scale,
            )

            x0_np = x0.squeeze().cpu().numpy()
            xh_np = x_healthy.squeeze().cpu().numpy()

            orig = postprocess_slice(x0_np[3])
            rec  = postprocess_slice(xh_np[3])

            # ---- SSIM/L2 anomaly map ----
            diff = ssim_l2_anomaly(orig, rec)

            # Resize GT mask to image_size for Dice/AUROC
            h, w = seg_slice.shape
            pad_h = max(0, image_size - h)
            pad_w = max(0, image_size - w)
            gt_resized = np.pad(
                gt_mask.astype(np.uint8),
                ((pad_h // 2, pad_h - pad_h // 2),
                 (pad_w // 2, pad_w - pad_w // 2)),
                mode="constant",
            )
            gt_resized = gt_resized[:image_size, :image_size].astype(bool)

            # Otsu threshold for this slice
            try:
                thr = threshold_otsu(diff)
            except ValueError:
                thr = diff.mean()
            pred_bin = diff > thr
            dice = dice_score(pred_bin, gt_resized)

            dice_list.append(dice)
            all_scores.append(diff.flatten())
            all_labels.append(gt_resized.flatten().astype(np.uint8))

            # Scalar slice-level anomaly score (mean anomaly)
            slice_score = float(diff.mean())
            slice_scores.append(slice_score)
            slice_labels.append(has_tumor)

            slice_rows.append(
                {
                    "subject": subj_id,
                    "slice_idx": s_idx,
                    "has_tumor": has_tumor,
                    "dice_otsu": dice,
                    "mean_anomaly": slice_score,
                    "otsu_threshold": float(thr),
                }
            )

    if not dice_list:
        print("No slices evaluated (dice_list empty). Check data paths.")
        return

    all_scores_flat = np.concatenate(all_scores, axis=0)
    all_labels_flat = np.concatenate(all_labels, axis=0)

    # Pixel-level AUROC
    if all_labels_flat.max() == 1 and all_labels_flat.min() == 0:
        auc_pixels = roc_auc_score(all_labels_flat, all_scores_flat)
    else:
        auc_pixels = float("nan")

    # Slice-level AUROC (tumor vs no-tumor, using mean anomaly)
    slice_labels_arr = np.array(slice_labels)
    slice_scores_arr = np.array(slice_scores)
    if slice_labels_arr.max() == 1 and slice_labels_arr.min() == 0:
        auc_slices = roc_auc_score(slice_labels_arr, slice_scores_arr)
    else:
        auc_slices = float("nan")

    mean_dice = float(np.mean(dice_list))

    # Write plain-text metrics file
    metrics_path = os.path.join(out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mean Dice (Otsu, slice-wise, SSIM/L2): {mean_dice:.4f}\n")
        f.write(f"Global AUROC (pixel-level, SSIM/L2): {auc_pixels:.4f}\n")
        f.write(f"Slice-wise AUROC (mean anomaly vs tumor presence): {auc_slices:.4f}\n")

    print("===== Evaluation finished =====")
    print(f"Mean Dice (slice-wise): {mean_dice:.4f}")
    print(f"Global AUROC (pixel-level): {auc_pixels:.4f}")
    print(f"Slice-wise AUROC: {auc_slices:.4f}")
    print(f"Saved metrics to: {metrics_path}")

    # ---- CSV outputs ----
    # Per-slice metrics
    df_slices = pd.DataFrame(slice_rows)
    slice_csv_path = os.path.join(out_dir, "slice_metrics.csv")
    df_slices.to_csv(slice_csv_path, index=False)
    print("Saved per-slice metrics CSV:", slice_csv_path)

    # Summary metrics (single row)
    df_summary = pd.DataFrame(
        [
            {
                "mean_dice_slice_otsu": mean_dice,
                "pixel_auroc": auc_pixels,
                "slice_auroc": auc_slices,
                "num_slices": len(dice_list),
            }
        ]
    )
    summary_csv_path = os.path.join(out_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print("Saved summary metrics CSV:", summary_csv_path)


# =========================
# Sweep over (steps, guidance_scale)
# =========================

def sweep_eval_brats(
    data_root: str,
    ddpm_ckpt: str,
    clf_ckpt: str,
    out_root: str = "eval_sweep",
    settings: Optional[List[Tuple[int, float]]] = None,
    max_subjects: Optional[int] = 5,
):
    if settings is None:
        settings = [
            (20, 3.0),
            (20, 5.0),
            (40, 3.0),
            (40, 5.0),
        ]

    os.makedirs(out_root, exist_ok=True)

    print("==== Starting sweep over (steps, guidance_scale) ====")
    print("Settings:", settings)

    for steps, s in settings:
        tag = f"steps{steps}_s{s:.1f}".replace(".", "p")
        out_dir = os.path.join(out_root, tag)
        print(f"\n--- Evaluating setting: steps={steps}, s={s} ---")
        evaluate_brats(
            data_root=data_root,
            ddpm_ckpt=ddpm_ckpt,
            clf_ckpt=clf_ckpt,
            out_dir=out_dir,
            steps=steps,
            guidance_scale=s,
            max_subjects=max_subjects,
        )


# =========================
# Tkinter GUI
# =========================

class AnomalyGUI:
    def __init__(self, root):
        self.root = root
        root.title("BRATS 4-channel DDPM – Guided Anomaly (CPU, SSIM/L2)")
        root.geometry("860x340")

        # ----- DDPM training -----
        self.train_root_var = tk.StringVar(value="")
        tk.Label(root, text="Training data folder (BraTS):").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(root, textvariable=self.train_root_var, width=60).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Button(root, text="Train DDPM (1 epoch, CPU)",
                  command=self.train_ddpm_clicked).grid(
            row=1, column=1, pady=5
        )

        # ----- Classifier training -----
        self.ddpm_for_clf_var = tk.StringVar()
        tk.Label(root, text="DDPM ckpt for classifier:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(root, textvariable=self.ddpm_for_clf_var, width=60).grid(
            row=2, column=1, padx=5, pady=5
        )
        tk.Button(root, text="Browse", command=self.choose_ddpm_for_clf).grid(
            row=2, column=2, padx=5, pady=5
        )

        tk.Button(root, text="Train classifier (noisy x_t)",
                  command=self.train_clf_clicked).grid(
            row=3, column=1, pady=5
        )

        # ----- Anomaly inference -----
        self.ddpm_ckpt_var = tk.StringVar()
        self.clf_ckpt_var = tk.StringVar()
        self.nifti_var = tk.StringVar()

        tk.Label(root, text="DDPM checkpoint (.pt):").grid(
            row=4, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(root, textvariable=self.ddpm_ckpt_var, width=60).grid(
            row=4, column=1, padx=5, pady=5
        )
        tk.Button(root, text="Browse", command=self.choose_ddpm_ckpt).grid(
            row=4, column=2, padx=5, pady=5
        )

        tk.Label(root, text="Classifier checkpoint (.pt):").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(root, textvariable=self.clf_ckpt_var, width=60).grid(
            row=5, column=1, padx=5, pady=5
        )
        tk.Button(root, text="Browse", command=self.choose_clf_ckpt).grid(
            row=5, column=2, padx=5, pady=5
        )

        tk.Label(root, text="FLAIR volume (*.nii):").grid(
            row=6, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(root, textvariable=self.nifti_var, width=60).grid(
            row=6, column=1, padx=5, pady=5
        )
        tk.Button(root, text="Browse", command=self.choose_nifti).grid(
            row=6, column=2, padx=5, pady=5
        )

        tk.Button(root, text="Run anomaly detection (SSIM/L2)",
                  command=self.run_anomaly).grid(
            row=7, column=1, pady=10
        )

    # --- training DDPM ---
    def choose_train_root(self):
        path = filedialog.askdirectory()
        if path:
            self.train_root_var.set(path)

    def train_ddpm_clicked(self):
        data_root = self.train_root_var.get().strip()
        if not data_root or not os.path.isdir(data_root):
            messagebox.showerror(
                "Error",
                "Please select a valid training data folder (BraTS root).",
            )
            return

        def _run():
            try:
                train_ddpm(
                    data_root=data_root,
                    out_dir="checkpoints",
                    image_size=IMAGE_SIZE_DEFAULT,
                    batch_size=4,
                    epochs=1,
                    timesteps=DDPM_TIMESTEPS_DEFAULT,
                    base_channels=64,
                    lr=1e-4,
                    max_subjects=None,
                )
                messagebox.showinfo(
                    "DDPM training finished",
                    "Checkpoints saved in ./checkpoints/",
                )
            except Exception as e:
                messagebox.showerror("Training error", str(e))

        threading.Thread(target=_run, daemon=True).start()

    # --- training classifier ---
    def choose_ddpm_for_clf(self):
        path = filedialog.askopenfilename(
            filetypes=[("DDPM checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.ddpm_for_clf_var.set(path)

    def train_clf_clicked(self):
        data_root = self.train_root_var.get().strip()
        ddpm_ckpt = self.ddpm_for_clf_var.get().strip()
        if not data_root or not os.path.isdir(data_root):
            messagebox.showerror(
                "Error",
                "Please select training data folder (same BraTS root).",
            )
            return
        if not os.path.isfile(ddpm_ckpt):
            messagebox.showerror(
                "Error",
                "Please select a valid DDPM checkpoint for classifier training.",
            )
            return

        def _run():
            try:
                train_classifier(
                    data_root=data_root,
                    ddpm_ckpt=ddpm_ckpt,
                    out_dir="checkpoints",
                    image_size=IMAGE_SIZE_DEFAULT,
                    batch_size=8,
                    epochs=1,
                    timesteps=DDPM_TIMESTEPS_DEFAULT,
                    lr=1e-4,
                    max_subjects=None,
                )
                messagebox.showinfo(
                    "Classifier training finished",
                    "Classifier checkpoints saved in ./checkpoints/",
                )
            except Exception as e:
                messagebox.showerror("Training error", str(e))

        threading.Thread(target=_run, daemon=True).start()

    # --- anomaly ---
    def choose_ddpm_ckpt(self):
        path = filedialog.askopenfilename(
            filetypes=[("DDPM checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.ddpm_ckpt_var.set(path)

    def choose_clf_ckpt(self):
        path = filedialog.askopenfilename(
            filetypes=[("Classifier checkpoint", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.clf_ckpt_var.set(path)

    def choose_nifti(self):
        path = filedialog.askopenfilename(
            filetypes=[("FLAIR NIfTI", "*_flair.nii"), ("All files", "*.*")]
        )
        if path:
            self.nifti_var.set(path)

    def run_anomaly(self):
        ddpm_ckpt = self.ddpm_ckpt_var.get().strip()
        clf_ckpt = self.clf_ckpt_var.get().strip()
        nii = self.nifti_var.get().strip()

        if not os.path.isfile(ddpm_ckpt):
            messagebox.showerror("Error", "Select a valid DDPM checkpoint.")
            return
        if not os.path.isfile(clf_ckpt):
            messagebox.showerror("Error", "Select a valid classifier checkpoint.")
            return
        if not os.path.isfile(nii):
            messagebox.showerror("Error", "Select a valid *_flair.nii file.")
            return

        try:
            out_path = run_anomaly_single(ddpm_ckpt, clf_ckpt, nii)
            messagebox.showinfo("Done", f"Saved anomaly PNG and CSV.\n{out_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# =========================
# Main (CLI + GUI)
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_ddpm", "train_classifier", "eval", "sweep_eval", "gui"],
        default="gui",
    )
    parser.add_argument("--data_root", type=str,
                        help="BraTS root folder with BraTS20_Training_* etc.")
    parser.add_argument("--ddpm_ckpt", type=str,
                        help="DDPM checkpoint (.pt)")
    parser.add_argument("--clf_ckpt", type=str,
                        help="Classifier checkpoint (.pt)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_subjects", type=int, default=0,
                        help="0 = use all subjects, >0 = limit subjects")
    args = parser.parse_args()

    max_subj = None if args.max_subjects == 0 else args.max_subjects

    if args.mode == "train_ddpm":
        if not args.data_root:
            raise SystemExit("--data_root is required for train_ddpm.")
        train_ddpm(
            data_root=args.data_root,
            out_dir="checkpoints",
            image_size=IMAGE_SIZE_DEFAULT,
            batch_size=args.batch_size,
            epochs=args.epochs,
            timesteps=DDPM_TIMESTEPS_DEFAULT,
            base_channels=64,
            lr=1e-4,
            max_subjects=max_subj,
        )

    elif args.mode == "train_classifier":
        if not args.data_root or not args.ddpm_ckpt:
            raise SystemExit("--data_root and --ddpm_ckpt are required for train_classifier.")
        train_classifier(
            data_root=args.data_root,
            ddpm_ckpt=args.ddpm_ckpt,
            out_dir="checkpoints",
            image_size=IMAGE_SIZE_DEFAULT,
            batch_size=max(2, args.batch_size),
            epochs=args.epochs,
            timesteps=DDPM_TIMESTEPS_DEFAULT,
            lr=1e-4,
            max_subjects=max_subj,
        )

    elif args.mode == "eval":
        if not args.data_root or not args.ddpm_ckpt or not args.clf_ckpt:
            raise SystemExit("--data_root, --ddpm_ckpt and --clf_ckpt are required for eval.")
        evaluate_brats(
            data_root=args.data_root,
            ddpm_ckpt=args.ddpm_ckpt,
            clf_ckpt=args.clf_ckpt,
            out_dir="eval_results",
            steps=30,
            guidance_scale=5.0,
            max_subjects=max_subj,
        )

    elif args.mode == "sweep_eval":
        if not args.data_root or not args.ddpm_ckpt or not args.clf_ckpt:
            raise SystemExit("--data_root, --ddpm_ckpt and --clf_ckpt are required for sweep_eval.")
        sweep_eval_brats(
            data_root=args.data_root,
            ddpm_ckpt=args.ddpm_ckpt,
            clf_ckpt=args.clf_ckpt,
            out_root="eval_sweep",
            settings=[
                (20, 3.0),
                (20, 5.0),
                (40, 3.0),
                (40, 5.0),
            ],
            max_subjects=max_subj if max_subj is not None else 5,
        )

    else:  # gui
        root = tk.Tk()
        app = AnomalyGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()
