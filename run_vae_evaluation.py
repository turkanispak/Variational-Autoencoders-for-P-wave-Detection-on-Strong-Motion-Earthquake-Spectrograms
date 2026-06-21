"""
VAE evaluation (Table 1 + reconstruction-detection trade-off) and Attention-VAE
SNR-binned detection performance.

Layout expected under MODELS_DIR (flat):
    experiments/
        <run_name>.pt              one file per trained model
        ...
        test_p_recordings.json     single shared test split

Outputs (in OUT_DIR):
    results_summary.csv            detection table; SNR table appended if RUN_SNR
    01_recon_det_tradeoff.png      AUC vs MAE scatter (VAE only)
    auc_vs_snr_vae.png             Attention-VAE AUC across SNR bins (VAE only)

"""

import os
import glob
import json
import random
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")

# ==========================================================================
# CONFIG  -- edit
# ==========================================================================
DATA_PATH  = "dataset.hdf5"
MODELS_DIR = "experiments"                                  
TEST_JSON  = "experiments/test_p_recordings.json"            
SNR_MODEL  = "experiments/attn_no_skips_ld128_d16_h4_20251013_012554.pt"  # model used for SNR

RUN_GRID = True    # AUC + MAE over all models -> detection table + trade-off plot
RUN_SNR  = True    # SNR-binned AUC for SNR_MODEL

NOISE_SEC = 1.0    # pre-P noise window for SNR
SIG_SEC   = 1.44   # post-P signal window for SNR
N_BINS    = 4      # SNR quantile bins

OUT_DIR = "eval_outputs"
SEED    = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Frozen evaluation geometry
SAMPLING_RATE       = 100
SNIPPET_SEC         = 2.44
SHIFT_SEC           = 0.1
NCC_EARLY_EXTRA_SEC = 2.44

# Spectrogram
N_FFT = 62
HOP = 2
FREQ_BINS = 32
TIME_BINS = 92

CSV_PATH = None  # set in main()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ==========================================================================
# DATA UTILITIES
# ==========================================================================
def read_hdf5(file_path, sampling_rate, expected_seconds):
    samples_dict = {"sample": [], "p_index": [], "rec_name": []}
    expected_len = expected_seconds * sampling_rate
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    with h5py.File(file_path, "r") as hdf5_file:
        group = hdf5_file["data"]
        for key in group.keys():
            dataset = group[key]
            data = dataset[:]
            if data.shape[1] != expected_len:
                continue
            p_ts = dataset.attrs["p_arrival_sample"]
            p_idx = int(p_ts * sampling_rate) if p_ts != "None" else None
            samples_dict["sample"].append(data)
            samples_dict["rec_name"].append(key)
            samples_dict["p_index"].append(p_idx)
    return samples_dict


def subset_samples(samples_dict, rec_names):
    name_to_idx = {name: i for i, name in enumerate(samples_dict["rec_name"])}
    indices = [name_to_idx[n] for n in rec_names if n in name_to_idx]
    return {k: ([v[i] for i in indices] if isinstance(v, list) else v)
            for k, v in samples_dict.items()}


def waveform_to_spec(snippet):
    channels = []
    for ch in range(3):
        _, _, Sxx = sg.spectrogram(
            snippet[ch], fs=SAMPLING_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP,
            window="hann", mode="magnitude",
        )
        Sxx = np.log1p(Sxx)
        if Sxx.shape[0] > FREQ_BINS:
            Sxx = Sxx[:FREQ_BINS, :]
        if Sxx.shape[1] > TIME_BINS:
            Sxx = Sxx[:, :TIME_BINS]
        pad_f = max(0, FREQ_BINS - Sxx.shape[0])
        pad_t = max(0, TIME_BINS - Sxx.shape[1])
        if pad_f or pad_t:
            Sxx = np.pad(Sxx, ((0, pad_f), (0, pad_t)), "constant")
        channels.append(Sxx)
    return np.stack(channels, axis=0)


# ==========================================================================
# MODEL DEFINITIONS
# ==========================================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(),
                                  ResidualBlock(12), ResidualBlock(12))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(),
                                  ResidualBlock(24), ResidualBlock(24))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(),
                                  ResidualBlock(48), ResidualBlock(48))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(48 * 8 * 23, latent_dim)
        self.fc_logvar = nn.Linear(48 * 8 * 23, latent_dim)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        self.skip1, self.skip2 = x1, x2
        mu = self.fc_mu(self.flatten(x3))
        logvar = self.fc_logvar(self.flatten(x3))
        return mu, logvar, x3


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(),
                                  ResidualBlock(24), ResidualBlock(24))
        self.up2 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(),
                                  ResidualBlock(12), ResidualBlock(12))
        self.out_conv = nn.Conv2d(12, 3, 1)

    def forward_from_features(self, x, skip1, skip2):
        x = self.up3(x)
        x = self.dec3(torch.cat([x, skip2], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, skip1], dim=1))
        return self.out_conv(x)


class ConvVAE(nn.Module):
    """Hybrid: attention + skips."""
    def __init__(self, latent_dim=128, transformer_depth=1, transformer_heads=4):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder()
        self.fc = nn.Linear(latent_dim, 48 * 8 * 23)
        self.z_to_tokens = nn.Linear(latent_dim, 184 * 48)
        nn.init.xavier_uniform_(self.z_to_tokens.weight)
        nn.init.constant_(self.z_to_tokens.bias, 0)
        self.transformer_blocks = nn.ModuleList(
            [Block(dim=48, num_heads=transformer_heads) for _ in range(transformer_depth)])
        self.norm = nn.LayerNorm(48)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        tokens = self.z_to_tokens(z).view(-1, 184, 48)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat = tokens.transpose(1, 2).reshape(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat, self.encoder.skip1, self.encoder.skip2)
        return recon, mu, logvar


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class TransformerBlock_NoSkip(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, qkv_bias=True, drop=0.,
                 attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = self.mlp(self.norm2(x))
        return x


class ConvEncoder_NoSkips(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(inplace=True), ConvBlock(12))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(inplace=True), ConvBlock(24))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(inplace=True), ConvBlock(48))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(48 * 8 * 23, latent_dim)
        self.fc_logvar = nn.Linear(48 * 8 * 23, latent_dim)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.pool1(x))
        x = self.enc3(self.pool2(x))
        flat = self.flatten(x)
        return self.fc_mu(flat), self.fc_logvar(flat), x


class ConvDecoder_NoSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(24))
        self.up2 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(12))
        self.out_conv = nn.Conv2d(12, 3, 1)

    def forward_from_features(self, x):
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        return self.out_conv(x)


class ConvVAE_Attention_NoSkips(nn.Module):
    def __init__(self, latent_dim=128, transformer_depth=1, transformer_heads=4):
        super().__init__()
        self.encoder = ConvEncoder_NoSkips(latent_dim=latent_dim)
        self.decoder = ConvDecoder_NoSkips()
        self.z_to_tokens = nn.Linear(latent_dim, 184 * 48)
        nn.init.xavier_uniform_(self.z_to_tokens.weight)
        nn.init.constant_(self.z_to_tokens.bias, 0)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock_NoSkip(dim=48, num_heads=transformer_heads) for _ in range(transformer_depth)])
        self.post_norm = nn.LayerNorm(48)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        tokens = self.z_to_tokens(z).view(-1, 184, 48)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.post_norm(tokens)
        feat = tokens.transpose(1, 2).reshape(-1, 48, 8, 23)
        return self.decoder.forward_from_features(feat), mu, logvar


class ConvVAE_Skip_NoAttention(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder()
        self.fc = nn.Linear(latent_dim, 48 * 8 * 23)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        feat = self.fc(z).view(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat, self.encoder.skip1, self.encoder.skip2)
        return recon, mu, logvar


class BasicConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class BasicEncoder_NoSkips(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 12, 3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(12))
        self.pool1 = nn.MaxPool2d(2)
        self.stage2 = nn.Sequential(nn.Conv2d(12, 24, 3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(24))
        self.pool2 = nn.MaxPool2d(2)
        self.stage3 = nn.Sequential(nn.Conv2d(24, 48, 3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(48))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(48 * 8 * 23, latent_dim)
        self.fc_logvar = nn.Linear(48 * 8 * 23, latent_dim)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(self.pool1(x))
        x = self.stage3(self.pool2(x))
        flat = self.flatten(x)
        return self.fc_mu(flat), self.fc_logvar(flat), x


class BasicDecoder_NoSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec1 = BasicConvBlock(24)
        self.up2 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.dec2 = BasicConvBlock(12)
        self.out_conv = nn.Conv2d(12, 3, 1)

    def forward_from_features(self, feat):
        x = self.dec1(self.up1(feat))
        x = self.dec2(self.up2(x))
        return self.out_conv(x)


class ConvVAE_Basic(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = BasicEncoder_NoSkips(latent_dim=latent_dim)
        self.decoder = BasicDecoder_NoSkips()
        self.fc_z_to_feat = nn.Linear(latent_dim, 48 * 8 * 23)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        feat = self.fc_z_to_feat(z).view(-1, 48, 8, 23)
        return self.decoder.forward_from_features(feat), mu, logvar


# ==========================================================================
# MODEL LOADING
# ==========================================================================
def parse_run_config(run_name):
    name = run_name.lower()
    if "basic" in name and "no_attn" in name:
        arch = "basic"
    elif "attn_no_skip" in name:
        arch = "attn_no_skips"
    elif "skip_no_attn" in name:
        arch = "skip_no_attn"
    elif "hybrid" in name:
        arch = "original"
    else:
        arch = "original"
    ld, depth, heads = 128, 1, 4
    for p in run_name.split("_"):
        if p.startswith("ld") and p[2:].isdigit():
            ld = int(p[2:])
        elif p.startswith("d") and p[1:].isdigit():
            depth = int(p[1:])
        elif p.startswith("h") and p[1:].isdigit():
            heads = int(p[1:])
    return arch, ld, depth, heads


def build_model(arch, ld, depth, heads):
    if arch == "original":
        return ConvVAE(latent_dim=ld, transformer_depth=depth, transformer_heads=heads)
    if arch == "skip_no_attn":
        return ConvVAE_Skip_NoAttention(latent_dim=ld)
    if arch == "attn_no_skips":
        return ConvVAE_Attention_NoSkips(latent_dim=ld, transformer_depth=depth, transformer_heads=heads)
    return ConvVAE_Basic(latent_dim=ld)


def load_model(model_file):
    run_name = os.path.splitext(os.path.basename(model_file))[0]
    arch, ld, depth, heads = parse_run_config(run_name)
    model = build_model(arch, ld, depth, heads).to(DEVICE)
    state = torch.load(model_file, map_location=DEVICE, weights_only=False)
    if list(state.keys())[0].startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model, arch, ld, depth, heads


# ==========================================================================
# EVALUATION
# ==========================================================================
def sliding_window_eval(model, samples_dict):
    model.eval()
    snippet_len = int(SNIPPET_SEC * SAMPLING_RATE)
    shift_len = int(SHIFT_SEC * SAMPLING_RATE)
    delta_len = int(NCC_EARLY_EXTRA_SEC * SAMPLING_RATE)
    results = []
    with torch.no_grad():
        iterator = tqdm(enumerate(samples_dict["sample"]),
                        total=len(samples_dict["sample"]), desc="  Evaluated", unit="rec")
        for i, signal in iterator:
            p_idx = samples_dict["p_index"][i]
            cutoff = signal.shape[1]
            if p_idx is not None:
                cutoff = min(signal.shape[1], p_idx + snippet_len + delta_len)
            ncc_curve, mae_curve, starts = [], [], []
            for start in range(0, cutoff - snippet_len + 1, shift_len):
                snippet = signal[:, start:start + snippet_len]
                spec_np = waveform_to_spec(snippet)
                spec_t = torch.tensor(spec_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                recon = model(spec_t)[0].squeeze(0).cpu().numpy()
                mae_curve.append(np.mean(np.abs(spec_np - recon)))
                a, b = spec_np.flatten(), recon.flatten()
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                ncc_curve.append(np.dot(a, b) / denom if denom > 0 else 0.0)
                starts.append(start)
            results.append({
                "rec_name": samples_dict["rec_name"][i],
                "p_index": p_idx,
                "ncc_curve": np.array(ncc_curve),
                "mae_curve": np.array(mae_curve),
                "starts": np.array(starts),
            })
    return results


def compute_auc(results):
    y_true, y_scores = [], []
    tol_samples = int(0.5 * SAMPLING_RATE)
    win_len = int(SNIPPET_SEC * SAMPLING_RATE)
    for res in results:
        p_idx = res["p_index"]
        if p_idx is None:
            continue
        for ncc, start in zip(res["ncc_curve"], res["starts"]):
            end = start + win_len
            is_pos = (abs(p_idx - start) <= tol_samples) or (start <= p_idx < end)
            y_true.append(1 if is_pos else 0)
            y_scores.append(ncc)
    if not y_true or sum(y_true) == 0:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr)


# ==========================================================================
# SNR + binning
# ==========================================================================
def compute_snr_db(signal, p_idx, noise_sec, sig_sec):
    if p_idx is None:
        return np.nan
    n = signal.shape[1]
    n0, n1 = max(0, p_idx - int(noise_sec * SAMPLING_RATE)), p_idx
    s0, s1 = p_idx, min(n, p_idx + int(sig_sec * SAMPLING_RATE))
    if n1 - n0 < 2 or s1 - s0 < 2:
        return np.nan
    noise_pow = np.mean(signal[:, n0:n1] ** 2)
    sig_pow = np.mean(signal[:, s0:s1] ** 2)
    if noise_pow <= 0 or sig_pow <= 0:
        return np.nan
    return 10.0 * np.log10(sig_pow / noise_pow)


def assign_bins(snr, n_bins):
    snr = np.asarray(snr, dtype=float)
    valid = np.isfinite(snr)
    edges = np.quantile(snr[valid], np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.full(len(snr), -1, dtype=int)
    idx[valid] = np.clip(np.digitize(snr[valid], edges[1:-1], right=False), 0, len(edges) - 2)
    labels = []
    for b in range(len(edges) - 1):
        lo = "-inf" if np.isneginf(edges[b]) else f"{edges[b]:.1f}"
        hi = "inf" if np.isposinf(edges[b + 1]) else f"{edges[b + 1]:.1f}"
        labels.append(f"{lo} to {hi} dB")
    return idx, labels


def per_bin_auc(name_to_res, per_rec, n_bins):
    out = []
    for b in range(n_bins):
        recs = per_rec.loc[per_rec["bin"] == b, "rec_name"].tolist()
        subset = [name_to_res[r] for r in recs if r in name_to_res]
        out.append(compute_auc(subset) if subset else float("nan"))
    return out


# ==========================================================================
# OUTPUT HELPERS
# ==========================================================================
def write_section(title, df, fresh):
    """Write a titled CSV section. fresh=True starts the file; else append."""
    mode = "w" if fresh else "a"
    with open(CSV_PATH, mode, newline="") as f:
        if not fresh:
            f.write("\n")
        f.write(f"# {title}\n")
    df.to_csv(CSV_PATH, mode="a", index=False)


def plot_tradeoff(df, save_path):
    style = {
        "basic":         {"label": "Basic VAE (No Attention, No Skips)", "marker": "o", "color": (0, 0.45, 0.74)},
        "skip_no_attn":  {"label": "Skip VAE (No Attention)",            "marker": "s", "color": (0.85, 0.33, 0.10)},
        "attn_no_skips": {"label": "Attention VAE (No Skips)",           "marker": "^", "color": (0.47, 0.67, 0.19)},
        "original":      {"label": "Hybrid VAE (Attention, Skips)",      "marker": "D", "color": (0.49, 0.18, 0.56)},
    }
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    aucs, maes = [], []
    for _, row in df.iterrows():
        if pd.isna(row["auc"]) or pd.isna(row["mae"]):
            continue
        aucs.append(row["auc"]); maes.append(row["mae"])
        s = style.get(row["arch"], {"label": row["arch"], "marker": "x", "color": "black"})
        plt.scatter(row["auc"], row["mae"], s=150, marker=s["marker"],
                    facecolors=s["color"], edgecolors="k", linewidths=1.5, zorder=3)
        ha = "right" if row["auc"] < 0.83 else "left"
        plt.text(row["auc"] + (-0.003 if ha == "right" else 0.003), row["mae"] + 0.00006,
                 s["label"], horizontalalignment=ha, fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    plt.grid(True, linestyle="-", linewidth=0.5)
    plt.xlabel(r"AUC (ROC) $\rightarrow$", fontweight="bold", fontsize=12)
    plt.ylabel(r"Reconstruction MAE $\leftarrow$ (lower is better)", fontweight="bold", fontsize=12)
    plt.title("Detection vs Reconstruction", fontweight="bold", fontsize=14)
    if aucs and maes:
        plt.xlim(min(aucs) - 0.01, max(aucs) + 0.01)
        plt.ylim(max(maes) + 0.0003, min(maes) - 0.0003)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Trade-off plot saved to {save_path}")


def plot_snr_vae(bin_df, save_path):
    bin_df = bin_df[bin_df["n"] > 0].reset_index(drop=True)
    x = np.arange(len(bin_df))
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(x, bin_df["auc"], "-o", color=(0, 0.45, 0.74), lw=2, markersize=10,
             markerfacecolor=(0, 0.45, 0.74), markeredgecolor="k", label="Attention-VAE")
    for xi, row in zip(x, bin_df.itertuples()):
        plt.annotate(f"n={int(row.n)}", (xi, row.auc), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontweight="bold")
    plt.xticks(x, bin_df["snr_range"], rotation=15)
    plt.ylim(0.5, 1.0)
    plt.ylabel("ROC AUC Score", fontweight="bold", fontsize=12)
    plt.xlabel("SNR Range (dB)", fontweight="bold", fontsize=12)
    plt.title("Detection Performance vs SNR", fontweight="bold", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  SNR plot saved to {save_path}")


# ==========================================================================
# DRIVERS
# ==========================================================================
def run_grid_evaluation(full_data):
    print("Grid evaluation (AUC + MAE)...")
    model_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))
    print(f"  Found {len(model_files)} model(s) in {MODELS_DIR}")
    with open(TEST_JSON) as f:
        test_data = subset_samples(full_data, json.load(f))
    rows = []
    for model_file in model_files:
        run_name = os.path.splitext(os.path.basename(model_file))[0]
        try:
            model, arch, ld, depth, heads = load_model(model_file)
        except Exception as e:
            print(f"  [skip] {run_name}: {e}")
            continue
        print(f"\n  {run_name}  (arch={arch} ld={ld} depth={depth} heads={heads})")
        res = sliding_window_eval(model, test_data)
        auc_score = compute_auc(res)
        maes = np.concatenate([r["mae_curve"] for r in res if len(r["mae_curve"])])
        mean_mae = float(np.mean(maes)) if len(maes) else float("nan")
        print(f"    AUC={auc_score:.4f}  MAE={mean_mae:.6f}")
        rows.append({"run": run_name, "arch": arch,
                     "auc": round(auc_score, 3), "mae": round(mean_mae, 5)})
    if not rows:
        print("  No grid results.")
        return
    df = pd.DataFrame(rows)
    write_section("VAE detection summary", df, fresh=True)
    plot_tradeoff(df, os.path.join(OUT_DIR, "01_recon_det_tradeoff.png"))
    print(f"  Detection summary written to {CSV_PATH}")


def run_snr_evaluation(full_data):
    print("\nSNR-binned evaluation (Attention-VAE)...")
    with open(TEST_JSON) as f:
        data = subset_samples(full_data, json.load(f))
    snr = np.array([compute_snr_db(data["sample"][i], data["p_index"][i], NOISE_SEC, SIG_SEC)
                    for i in range(len(data["sample"]))])
    bin_idx, labels = assign_bins(snr, N_BINS)
    per_rec = pd.DataFrame({"rec_name": data["rec_name"], "bin": bin_idx})
    model = load_model(SNR_MODEL)[0]
    res_map = {r["rec_name"]: r for r in sliding_window_eval(model, data)}
    aucs = per_bin_auc(res_map, per_rec, len(labels))
    bin_df = pd.DataFrame([
        {"bin": b, "snr_range": labels[b], "n": int((per_rec["bin"] == b).sum()),
         "auc": round(aucs[b], 3)} for b in range(len(labels))])
    write_section("Attention-VAE SNR bins", bin_df, fresh=not (RUN_GRID and os.path.exists(CSV_PATH)))
    for r in bin_df.itertuples():
        print(f"  Bin {r.bin} [{r.snr_range}]: n={r.n:3d}  AUC={r.auc:.3f}")
    plot_snr_vae(bin_df, os.path.join(OUT_DIR, "auc_vs_snr.png"))
    print(f"  SNR table appended to {CSV_PATH}")


def main():
    global CSV_PATH
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    CSV_PATH = os.path.join(OUT_DIR, "results_summary.csv")
    full_data = read_hdf5(DATA_PATH, SAMPLING_RATE, 30)
    if RUN_GRID:
        run_grid_evaluation(full_data)
    if RUN_SNR:
        run_snr_evaluation(full_data)
    print("\nDone.")


if __name__ == "__main__":
    main()