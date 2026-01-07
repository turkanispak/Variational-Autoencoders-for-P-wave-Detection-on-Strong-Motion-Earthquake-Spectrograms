"""
Seismic VAE Evaluation Pipeline
"""

import os
import glob
import json
import h5py
import numpy as np
import pandas as pd
import scipy.signal as sg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import warnings
import re

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "dataset.hdf5"        
MODELS_DIR = "experiments"        
OUTPUT_DIR = "eval_outputs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLING_RATE = 100
EXPECTED_SECONDS = 30
SNIPPET_SEC = 2.44
SHIFT_SEC = 0.1
TOLERANCE_SEC = 0.0
NCC_EARLY_EXTRA_SEC = 2.44

N_FFT = 62
HOP = 2
FREQ_BINS = 32
TIME_BINS = 92


# ==========================================
# 2. DATA UTILITIES
# ==========================================

def read_hdf5(file_path, sampling_rate, expected_seconds):
    samples_dict = {'sample': [], 'p_index': [], 'rec_name': []}
    expected_len = expected_seconds * sampling_rate

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    with h5py.File(file_path, 'r') as hdf5_file:
        dataset_group = hdf5_file['data']
        for key in dataset_group.keys():
            dataset = dataset_group[key]
            data = dataset[:]
            if data.shape[1] != expected_len: continue

            p_ts = dataset.attrs['p_arrival_sample']
            p_idx = int(p_ts * sampling_rate) if p_ts != 'None' else None

            samples_dict['sample'].append(data)
            samples_dict['rec_name'].append(key)
            samples_dict['p_index'].append(p_idx)
            
    return samples_dict

def subset_samples(samples_dict, rec_names):
    name_to_idx = {name: i for i, name in enumerate(samples_dict['rec_name'])}
    indices = [name_to_idx[name] for name in rec_names if name in name_to_idx]
    subset = {}
    for k, v in samples_dict.items():
        if isinstance(v, list):
            subset[k] = [v[i] for i in indices]
        else:
            subset[k] = v
    return subset

def waveform_to_spec(snippet):
    channels = []
    for ch in range(3):
        f, t, Sxx = sg.spectrogram(
            snippet[ch], fs=SAMPLING_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP, window='hann', mode='magnitude'
        )
        Sxx = np.log1p(Sxx)
        if Sxx.shape[0] > FREQ_BINS: Sxx = Sxx[:FREQ_BINS, :]
        if Sxx.shape[1] > TIME_BINS: Sxx = Sxx[:, :TIME_BINS]
        pad_f = max(0, FREQ_BINS - Sxx.shape[0])
        pad_t = max(0, TIME_BINS - Sxx.shape[1])
        if pad_f > 0 or pad_t > 0:
            Sxx = np.pad(Sxx, ((0, pad_f), (0, pad_t)), 'constant')
        channels.append(Sxx)
    return np.stack(channels, axis=0)

# ==========================================
# 3. MODEL DEFINITIONS 
# ==========================================

# --- Common Components ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=[8, 23], patch_size=[4, 4], in_chans=48, embed_dim=48):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# --- Original Encoder/Decoder ---
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1), nn.ReLU(), ResidualBlock(12), ResidualBlock(12))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1), nn.ReLU(), ResidualBlock(24), ResidualBlock(24))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(), ResidualBlock(48), ResidualBlock(48))
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
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(48, 24, kernel_size=3, padding=1), nn.ReLU(), ResidualBlock(24), ResidualBlock(24))
        self.up2 = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(24, 12, kernel_size=3, padding=1), nn.ReLU(), ResidualBlock(12), ResidualBlock(12))
        self.out_conv = nn.Conv2d(12, 3, kernel_size=1)

    def forward_from_features(self, x, skip1, skip2):
        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec3(x)
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec2(x)
        return self.out_conv(x)

# --- VAE 1: Original (Attention + Skips) ---
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, transformer_depth=1, transformer_heads=4):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder()
        self.fc = nn.Linear(latent_dim, 48 * 8 * 23)
        self.z_to_tokens = nn.Linear(latent_dim, 184 * 48)
        nn.init.xavier_uniform_(self.z_to_tokens.weight)
        nn.init.constant_(self.z_to_tokens.bias, 0)
        self.transformer_blocks = nn.ModuleList([Block(dim=48, num_heads=transformer_heads) for _ in range(transformer_depth)])        
        self.norm = nn.LayerNorm(48)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, feat = self.encoder(x)
        mu = torch.clamp(mu, min=-10, max=10)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        tokens = self.z_to_tokens(z).view(-1, 184, 48)
        for blk in self.transformer_blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        feat_transformed = tokens.transpose(1, 2).reshape(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat_transformed, self.encoder.skip1, self.encoder.skip2)
        return recon, mu, logvar

# --- Helpers for No-Skip variants ---
class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock_NoSkip(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU):
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
        self.enc1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1), nn.ReLU(inplace=True), ConvBlock(12))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1), nn.ReLU(inplace=True), ConvBlock(24))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1), nn.ReLU(inplace=True), ConvBlock(48))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(48 * 8 * 23, latent_dim)
        self.fc_logvar = nn.Linear(48 * 8 * 23, latent_dim)
    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)
        x = self.enc3(x)
        flat = self.flatten(x)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, x

class ConvDecoder_NoSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(24))
        self.up2 = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(12))
        self.out_conv = nn.Conv2d(12, 3, kernel_size=1)
    def forward_from_features(self, x):
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up2(x)
        x = self.dec2(x)
        return self.out_conv(x)

# --- VAE 2: Attention, No Skips ---
class ConvVAE_Attention_NoSkips(nn.Module):
    def __init__(self, latent_dim=128, transformer_depth=1, transformer_heads=4):
        super().__init__()
        self.encoder = ConvEncoder_NoSkips(latent_dim=latent_dim)
        self.decoder = ConvDecoder_NoSkips()
        self.z_to_tokens = nn.Linear(latent_dim, 184 * 48)
        nn.init.xavier_uniform_(self.z_to_tokens.weight)
        nn.init.constant_(self.z_to_tokens.bias, 0)
        self.transformer_blocks = nn.ModuleList([TransformerBlock_NoSkip(dim=48, num_heads=transformer_heads) for _ in range(transformer_depth)])
        self.post_norm = nn.LayerNorm(48)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        tokens = self.z_to_tokens(z).view(-1, 184, 48)
        for blk in self.transformer_blocks: tokens = blk(tokens)
        tokens = self.post_norm(tokens)
        feat = tokens.transpose(1, 2).reshape(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat)
        return recon, mu, logvar

# --- VAE 3: Skips, No Attention ---
class ConvVAE_Skip_NoAttention(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.decoder = ConvDecoder()
        self.fc = nn.Linear(latent_dim, 48 * 8 * 23)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        feat = self.fc(z).view(-1, 48, 8, 23)
        feat = feat.view(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat, self.encoder.skip1, self.encoder.skip2)
        return recon, mu, logvar

# --- Basic Blocks for Basic VAE ---
class BasicConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class BasicEncoder_NoSkips(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(12))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.stage2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(24))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.stage3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), BasicConvBlock(48))
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(48 * 8 * 23, latent_dim)
        self.fc_logvar = nn.Linear(48 * 8 * 23, latent_dim)
    def forward(self, x):
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        flat = self.flatten(x)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar, x

class BasicDecoder_NoSkips(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec1 = BasicConvBlock(24)
        self.up2 = nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2)
        self.dec2 = BasicConvBlock(12)
        self.out_conv = nn.Conv2d(12, 3, kernel_size=1)
    def forward_from_features(self, feat):
        x = self.dec1(self.up1(feat))
        x = self.dec2(self.up2(x))
        return self.out_conv(x)

# --- VAE 4: Basic (No Attention, No Skips) ---
class ConvVAE_Basic(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = BasicEncoder_NoSkips(latent_dim=latent_dim)
        self.decoder = BasicDecoder_NoSkips()
        self.fc_z_to_feat = nn.Linear(latent_dim, 48 * 8 * 23)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, _ = self.encoder(x)
        mu = torch.clamp(mu, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        feat = self.fc_z_to_feat(z).view(-1, 48, 8, 23)
        recon = self.decoder.forward_from_features(feat)
        return recon, mu, logvar

# ==========================================
# 3. EVALUATION FUNCTIONS
# ==========================================

def sliding_window_eval(model, samples_dict):
    model.eval()
    snippet_len = int(SNIPPET_SEC * SAMPLING_RATE)
    shift_len = int(SHIFT_SEC * SAMPLING_RATE)
    delta_len = int(NCC_EARLY_EXTRA_SEC * SAMPLING_RATE)

    all_results = []
    
    # Progress bar
    total_samples = len(samples_dict['sample'])
    
    with torch.no_grad():
        # tqdm for progress tracking
        iterator = tqdm(enumerate(samples_dict['sample']), total=total_samples, desc="  Evaluated", unit="rec")
        
        for i, signal in iterator:
            rec_name = samples_dict['rec_name'][i]
            p_idx = samples_dict['p_index'][i]
            
            cutoff = signal.shape[1]
            if p_idx is not None:
                cutoff = min(signal.shape[1], p_idx + snippet_len + delta_len)
            
            ncc_curve, mae_curve, starts = [], [], []
            for start in range(0, cutoff - snippet_len + 1, shift_len):
                end = start + snippet_len
                snippet = signal[:, start:end]
                
                spec_np = waveform_to_spec(snippet)
                spec_t = torch.tensor(spec_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                recon, _, _ = model(spec_t)
                recon_np = recon.squeeze(0).cpu().numpy()
                
                mae = np.mean(np.abs(spec_np - recon_np))
                mae_curve.append(mae)
                
                flat_orig = spec_np.flatten()
                flat_recon = recon_np.flatten()
                denom = np.linalg.norm(flat_orig) * np.linalg.norm(flat_recon)
                ncc = np.dot(flat_orig, flat_recon) / denom if denom > 0 else 0
                ncc_curve.append(ncc)
                starts.append(start)
                
            all_results.append({
                'rec_name': rec_name,
                'p_index': p_idx,
                'ncc_curve': np.array(ncc_curve),
                'mae_curve': np.array(mae_curve),
                'starts': np.array(starts),
                'signal': signal
            })
    return all_results

def compute_auc(results):
    y_true, y_scores = [], []
    tol_samples = int(TOLERANCE_SEC * SAMPLING_RATE)
    win_len = int(SNIPPET_SEC * SAMPLING_RATE)
    
    for res in results:
        p_idx = res['p_index']
        if p_idx is None: continue
        
        for ncc, start in zip(res['ncc_curve'], res['starts']):
            end = start + win_len
            is_pos = (abs(p_idx - start) <= tol_samples) or (start <= p_idx < end)
            y_true.append(1 if is_pos else 0)
            y_scores.append(ncc)
            
    if not y_true or sum(y_true) == 0: return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc(fpr, tpr)

def plot_tradeoff(df, save_path):
    """
    Plotter for Detection vs Reconstruction.
    """

    # Color map
    style_map = {
        'basic': {
            'label': 'Basic VAE (No Attention, No Skips)',
            'marker': 'o', 
            'color': (0, 0.45, 0.74)
        },
        'skip_no_attn': {
            'label': 'Skip VAE (No Attention)',
            'marker': 's', 
            'color': (0.85, 0.33, 0.10)
        },
        'attn_no_skips': {
            'label': 'Attention VAE (No Skips)',
            'marker': '^', 
            'color': (0.47, 0.67, 0.19)
        },
        'original': {
            'label': 'Hybrid VAE (Attention, Skips)',
            'marker': 'D', 
            'color': (0.49, 0.18, 0.56)
        }
    }

    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()

    # Data collection for axis limits
    aucs = []
    maes = []

    for idx, row in df.iterrows():
        arch = row['arch']
        auc_val = row['auc']
        mae_val = row['mae']
        
        if pd.isna(auc_val) or pd.isna(mae_val):
            continue

        aucs.append(auc_val)
        maes.append(mae_val)

        style = style_map.get(arch, {'label': arch, 'marker': 'x', 'color': 'black'})
        
        # Scatter point
        plt.scatter(
            auc_val, mae_val, 
            s=150, # Size (equivalent to msz=90 in matlab roughly)
            marker=style['marker'],
            facecolors=style['color'],
            edgecolors='k',
            linewidths=1.5,
            zorder=3
        )
        
        # Annotation text
        ha = 'right' if auc_val < 0.83 else 'left'
        x_offset = -0.003 if ha == 'right' else 0.003
        y_offset = 0.00006
        
        plt.text(
            auc_val + x_offset, mae_val + y_offset, 
            style['label'],
            horizontalalignment=ha,
            fontweight='bold',
            fontsize=10
        )

    ax.invert_yaxis() # Lower MAE is better
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xlabel(r'AUC (ROC) $\rightarrow$', fontweight='bold', fontsize=12)
    plt.ylabel(r'Reconstruction MAE $\leftarrow$ (lower is better)', fontweight='bold', fontsize=12)
    plt.title('Detection vs Reconstruction', fontweight='bold', fontsize=14)
    
    # Axis padding
    if aucs and maes:
        x_pad = 0.01
        y_pad = 0.0003
        plt.xlim(min(aucs) - x_pad, max(aucs) + x_pad)
        plt.ylim(max(maes) + y_pad, min(maes) - y_pad) # Inverted Y limits

    # Bold tick labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Trade-off plot saved to {save_path}")

# ==========================================
# 4. MAIN LOOP
# ==========================================

def parse_run_config(run_name, parent_folder):
    name_lower = run_name.lower() + "_" + parent_folder.lower()
    if "basic" in name_lower and "no_attn" in name_lower: arch = "basic"
    elif "attn_no_skip" in name_lower: arch = "attn_no_skips"
    elif "skip_no_attn" in name_lower: arch = "skip_no_attn"
    elif "hybrid" in name_lower: arch = "original"
    else: arch = "original"

    parts = run_name.split("_")
    ld, depth, heads = 128, 1, 4
    for p in parts:
        if p.startswith("ld") and p[2:].isdigit(): ld = int(p[2:])
        if p.startswith("d") and p[1:].isdigit(): depth = int(p[1:])
        if p.startswith("h") and p[1:].isdigit(): heads = int(p[1:])
    return arch, ld, depth, heads

def main():
    print(f"Starting Evaluation Pipeline...")
    full_data = read_hdf5(DATA_PATH, SAMPLING_RATE, EXPECTED_SECONDS)
    exp_folders = [f for f in glob.glob(os.path.join(MODELS_DIR, "*", "*")) if os.path.isdir(f)]
    # Handle both depth structures
    if not exp_folders:
        exp_folders = [f for f in glob.glob(os.path.join(MODELS_DIR, "*")) if os.path.isdir(f)]

    print(f"Found {len(exp_folders)} folders in {MODELS_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_summary = []

    for exp_dir in exp_folders:
        run_name = os.path.basename(exp_dir)
        parent = os.path.basename(os.path.dirname(exp_dir))
        
        model_pt = os.path.join(exp_dir, "model.pt")
        test_json = os.path.join(exp_dir, "test_p_recordings.json")
        if not os.path.exists(model_pt): continue
        
        print(f"\nProcessing: {run_name}")
        arch, ld, depth, heads = parse_run_config(run_name, parent)
        print(f"  Arch: {arch} | LD: {ld} | Depth: {depth} | Heads: {heads}")

        # Instantiate specific model
        if arch == "original":
            model = ConvVAE(latent_dim=ld, transformer_depth=depth, transformer_heads=heads)
        elif arch == "skip_no_attn":
            model = ConvVAE_Skip_NoAttention(latent_dim=ld)
        elif arch == "attn_no_skips":
            model = ConvVAE_Attention_NoSkips(latent_dim=ld, transformer_depth=depth, transformer_heads=heads)
        elif arch == "basic":
            model = ConvVAE_Basic(latent_dim=ld)
        
        model.to(DEVICE)
        
        try:
            state_dict = torch.load(model_pt, map_location=DEVICE, weights_only=False)
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue

        if not os.path.exists(test_json):
            print("  [WARN] Missing test_p_recordings.json")
            continue
            
        with open(test_json, 'r') as f: test_recs = json.load(f)
        test_data = subset_samples(full_data, test_recs)
        if not test_data['rec_name']: continue
        
        eval_results = sliding_window_eval(model, test_data)
        auc_score = compute_auc(eval_results)
        all_maes = np.concatenate([r['mae_curve'] for r in eval_results if len(r['mae_curve']) > 0])
        mean_mae = np.mean(all_maes) if len(all_maes) > 0 else float('nan')
        
        print(f"  -> AUC: {auc_score:.4f} | MAE: {mean_mae:.6f}")
        results_summary.append({'run': run_name, 'arch': arch, 'auc': auc_score, 'mae': mean_mae})
        
    if results_summary:
        df = pd.DataFrame(results_summary)
        csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nEvaluation Complete. Summary saved to {csv_path}")
        
        # Generate the trade-off plot
        plot_tradeoff(df, os.path.join(OUTPUT_DIR, "01_recon_det_tradeoff.png"))
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    main()