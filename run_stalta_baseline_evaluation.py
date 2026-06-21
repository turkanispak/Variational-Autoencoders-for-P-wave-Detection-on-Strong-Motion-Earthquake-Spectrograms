"""
STA/LTA baseline evaluation, as an external non-learning benchmark for the VAEs.

STA/LTA settings follow Zhu, Mousavi & Beroza (2019), "Seismic Signal Denoising
and Decomposition Using Deep Neural Networks", IEEE TGRS 57(11):9476-9488
(arXiv:1811.02695), Sec. D: classic CFT with short/long windows of 0.5 s / 5 s.

Run AFTER run_vae_evaluation.py. This script reads the VAE results from
results_summary.csv and writes its own results back into the same files:
    - results_summary.csv          appends "STA/LTA detection" + "STA/LTA SNR bins"
    - 01_recon_det_tradeoff.png     overwritten with the STA/LTA baseline line added
    - auc_vs_snr.png                overwritten with STA/LTA overlaid on the VAE curve
"""

import os
import json
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from run_vae_evaluation import read_hdf5, subset_samples

# ==========================================================================
# CONFIG  -- edit 
# ==========================================================================
DATA_PATH = "dataset.hdf5"
TEST_JSON = "experiments/test_p_recordings.json"   # same split used by the VAE

NSTA_SEC = 0.5     # short-term window (Zhu, Mousavi & Beroza 2019)
NLTA_SEC = 5.0     # long-term window
NOISE_SEC = 1.0    # pre-P noise window for SNR  (match script 1)
SIG_SEC   = 1.44   # post-P signal window for SNR (match script 1)
N_BINS    = 4      # SNR quantile bins

OUT_DIR = "eval_outputs"
# ==========================================================================

SAMPLING_RATE       = 100
SNIPPET_SEC         = 2.44
SHIFT_SEC           = 0.1
NCC_EARLY_EXTRA_SEC = 2.44

CSV_PATH = os.path.join(OUT_DIR, "results_summary.csv")

# --- classic STA/LTA backend (ObsPy if available, NumPy fallback otherwise) ---
try:
    from obspy.signal.trigger import classic_sta_lta
    _HAVE_OBSPY = True
except Exception:
    _HAVE_OBSPY = False

    def classic_sta_lta(a, nsta, nlta):
        a = np.asarray(a, dtype=np.float64); sq = a ** 2
        sta = np.full(len(a), np.nan); lta = np.full(len(a), np.nan)
        for i in range(len(a)):
            if i >= nsta:
                sta[i] = sq[i - nsta:i].mean()
            if i >= nlta:
                lta[i] = sq[i - nlta:i].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            cft = sta / lta
        cft[~np.isfinite(cft)] = 0.0
        return cft


# ==========================================================================
# STA/LTA scoring
# ==========================================================================
def sta_lta_cft(signal, nsta_samp, nlta_samp):
    trace = np.sqrt(np.mean(signal ** 2, axis=0))          # RMS across 3 channels
    cft = classic_sta_lta(trace, int(nsta_samp), int(nlta_samp))
    if nlta_samp < len(cft):
        cft[: int(nlta_samp)] = 1.0                        # neutralize LTA ramp-up
    return cft


def build_window_scores(samples_dict, nsta_samp, nlta_samp):
    snippet_len = int(SNIPPET_SEC * SAMPLING_RATE)
    shift_len = int(SHIFT_SEC * SAMPLING_RATE)
    delta_len = int(NCC_EARLY_EXTRA_SEC * SAMPLING_RATE)
    results = []
    for i, signal in enumerate(samples_dict["sample"]):
        p_idx = samples_dict["p_index"][i]
        cft = sta_lta_cft(signal, nsta_samp, nlta_samp)
        cutoff = signal.shape[1]
        if p_idx is not None:
            cutoff = min(signal.shape[1], p_idx + snippet_len + delta_len)
        scores, starts = [], []
        for start in range(0, cutoff - snippet_len + 1, shift_len):
            end = start + snippet_len
            scores.append(float(np.max(cft[start:end])) if end <= len(cft) else 0.0)
            starts.append(start)
        results.append({
            "rec_name": samples_dict["rec_name"][i],
            "p_index": p_idx,
            "score_curve": np.array(scores),   # per-window max characteristic function
            "starts": np.array(starts),
        })
    return results


def compute_auc(results):
    """Strict window-containment labeling (no tolerance); raw-waveform detector."""
    y_true, y_scores = [], []
    win_len = int(SNIPPET_SEC * SAMPLING_RATE)
    for res in results:
        p_idx = res["p_index"]
        if p_idx is None:
            continue
        for score, start in zip(res["score_curve"], res["starts"]):
            end = start + win_len
            y_true.append(1 if (start <= p_idx < end) else 0)
            y_scores.append(score)
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
# results_summary.csv 
# ==========================================================================
def read_section(title):
    if not os.path.exists(CSV_PATH):
        return None
    lines = open(CSV_PATH).read().splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.strip() == f"# {title}"), None)
    if start is None:
        return None
    block = []
    for ln in lines[start:]:
        if ln.strip() == "" or ln.startswith("#"):
            break
        block.append(ln)
    return pd.read_csv(StringIO("\n".join(block))) if block else None


def append_section(title, df):
    with open(CSV_PATH, "a", newline="") as f:
        f.write(f"\n# {title}\n")
    df.to_csv(CSV_PATH, mode="a", index=False)


# ==========================================================================
# Figures
# ==========================================================================
def plot_tradeoff(vae_df, sta_auc, save_path):
    style = {
        "basic":         {"label": "Basic VAE (No Attention, No Skips)", "marker": "o", "color": (0, 0.45, 0.74)},
        "skip_no_attn":  {"label": "Skip VAE (No Attention)",            "marker": "s", "color": (0.85, 0.33, 0.10)},
        "attn_no_skips": {"label": "Attention VAE (No Skips)",           "marker": "^", "color": (0.47, 0.67, 0.19)},
        "original":      {"label": "Hybrid VAE (Attention, Skips)",      "marker": "D", "color": (0.49, 0.18, 0.56)},
    }
    plt.figure(figsize=(8, 6), dpi=300)
    ax = plt.gca()
    aucs, maes = [], []
    for _, row in vae_df.iterrows():
        if pd.isna(row["auc"]) or pd.isna(row["mae"]):
            continue
        aucs.append(row["auc"]); maes.append(row["mae"])
        s = style.get(row["arch"], {"label": row["arch"], "marker": "x", "color": "black"})
        plt.scatter(row["auc"], row["mae"], s=150, marker=s["marker"],
                    facecolors=s["color"], edgecolors="k", linewidths=1.5, zorder=3)
        ha = "right" if row["auc"] < 0.83 else "left"
        plt.text(row["auc"] + (-0.003 if ha == "right" else 0.003), row["mae"] + 0.00006,
                 s["label"], horizontalalignment=ha, fontweight="bold", fontsize=10)
    # STA/LTA baseline: detection-only -> vertical reference line
    plt.axvline(sta_auc, color=(0.85, 0.10, 0.10), linestyle="--", lw=2, zorder=2)
    if maes:
        plt.text(sta_auc - 0.003, (max(maes) + min(maes)) / 2,
                 f"STA/LTA Baseline (AUC = {sta_auc:.3f})", color=(0.85, 0.10, 0.10),
                 rotation=90, va="center", ha="right", fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    plt.grid(True, linestyle="-", linewidth=0.5)
    plt.xlabel(r"AUC (ROC) $\rightarrow$", fontweight="bold", fontsize=12)
    plt.ylabel(r"Reconstruction MAE $\leftarrow$ (lower is better)", fontweight="bold", fontsize=12)
    plt.title("Detection vs Reconstruction", fontweight="bold", fontsize=14)
    if aucs and maes:
        plt.xlim(min(min(aucs), sta_auc) - 0.01, max(max(aucs), sta_auc) + 0.01)
        plt.ylim(max(maes) + 0.0003, min(maes) - 0.0003)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Trade-off plot updated: {save_path}")


def plot_snr(sta_df, vae_df, save_path):
    sta_df = sta_df[sta_df["n"] > 0].reset_index(drop=True)
    x = np.arange(len(sta_df))
    plt.figure(figsize=(8, 6), dpi=300)
    if vae_df is not None:
        v = vae_df[vae_df["n"] > 0].reset_index(drop=True)
        plt.plot(x, v["auc"], "-o", color=(0, 0.45, 0.74), lw=2, markersize=10,
                 markerfacecolor=(0, 0.45, 0.74), markeredgecolor="k", label="Attention-VAE")
    plt.plot(x, sta_df["auc"], "--s", color=(0.85, 0.33, 0.10), lw=2, markersize=9,
             markerfacecolor=(0.85, 0.33, 0.10), markeredgecolor="k", label="STA/LTA")
    for xi, row in zip(x, sta_df.itertuples()):
        ytop = row.auc
        if vae_df is not None:
            vv = vae_df[vae_df["n"] > 0].reset_index(drop=True)
            if xi < len(vv):
                ytop = max(ytop, vv.loc[xi, "auc"])
        plt.annotate(f"n={int(row.n)}", (xi, ytop), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontweight="bold")
    plt.xticks(x, sta_df["snr_range"], rotation=15)
    plt.ylim(0.5, 1.0)
    plt.ylabel("ROC AUC Score", fontweight="bold", fontsize=12)
    plt.xlabel("SNR Range (dB)", fontweight="bold", fontsize=12)
    plt.title("Detection Performance vs SNR", fontweight="bold", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, lw=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  SNR plot updated: {save_path}")


# ==========================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nsta_samp = int(round(NSTA_SEC * SAMPLING_RATE))
    nlta_samp = int(round(NLTA_SEC * SAMPLING_RATE))
    print(f"STA/LTA baseline (classic)  |  obspy={'yes' if _HAVE_OBSPY else 'NumPy fallback'}")
    print(f"  nsta={NSTA_SEC}s  nlta={NLTA_SEC}s  combine=rms")

    full = read_hdf5(DATA_PATH, SAMPLING_RATE, 30)
    with open(TEST_JSON) as f:
        data = subset_samples(full, json.load(f))
    print(f"  Evaluating on {len(data['rec_name'])} records.")

    results = build_window_scores(data, nsta_samp, nlta_samp)
    sta_map = {r["rec_name"]: r for r in results}

    # --- overall detection AUC -> append to results_summary.csv ---
    overall = compute_auc(results)
    print(f"\n  STA/LTA overall window-AUC = {overall:.4f}")
    append_section("STA/LTA detection", pd.DataFrame(
        [{"method": "STA/LTA", "auc": round(overall, 3), "n_records": len(data["rec_name"])}]))

    # --- per-SNR-bin AUC -> append to results_summary.csv ---
    snr = np.array([compute_snr_db(data["sample"][i], data["p_index"][i], NOISE_SEC, SIG_SEC)
                    for i in range(len(data["sample"]))])
    bin_idx, labels = assign_bins(snr, N_BINS)
    per_rec = pd.DataFrame({"rec_name": data["rec_name"], "bin": bin_idx})
    aucs = per_bin_auc(sta_map, per_rec, len(labels))
    sta_bins = pd.DataFrame([
        {"bin": b, "snr_range": labels[b], "n": int((per_rec["bin"] == b).sum()),
         "auc": round(aucs[b], 3)} for b in range(len(labels))])
    append_section("STA/LTA SNR bins", sta_bins)
    for r in sta_bins.itertuples():
        print(f"  Bin {r.bin} [{r.snr_range}]: n={r.n:3d}  STA/LTA AUC={r.auc:.3f}")

    # --- update figures using the VAE sections ---
    vae_det = read_section("VAE detection summary")
    if vae_det is not None:
        plot_tradeoff(vae_det, overall, os.path.join(OUT_DIR, "01_recon_det_tradeoff.png"))
    else:
        print("  [skip] trade-off plot: VAE detection summary not found in CSV.")

    vae_bins = read_section("Attention-VAE SNR bins")
    plot_snr(sta_bins, vae_bins, os.path.join(OUT_DIR, "auc_vs_snr.png"))
    if vae_bins is None:
        print("  (note: VAE SNR bins not found -> STA/LTA-only SNR curve)")

    print(f"\n  Appended to {CSV_PATH} and updated figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()