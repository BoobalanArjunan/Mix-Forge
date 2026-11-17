import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

# -------------------------------------------------------------
# PAPER 1: Architecture Figure (7 Tools)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.0)
ax.axis('off')

# Background canvas
bg = patches.Rectangle((0, 0), 14, 7.0, facecolor="#F8FAFC", edgecolor="none", zorder=0)
ax.add_patch(bg)

# Header
ax.text(7.0, 6.65, "OpenSource AudioIntegrity Suite: 7 Quality Assurance Tools", 
        ha='center', va='center', fontsize=13, fontweight='bold', color="#0F172A")
ax.text(7.0, 6.35, "Pre-Training Dataset Hygiene • Physical DSP Invariants • Deterministic Gates • Commodity CPU Execution", 
        ha='center', va='center', fontsize=9, fontstyle='italic', color="#475569")

# Top raw input
top_box = patches.FancyBboxPatch((2.5, 5.6), 9.0, 0.55, boxstyle="round,pad=0.06,rounding_size=0.1",
                                facecolor="#1E293B", edgecolor="#0F172A", lw=1.5, zorder=2)
ax.add_patch(top_box)
ax.text(7.0, 5.87, "Raw Audio Training Corpora (LibriSpeech, Common Voice, Web Audio, Music Stems)", 
        ha='center', va='center', fontsize=9.5, fontweight='bold', color="#FFFFFF", zorder=3)

# Arrows
ax.annotate('', xy=(3.5, 5.1), xytext=(5.0, 5.6), arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=15))
ax.annotate('', xy=(10.5, 5.1), xytext=(9.0, 5.6), arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.8, mutation_scale=15))

# Group 1: Dataset & Signal Quality (3 tools)
g1_box = patches.FancyBboxPatch((0.5, 1.2), 6.2, 3.8, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#E0F2FE", edgecolor="#38BDF8", lw=1.5, zorder=1)
ax.add_patch(g1_box)
h1_box = patches.FancyBboxPatch((0.6, 4.5), 6.0, 0.4, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#0284C7", edgecolor="none", zorder=2)
ax.add_patch(h1_box)
ax.text(3.6, 4.7, "GROUP I: SIGNAL QUALITY & PARTITION HYGIENE", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#FFFFFF", zorder=3)

tools_g1 = [
    ("WaveWash", "Cubic Hermite Spline De-Clipping & BS.1770", "71.8% MSE drop on clipped speech; DC removal; zero false flags"),
    ("SplitSafe", "Landmark Peak Hashing & Time Voting", "Audited 5,323 LibriSpeech files; detected 56/56 planted leaks (0 FPR)"),
    ("BandCheck", "Log-Spectral Slope Cliff Derivative", "100% brickwall detection on 185 tracks; 85.7% recall on edge suite")
]
y = 4.05
for name, algo, desc in tools_g1:
    c = patches.FancyBboxPatch((0.7, y-0.75), 5.8, 0.82, boxstyle="round,pad=0.04,rounding_size=0.08",
                              facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=1.0, zorder=2)
    ax.add_patch(c)
    ax.text(0.9, y-0.12, name, fontsize=8.5, fontweight='bold', color="#0284C7", zorder=3)
    ax.text(0.9, y-0.35, algo, fontsize=7.2, fontweight='semibold', color="#1E293B", zorder=3)
    ax.text(0.9, y-0.58, desc, fontsize=6.5, color="#64748B", zorder=3)
    y -= 0.95

# Group 2: Harmonic, Temporal & Spatial Integrity (4 tools)
g2_box = patches.FancyBboxPatch((7.3, 1.2), 6.2, 3.8, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#EDE9FE", edgecolor="#A78BFA", lw=1.5, zorder=1)
ax.add_patch(g2_box)
h2_box = patches.FancyBboxPatch((7.4, 4.5), 6.0, 0.4, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#7C3AED", edgecolor="none", zorder=2)
ax.add_patch(h2_box)
ax.text(10.4, 4.7, "GROUP II: HARMONIC, TEMPORAL & SPATIAL INTEGRITY", ha='center', va='center', fontsize=8.5, fontweight='bold', color="#FFFFFF", zorder=3)

tools_g2 = [
    ("TempoCheck", "Onset Flux Autocorrelation Tempogram", "100% Acc2 harmonic tempo validation; flags metric doubling/halving"),
    ("TonicFind", "12-Bin Chroma & Krumhansl-Kessler Profiles", "100% key accuracy on 25 stems; A440 detune comp (+-50 cents)"),
    ("PhaseCheck", "Sliding Pearson Phase & Mono Downmix", "100% recall on 180° polarity flips; flags mono phase drop (< -3 dB)"),
    ("DriftCheck", "Band Energy Cross-Correlation & Permutation", "Catches clock drift (ms/min) & frame mismatch; p < 0.01")
]
y = 4.05
for name, algo, desc in tools_g2:
    c = patches.FancyBboxPatch((7.5, y-0.55), 5.8, 0.62, boxstyle="round,pad=0.03,rounding_size=0.07",
                              facecolor="#FFFFFF", edgecolor="#DDD6FE", lw=0.9, zorder=2)
    ax.add_patch(c)
    ax.text(7.7, y-0.08, name, fontsize=8.0, fontweight='bold', color="#7C3AED", zorder=3)
    ax.text(7.7, y-0.28, algo, fontsize=6.8, fontweight='semibold', color="#1E293B", zorder=3)
    ax.text(7.7, y-0.45, desc, fontsize=6.2, color="#64748B", zorder=3)
    y -= 0.72

# Bottom Output
out_box = patches.FancyBboxPatch((2.0, 0.25), 10.0, 0.55, boxstyle="round,pad=0.06,rounding_size=0.1",
                                 facecolor="#0F172A", edgecolor="#334155", lw=1.5, zorder=2)
ax.add_patch(out_box)
ax.text(7.0, 0.52, "Clean, Audited PyTorch DataLoaders & Quality-Gated CI/CD Workflows", 
        ha='center', va='center', fontsize=9.2, fontweight='bold', color="#38BDF8", zorder=3)

ax.annotate('', xy=(4.5, 0.8), xytext=(3.6, 1.2), arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.6))
ax.annotate('', xy=(9.5, 0.8), xytext=(10.4, 1.2), arrowprops=dict(arrowstyle="-|>", color="#64748B", lw=1.6))

plt.tight_layout()
plt.savefig("paper/paper1_dataset_integrity/figures/paper1_architecture.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/paper1_dataset_integrity/figures/paper1_architecture.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved paper1_architecture.pdf & .png")

# -------------------------------------------------------------
# PAPER 2: Architecture & Bleed Matrix Figure (3 Tools)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.5)
ax.axis('off')

bg = patches.Rectangle((0, 0), 12, 6.5, facecolor="#F8FAFC", edgecolor="none", zorder=0)
ax.add_patch(bg)

ax.text(6.0, 6.15, "OpenSource AudioEvaluation Suite: 3 Diagnostic Benchmarking Tools", 
        ha='center', va='center', fontsize=12.5, fontweight='bold', color="#0F172A")
ax.text(6.0, 5.85, "Reproducible Ground Truth • Subspace Bleed Decomposition • Full-Duplex Conversational Dynamics", 
        ha='center', va='center', fontsize=8.8, fontstyle='italic', color="#475569")

# Left Column: Source Separation Ground Truth (MixForge & StemCheck)
b_left = patches.FancyBboxPatch((0.5, 0.8), 5.3, 4.6, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor="#D1FAE5", edgecolor="#34D399", lw=1.5, zorder=1)
ax.add_patch(b_left)
h_left = patches.FancyBboxPatch((0.6, 4.9), 5.1, 0.4, boxstyle="round,pad=0.04,rounding_size=0.08",
                                facecolor="#059669", edgecolor="none", zorder=2)
ax.add_patch(h_left)
ax.text(3.15, 5.1, "TRACK A: SOURCE SEPARATION GROUND TRUTH", ha='center', va='center', fontsize=8.2, fontweight='bold', color="#FFFFFF", zorder=3)

# Tool 1: MixForge
c1 = patches.FancyBboxPatch((0.7, 3.2), 4.9, 1.45, boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=1.0, zorder=2)
ax.add_patch(c1)
ax.text(0.9, 4.35, "MixForge-Audio", fontsize=9.0, fontweight='bold', color="#059669", zorder=3)
ax.text(0.9, 4.08, "Exact Linear Superposition & Calibrated Bleed", fontsize=7.2, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(0.9, 3.75, "• Mix = Sum(Stems) with double error < 10^-12\n• Butterworth off-axis acoustic bleed modeling\n• Cryptographic tamper-evident recipe.json", fontsize=6.5, color="#475569", zorder=3)

# Tool 2: StemCheck
c2 = patches.FancyBboxPatch((0.7, 1.4), 4.9, 1.55, boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=1.0, zorder=2)
ax.add_patch(c2)
ax.text(0.9, 2.65, "StemCheck", fontsize=9.0, fontweight='bold', color="#059669", zorder=3)
ax.text(0.9, 2.38, "Cross-Stem Bleed Matrix Decomposition", fontsize=7.2, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(0.9, 2.05, "• Least-squares projection (S^T S)^-1 S^T s_hat\n• Separates physical bleed (dB) from artifact noise\n• Blind sanity checks: conservation, clipping, DC", fontsize=6.5, color="#475569", zorder=3)

# Right Column: Conversational Voice AI (TurnScope)
b_right = patches.FancyBboxPatch((6.2, 0.8), 5.3, 4.6, boxstyle="round,pad=0.08,rounding_size=0.12",
                                 facecolor="#FEF3C7", edgecolor="#FBBF24", lw=1.5, zorder=1)
ax.add_patch(b_right)
h_right = patches.FancyBboxPatch((6.3, 4.9), 5.1, 0.4, boxstyle="round,pad=0.04,rounding_size=0.08",
                                 facecolor="#D97706", edgecolor="none", zorder=2)
ax.add_patch(h_right)
ax.text(8.85, 5.1, "TRACK B: FULL-DUPLEX CONVERSATIONAL FLOW", ha='center', va='center', fontsize=8.2, fontweight='bold', color="#FFFFFF", zorder=3)

# Tool 3: TurnScope
c3 = patches.FancyBboxPatch((6.4, 1.4), 4.9, 3.25, boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor="#FFFFFF", edgecolor="#FDE68A", lw=1.0, zorder=2)
ax.add_patch(c3)
ax.text(6.6, 4.35, "TurnScope", fontsize=9.5, fontweight='bold', color="#D97706", zorder=3)
ax.text(6.6, 4.05, "Dual-Track Energy State Machine & Tail Latency", fontsize=7.5, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(6.6, 3.45, "• 18 dB Inter-Channel Dominance Filter:\n   Suppresses acoustic echo & loudspeaker crosstalk\n• High-Precision Turn-Taking Metrics:\n   - Hand-over latency (P50, P90, P99 percentiles)\n   - Customer barge-in rate & response yield\n   - Agent interruptions & awkward dead air (>= 3s)\n• Evaluated on MagicHub Full-Duplex Corpus:\n   67.1% reduction in P90 tail latency error", fontsize=6.8, color="#475569", zorder=3)

# Bottom note
ax.text(6.0, 0.35, "Target Venues: DAFx / AES / ISMIR • Evaluated on MUSDB18-HQ, Demucs v4 & MagicHub", 
        ha='center', va='center', fontsize=8.0, fontweight='bold', color="#475569")

plt.tight_layout()
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved paper2_architecture.pdf & .png")

# Also copy turnscope ECDF into paper 2 figures
import shutil
shutil.copy("paper/figures/turnscope_latency_ecdf.pdf", "paper/paper2_audio_evaluation/figures/turnscope_latency_ecdf.pdf")
shutil.copy("paper/figures/turnscope_latency_ecdf.png", "paper/paper2_audio_evaluation/figures/turnscope_latency_ecdf.png")
shutil.copy("paper/figures/ablation_sweeps.pdf", "paper/paper1_dataset_integrity/figures/paper1_benchmarks.pdf")
shutil.copy("paper/figures/ablation_sweeps.png", "paper/paper1_dataset_integrity/figures/paper1_benchmarks.png")
print("All figures successfully organized into paper1 and paper2 folders.")
