import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

fig, ax = plt.subplots(figsize=(11.5, 4.9), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.25)
ax.axis("off")

# Background
fig.patch.set_facecolor("#FFFFFF")
ax.set_facecolor("#FFFFFF")

# Header Title Box
title_box = patches.FancyBboxPatch((0.5, 4.52), 11.0, 0.65, boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor="#0F172A", edgecolor="none", zorder=2)
ax.add_patch(title_box)
ax.text(6.0, 4.92, "OpenSource AudioEvaluation Suite: Diagnostic Benchmarking Architecture",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#FFFFFF", zorder=3)
ax.text(6.0, 4.66, "Model Evaluation • Decoupled Bleed Diagnostics • Conversational Tail Latency",
        ha="center", va="center", fontsize=8.5, color="#94A3B8", zorder=3)

# ----------------- TRACK A: Source Separation Ground Truth -----------------
track_a_bg = patches.FancyBboxPatch((0.5, 0.58), 5.35, 3.82, boxstyle="round,pad=0.04,rounding_size=0.1",
                                   facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.2, zorder=1)
ax.add_patch(track_a_bg)

# Track A Header
header_a = patches.FancyBboxPatch((0.65, 3.95), 5.05, 0.35, boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor="#15803D", edgecolor="none", zorder=2)
ax.add_patch(header_a)
ax.text(3.17, 4.12, "TRACK A: SOURCE SEPARATION GROUND TRUTH", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF", zorder=3)

# Tool 1: mixforge-audio
card_mf = patches.FancyBboxPatch((0.68, 2.40), 4.99, 1.42, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=1.0, zorder=2)
ax.add_patch(card_mf)
ax.text(0.88, 3.56, "mixforge-audio", fontsize=10.0, fontweight="bold", color="#15803D", zorder=3)
ax.text(2.65, 3.56, "[Linear Synthesis Engine]", fontsize=8.2, color="#64748B", zorder=3)
ax.text(0.88, 3.26, "• Exact Linear Superposition: Mix = \u03a3 Stems", fontsize=8.2, fontweight="semibold", color="#1E293B", zorder=3)
ax.text(1.08, 3.02, "Float64 residual < 10\u207b\u00b9\u00b2  |  Float32 container peak \u2264 6.52 \u00d7 10\u207b\u2078", fontsize=7.6, fontweight="medium", color="#334155", zorder=3)
ax.text(0.88, 2.76, "• Calibrated Bleed: 4th-order Butterworth low-pass simulation", fontsize=7.8, color="#1E293B", zorder=3)
ax.text(0.88, 2.52, "• Recipe Serialization: Cryptographic SHA-256 recipe.json manifest", fontsize=7.8, color="#1E293B", zorder=3)

# Tool 2: stemcheck
card_sc = patches.FancyBboxPatch((0.68, 0.72), 4.99, 1.54, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=1.0, zorder=2)
ax.add_patch(card_sc)
ax.text(0.88, 2.02, "stemcheck", fontsize=10.0, fontweight="bold", color="#15803D", zorder=3)
ax.text(2.18, 2.02, "[Subspace Bleed Diagnostic]", fontsize=8.2, color="#64748B", zorder=3)
ax.text(0.88, 1.72, "• Subspace Projection:  c = (S\u1d40S + \u03bbI)\u207b\u00b9 S\u1d40 \u015d", fontsize=8.4, fontweight="bold", color="#0F172A", zorder=3)
ax.text(1.08, 1.48, "Decouples cross-stem interference L_ij (dB) from residual error U_i", fontsize=7.6, color="#334155", zorder=3)
ax.text(0.88, 1.24, "• Bleed Validation: Recovers broadband injected bleed (\u2264 0.06 dB error)", fontsize=7.8, color="#1E293B", zorder=3)
ax.text(0.88, 1.00, "• Verification: Mixture reconstruction consistency, clipping, DC, duplicates", fontsize=7.8, color="#1E293B", zorder=3)

# ----------------- TRACK B: Conversational Voice AI -----------------
track_b_bg = patches.FancyBboxPatch((6.15, 0.58), 5.35, 3.82, boxstyle="round,pad=0.04,rounding_size=0.1",
                                   facecolor="#FFFBEB", edgecolor="#FCD34D", lw=1.2, zorder=1)
ax.add_patch(track_b_bg)

# Track B Header
header_b = patches.FancyBboxPatch((6.30, 3.95), 5.05, 0.35, boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor="#B45309", edgecolor="none", zorder=2)
ax.add_patch(header_b)
ax.text(8.82, 4.12, "TRACK B: FULL-DUPLEX CONVERSATIONAL DYNAMICS", 
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF", zorder=3)

# Tool 3: turnscope
card_ts = patches.FancyBboxPatch((6.30, 0.72), 5.05, 3.10, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#FDE68A", lw=1.0, zorder=2)
ax.add_patch(card_ts)
ax.text(6.48, 3.56, "turnscope", fontsize=10.0, fontweight="bold", color="#B45309", zorder=3)
ax.text(7.68, 3.56, "[Echo Suppression & Turn Latency Engine]", fontsize=8.2, color="#64748B", zorder=3)

ax.text(6.48, 3.24, "• 18 dB Energy-Dominance Heuristic:", fontsize=8.0, fontweight="bold", color="#1E293B", zorder=3)
ax.text(6.68, 3.02, "Suppresses acoustic loudspeaker-to-mic echo crosstalk", fontsize=7.6, color="#334155", zorder=3)
ax.text(6.68, 2.80, "Suppresses phantom turns; raises detection F1 to 0.897", fontsize=7.6, color="#334155", zorder=3)

ax.text(6.48, 2.48, "• Asynchronous State Machine:", fontsize=8.0, fontweight="bold", color="#1E293B", zorder=3)
ax.text(6.68, 2.26, "Transitions: Silence \u2192 Agent \u2192 Customer \u2192 Barge-In \u2192 Dead Air", fontsize=7.6, color="#334155", zorder=3)

ax.text(6.48, 1.94, "• Conversational Tail Latency & Barge-In Metrics:", fontsize=8.0, fontweight="bold", color="#1E293B", zorder=3)
ax.text(6.68, 1.72, "Tracks latency quantiles (P50, P90, P99) via empirical ECDF", fontsize=7.6, color="#334155", zorder=3)
ax.text(6.68, 1.50, "Measures customer barge-in latency & dead air (\u2265 3.0s)", fontsize=7.6, color="#334155", zorder=3)

ax.text(6.48, 1.18, "• Real-World Validation (MagicData-RAMC Corpus):", fontsize=8.0, fontweight="bold", color="#1E293B", zorder=3)
ax.text(6.68, 0.94, "Reduces P90 latency error by 67.1% (1.34s \u2192 0.44s)", fontsize=7.8, fontweight="bold", color="#047857", zorder=3)

# Footer bar
footer = patches.FancyBboxPatch((0.5, 0.1), 11.0, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8, zorder=1)
ax.add_patch(footer)
ax.text(6.0, 0.28, "Suite Benchmark Hub: https://github.com/BoobalanArjunan/OpenSource-AudioEvaluation-Suite • MIT License",
        ha="center", va="center", fontsize=8.0, fontweight="medium", color="#334155", zorder=2)

plt.tight_layout()
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.pdf", format="pdf", bbox_inches="tight")
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.png", format="png", dpi=300, bbox_inches="tight")
shutil.copy("paper/paper2_audio_evaluation/figures/paper2_architecture.pdf", "paper/paper2_audio_evaluation/paper2_architecture.pdf")
shutil.copy("paper/paper2_audio_evaluation/figures/paper2_architecture.png", "paper/paper2_audio_evaluation/paper2_architecture.png")
plt.close()
