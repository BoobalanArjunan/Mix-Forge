import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

# Set up figure
fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.2)
ax.axis('off')

# Background
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Header Title Box
title_box = patches.FancyBboxPatch((0.5, 4.45), 11.0, 0.65, boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor="#0F172A", edgecolor="none", zorder=2)
ax.add_patch(title_box)
ax.text(6.0, 4.85, "OpenSource AudioEvaluation Suite: Diagnostic Benchmarking Architecture",
        ha='center', va='center', fontsize=11.5, fontweight='bold', color="#FFFFFF", zorder=3)
ax.text(6.0, 4.58, "Model Evaluation • Decoupled Bleed Diagnostics • Conversational Tail Latency",
        ha='center', va='center', fontsize=8.5, color="#94A3B8", zorder=3)

# ----------------- TRACK A: Source Separation Ground Truth -----------------
track_a_bg = patches.FancyBboxPatch((0.5, 0.6), 5.35, 3.7, boxstyle="round,pad=0.04,rounding_size=0.1",
                                   facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.2, zorder=1)
ax.add_patch(track_a_bg)

# Track A Header
header_a = patches.FancyBboxPatch((0.65, 3.85), 5.05, 0.35, boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor="#15803D", edgecolor="none", zorder=2)
ax.add_patch(header_a)
ax.text(3.17, 4.02, "TRACK A: SOURCE SEPARATION GROUND TRUTH", 
        ha='center', va='center', fontsize=8.5, fontweight='bold', color="#FFFFFF", zorder=3)

# Tool 1: mixforge-audio
card_mf = patches.FancyBboxPatch((0.7, 2.35), 4.95, 1.35, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=1.0, zorder=2)
ax.add_patch(card_mf)
ax.text(0.9, 3.45, "mixforge-audio", fontsize=10.0, fontweight='bold', color="#15803D", zorder=3)
ax.text(2.6, 3.45, "[Linear Synthesis Engine]", fontsize=8.0, color="#64748B", zorder=3)
ax.text(0.9, 3.15, "• Exact Linear Superposition: Mix = \u03a3 Stems", fontsize=7.8, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(1.1, 2.92, "Float64 error < 10\u207b\u00b9\u00b2; Float32 container peak \u2264 6.52 \u00d7 10\u207b\u2078", fontsize=7.2, color="#475569", zorder=3)
ax.text(0.9, 2.68, "• Calibrated Bleed: 4th-order Butterworth low-pass simulation", fontsize=7.8, color="#1E293B", zorder=3)
ax.text(0.9, 2.45, "• Reproducibility: Cryptographic SHA-256 recipe.json serialization", fontsize=7.8, color="#1E293B", zorder=3)

# Tool 2: stemcheck
card_sc = patches.FancyBboxPatch((0.7, 0.8), 4.95, 1.45, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#BBF7D0", lw=1.0, zorder=2)
ax.add_patch(card_sc)
ax.text(0.9, 2.0, "stemcheck", fontsize=10.0, fontweight='bold', color="#15803D", zorder=3)
ax.text(2.1, 2.0, "[Subspace Bleed Diagnostic]", fontsize=8.0, color="#64748B", zorder=3)
ax.text(0.9, 1.72, "• Subspace Projection: c = (S\u1d40S)\u207b\u00b9 S\u1d40 \u015d", fontsize=7.8, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(1.1, 1.50, "Decouples cross-stem interference L_ij (dB) from artifacts U_i", fontsize=7.2, color="#475569", zorder=3)
ax.text(0.9, 1.28, "• Bleed Validation: Accurately recovers known bleed (\u2264 0.06 dB error)", fontsize=7.8, color="#1E293B", zorder=3)
ax.text(0.9, 1.05, "• Physical Invariants: Conservation sum, clipping, DC, duplicates", fontsize=7.8, color="#1E293B", zorder=3)

# ----------------- TRACK B: Conversational Voice AI -----------------
track_b_bg = patches.FancyBboxPatch((6.15, 0.6), 5.35, 3.7, boxstyle="round,pad=0.04,rounding_size=0.1",
                                   facecolor="#FFFBEB", edgecolor="#FCD34D", lw=1.2, zorder=1)
ax.add_patch(track_b_bg)

# Track B Header
header_b = patches.FancyBboxPatch((6.3, 3.85), 5.05, 0.35, boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor="#B45309", edgecolor="none", zorder=2)
ax.add_patch(header_b)
ax.text(8.82, 4.02, "TRACK B: FULL-DUPLEX CONVERSATIONAL DYNAMICS", 
        ha='center', va='center', fontsize=8.5, fontweight='bold', color="#FFFFFF", zorder=3)

# Tool 3: turnscope
card_ts = patches.FancyBboxPatch((6.35, 0.8), 4.95, 2.9, boxstyle="round,pad=0.03,rounding_size=0.06",
                                 facecolor="#FFFFFF", edgecolor="#FDE68A", lw=1.0, zorder=2)
ax.add_patch(card_ts)
ax.text(6.55, 3.45, "turnscope", fontsize=10.0, fontweight='bold', color="#B45309", zorder=3)
ax.text(7.7, 3.45, "[Echo Suppression & Turn Latency Engine]", fontsize=8.0, color="#64748B", zorder=3)

ax.text(6.55, 3.12, "• 18 dB Energy-Dominance Heuristic:", fontsize=8.0, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(6.75, 2.90, "Suppresses acoustic loudspeaker-to-mic echo crosstalk", fontsize=7.5, color="#475569", zorder=3)
ax.text(6.75, 2.70, "Reduces phantom turn detections; raises speech detection F1 to 0.897", fontsize=7.2, color="#475569", zorder=3)

ax.text(6.55, 2.40, "• Asynchronous State Machine:", fontsize=8.0, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(6.75, 2.18, "Transitions: Silence \u2192 Agent \u2192 Customer \u2192 Overlap \u2192 Dead Air", fontsize=7.5, color="#475569", zorder=3)

ax.text(6.55, 1.88, "• Conversational Tail Latency & Barge-In Metrics:", fontsize=8.0, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(6.75, 1.66, "Hand-over latency quantiles: P50, P90, and P99 ECDF tracking", fontsize=7.5, color="#475569", zorder=3)
ax.text(6.75, 1.45, "Measures customer barge-in response yield & dead air (\u2265 3.0s)", fontsize=7.5, color="#475569", zorder=3)

ax.text(6.55, 1.15, "• Real-World Validation (MagicHub Corpus):", fontsize=8.0, fontweight='semibold', color="#1E293B", zorder=3)
ax.text(6.75, 0.95, "Reduces P90 tail latency estimation error by 67.1% (1.34s \u2192 0.44s)", fontsize=7.5, color="#047857", fontweight='bold', zorder=3)

# Footer bar
footer = patches.FancyBboxPatch((0.5, 0.1), 11.0, 0.38, boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8, zorder=1)
ax.add_patch(footer)
ax.text(6.0, 0.28, "Suite Benchmark Hub: https://github.com/BoobalanArjunan/OpenSource-AudioEvaluation-Suite • MIT License",
        ha='center', va='center', fontsize=7.8, fontweight='medium', color="#334155", zorder=2)

plt.tight_layout()
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/paper2_audio_evaluation/figures/paper2_architecture.png", format='png', dpi=300, bbox_inches='tight')
shutil.copy("paper/paper2_audio_evaluation/figures/paper2_architecture.pdf", "paper/paper2_audio_evaluation/paper2_architecture.pdf")
shutil.copy("paper/paper2_audio_evaluation/figures/paper2_architecture.png", "paper/paper2_audio_evaluation/paper2_architecture.png")
plt.close()
print("Successfully generated clean, legible paper2_architecture.pdf & .png!")
