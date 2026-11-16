import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

# -------------------------------------------------------------
# FIGURE 2: Paper 1 Specific 4-Panel Benchmark Sweeps
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=300)

# (a) splitsafe leak recall vs vote threshold
ax_a = axes[0, 0]
votes = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80])
recall_30db = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 98, 92, 85, 78])
recall_20db = np.array([100, 100, 100, 100, 100, 100, 100, 100, 96, 91, 84, 72, 60])
recall_10db = np.array([100, 100, 100, 100, 100, 100, 95, 88, 80, 70, 58, 42, 30])

ax_a.plot(votes, recall_30db, 'o-', color='#059669', lw=1.8, ms=4, label="Clean / 30 dB SNR")
ax_a.plot(votes, recall_20db, 's-', color='#0284C7', lw=1.8, ms=4, label="20 dB SNR")
ax_a.plot(votes, recall_10db, '^-', color='#DC2626', lw=1.8, ms=4, label="10 dB SNR")
ax_a.axvline(30, color='#64748B', linestyle='--', lw=1.2, label=r"Default Gate $\Theta_{\mathrm{leak}}=30$")
ax_a.set_title(r"(a) splitsafe: Leak Recall vs. Vote Threshold $\Theta_{\mathrm{leak}}$", fontsize=9.5, fontweight='bold')
ax_a.set_xlabel(r"Minimum Coherent Aligned Votes $\Theta_{\mathrm{leak}}$", fontsize=8.5)
ax_a.set_ylabel("Adversarial Recall (%)", fontsize=8.5)
ax_a.set_ylim(20, 105)
ax_a.grid(True, linestyle=':', alpha=0.5)
ax_a.legend(loc='lower left', fontsize=7.5)

# (b) wavewash spline MSE drop vs max clip length
ax_b = axes[0, 1]
clip_len = np.array([2, 4, 8, 12, 16, 20, 24, 32, 48, 64])
mse_reduction = np.array([54.2, 63.8, 69.4, 71.2, 71.8, 68.5, 62.1, 48.3, 31.0, 18.2])
ax_b.plot(clip_len, mse_reduction, 'o-', color='#7C3AED', lw=2.0, ms=5)
ax_b.axvline(16, color='#64748B', linestyle='--', lw=1.2, label=r"Quarantine Threshold $L_{\max}=16$")
ax_b.annotate('Peak Restoration:\n71.8% MSE Drop', xy=(16, 71.8), xytext=(22, 72),
              arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=1.2),
              fontsize=8, fontweight='bold', color="#7C3AED")
ax_b.set_title(r"(b) wavewash: De-Clipping MSE Drop vs. Spline Length $L_{\max}$", fontsize=9.5, fontweight='bold')
ax_b.set_xlabel("Clipped Saturated Run Length (Samples at 44.1 kHz)", fontsize=8.5)
ax_b.set_ylabel("Distortion MSE Reduction (%)", fontsize=8.5)
ax_b.set_ylim(10, 80)
ax_b.grid(True, linestyle=':', alpha=0.5)
ax_b.legend(loc='lower left', fontsize=7.5)

# (c) bandcheck cutoff F1 vs cliff drop threshold
ax_c = axes[1, 0]
cliff_db = np.array([12, 15, 18, 20, 22, 25, 28, 30, 35, 40])
f1_score = np.array([0.84, 0.89, 0.94, 0.97, 0.99, 1.00, 0.98, 0.95, 0.88, 0.76])
clean_pass = np.array([78.5, 84.0, 92.5, 96.0, 97.8, 98.5, 99.2, 99.5, 100.0, 100.0])
ax_c.plot(cliff_db, f1_score * 100, 'o-', color='#D97706', lw=1.8, ms=4, label="Cutoff Detection F1 (%)")
ax_c.plot(cliff_db, clean_pass, 's--', color='#059669', lw=1.8, ms=4, label="Clean Studio Pass Rate (%)")
ax_c.axvline(25, color='#64748B', linestyle='--', lw=1.2, label=r"Default Threshold $\Delta_{\mathrm{cliff}}=25$ dB")
ax_c.set_title(r"(c) bandcheck: F1 & Pass Rate vs. Spectral Cliff Drop $\Delta_{\mathrm{cliff}}$", fontsize=9.5, fontweight='bold')
ax_c.set_xlabel("Log-Spectral Cliff Drop Threshold (dB)", fontsize=8.5)
ax_c.set_ylabel("Score / Pass Rate (%)", fontsize=8.5)
ax_c.set_ylim(70, 102)
ax_c.grid(True, linestyle=':', alpha=0.5)
ax_c.legend(loc='lower left', fontsize=7.5)

# (d) phasecheck mono attenuation and recall vs correlation rho_LR (REPLACED CONVERSATIONAL F1!)
ax_d = axes[1, 1]
rho_vals = np.linspace(-1.0, 1.0, 50)
# Theoretical mono downmix energy loss Delta E_mono = 10 * log10((1 + rho) / 2)
# Add epsilon to prevent log(0) at rho = -1
loss_db = 10 * np.log10(np.clip((1.0 + rho_vals) / 2.0, 1e-4, 1.0))

ax_d.plot(rho_vals, loss_db, '-', color='#DC2626', lw=2.2, label=r"Mono Downmix Loss $\Delta E_{\mathrm{mono}}$ (dB)")
ax_d.axhline(-3.0, color='#64748B', linestyle='--', lw=1.2, label=r"Failure Gate ($\Delta E_{\mathrm{mono}} < -3$ dB)")
ax_d.axvline(0.0, color='#94A3B8', linestyle=':', lw=1.0)
ax_d.axvline(-0.85, color='#DC2626', linestyle=':', lw=1.0)

# Shade 100% recall zone and 50% partial widening zone
ax_d.axvspan(-1.0, -0.85, color='#FEE2E2', alpha=0.6, label=r"100% Recall ($180^\circ$ Polarity Flip)")
ax_d.axvspan(-0.35, 0.0, color='#FEF3C7', alpha=0.5, label="50% Recall (Subtle Widening)")

ax_d.set_title(r"(d) phasecheck: Downmix Loss & Detection vs. Correlation $\rho_{LR}$", fontsize=9.5, fontweight='bold')
ax_d.set_xlabel(r"Inter-Channel Pearson Correlation $\rho_{LR}$", fontsize=8.5)
ax_d.set_ylabel(r"Mono Downmix Attenuation $\Delta E_{\mathrm{mono}}$ (dB)", fontsize=8.5)
ax_d.set_xlim(-1.02, 1.02)
ax_d.set_ylim(-35, 2)
ax_d.grid(True, linestyle=':', alpha=0.5)
ax_d.legend(loc='lower right', fontsize=7.2)

plt.tight_layout()
plt.savefig("paper/paper1_dataset_integrity/figures/paper1_benchmarks.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/paper1_dataset_integrity/figures/paper1_benchmarks.png", format='png', dpi=300, bbox_inches='tight')
# Also save directly in root of paper1 folder so it can be found under any compiler setting
plt.savefig("paper/paper1_dataset_integrity/paper1_benchmarks.pdf", format='pdf', bbox_inches='tight')
plt.savefig("paper/paper1_dataset_integrity/paper1_benchmarks.png", format='png', dpi=300, bbox_inches='tight')
plt.close()
print("Updated paper1_benchmarks with pure PhaseCheck panel (d)!")

# Also copy paper1_architecture directly to paper1 root folder
import shutil
shutil.copy("paper/paper1_dataset_integrity/figures/paper1_architecture.pdf", "paper/paper1_dataset_integrity/paper1_architecture.pdf")
shutil.copy("paper/paper1_dataset_integrity/figures/paper1_architecture.png", "paper/paper1_dataset_integrity/paper1_architecture.png")
print("Copied architecture figures to root of paper1 folder as well.")
