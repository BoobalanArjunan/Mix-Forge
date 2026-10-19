# Mix-Forge: Master Project Plan, Architecture & Checkpoint Log
**Author:** Boobalan Arjunan (<boobalan.it@gmail.com>)  
**Entity:** ComposerMind Labs  
**Project:** Mix-Forge (Ground-Truth Test Mix & Controlled Bleed Generator for Music Source Separation Evaluation)  
**Date Created / Milestone:** October 19, 2026  
**Status:** Alpha v0.1.0a1 (Ready for PyPI Packaging, CI/CD Active, 61/61 Unit Tests Passing)  
**Companion Architecture:** Audio ML Evaluation Suite (WaveWash ➔ SplitSafe ➔ Mix-Forge ➔ StemCheck)

---

## 1. Executive Summary

> **Core Problem in Music Source Separation:**  
> When evaluating source separation models (e.g., Demucs, Open-Unmix, Spleeter) or microphone bleed detectors, researchers face a fundamental evaluation bottleneck: on finished, commercial audio recordings, the "true ground truth" is unknown. Commercial studio tracks already contain unmeasured microphone bleed, unknown mastering compression, and stereo spatialization that bias evaluation metrics (SDR, SIR, SAR).

**Mix-Forge** is an open-source, local-first Python library and CLI engine engineered to generate **reproducible, mathematically exact test mixes with an unalterable answer key**. 

Given isolated stems (vocals, drums, bass, other), Mix-Forge:
1. Mixes stems at precisely chosen or seeded random gain levels.
2. Injects **calibrated acoustic bleed** (e.g., drums leaking into the vocal microphone at $-26\,\text{dB}$ with low-pass roll-off at $4000\,\text{Hz}$).
3. Preserves the exact, clean stems as the answer key alongside the leaky stems.
4. Generates an immutable, cryptographic `recipe.json` recording every parameter, sample offset, and gain adjustment.
5. Provides a rigorous verification command (`mixforge verify`) that mathematically verifies the mixture equals the sum of stems ($|\text{mix} - \sum \text{stems}| \le \epsilon$) and verifies that leaky stems match the recipe bleed.

---

## 2. Key Links, Metadata & Ecosystem

| Resource | Official Reference / Destination | Status / Target |
| :--- | :--- | :--- |
| **PyPI Package** | `https://pypi.org/project/mixforge/` | **v0.1.0a1** (`pip install --pre mixforge`) |
| **GitHub Repository** | `https://github.com/BoobalanArjunan/mixforge` | Branch `main`, Tag `v0.1.0a1`, CI/CD Workflows |
| **License** | MIT License | Permissive open-source distribution |
| **Test Suite** | `tests/test_mixforge.py` | **61 / 61 Tests Passing (100%)** |
| **Mutation Resilience** | 23 Mutation checks | **23 / 23 Caught (100% Mutation Coverage)** |
| **Downstream Consumer** | **StemCheck** | Bleed detection engine calibrated on Mix-Forge data |

---

## 3. Core Architecture & Signal Processing Guarantees

### A. Mathematical Sum Conservation
The central guarantee of Mix-Forge is absolute conservation:
$$\text{Mixture}[t] = \sum_{k \in \text{Stems}} S_k[t]$$
- If the algebraic sum of the stems exceeds the digital ceiling (0.98 peak limit), Mix-Forge uniformly scales the entire stem group downwards by peak factor $\gamma = \frac{\text{limit}}{\max |\sum S_k|}$, guaranteeing that no clipping occurs and that the linear superposition principle holds identically.
- In 64-bit float representations, the residual error $|\text{Mixture} - \sum S_k|$ is bounded strictly below $10^{-12}$.

### B. Calibrated Acoustic Bleed Modeling
Real-world acoustic bleed occurs when one instrument is captured by another instrument's microphone:
$$S_{\text{leaky}}[t] = S_{\text{clean}}[t] + g \cdot \mathcal{H}_{\text{LP}}(S_{\text{source}}[t])$$
Where:
- $g = 10^{\frac{\text{dB}}{20}}$ represents the linear attenuation factor.
- $\mathcal{H}_{\text{LP}}$ represents an optional 4th-order Butterworth low-pass filter (simulating acoustic damping and directional microphone high-frequency rolloff).

### C. Robust Real-World Input Normalization
Mix-Forge handles messy real-world datasets automatically without corrupting source files:
- **Sample Rate Mismatches:** Resamples outlier stems to the dataset's modal sample rate via high-fidelity polyphase filtering.
- **Channel Mismatches:** Automatically duplicates mono channels to stereo or matches channel geometry.
- **Length Mismatches:** Trims divergent stems to the shortest mutual duration.
- All corrective transformations are timestamped and logged in `recipe.json` under `notes`.

---

## 4. Software Stack & CLI Reference

### A. Technical Stack
- **Language:** Python 3.10+
- **Core Dependencies:** `numpy`, `scipy`, `soundfile` (zero heavy machine learning dependencies for instantaneous, CPU-efficient execution).
- **CLI Entrypoint:** `mixforge`

### B. Command-Line Workflows
```bash
# 1. Generate test mixes with calibrated bleed
mixforge make stems/ --out mixes/ --count 3 --seconds 10 \
    --gain-range -6 6 \
    --bleed "drums>vocals:-26@4000" --bleed "bass>other:-30" \
    --seed 42

# 2. Mathematically verify mix integrity and bleed compliance
mixforge verify mixes/

# 3. Export verification as JSON for CI/CD assertions
mixforge verify mixes/ --json
```

---

## 5. Execution Roadmap & Task Status

- [x] **Task 1: Environment & Unit Test Suite** — Clean `.venv` configured, dependencies installed, **61/61 unit tests passing**.
- [x] **Task 2: Distribution Package Validation** — Wheel (`.whl`) and source tarball (`.tar.gz`) built via `python -m build`, validated with `twine check dist/*` (100% PASSED).
- [x] **Task 3: Git Repository Initialization** — Initial commit `b74e3e7` staged on `main`, remote origin `BoobalanArjunan/mixforge` configured.
- [x] **Task 4: Synthetic Trial Pipeline** — Verified end-to-end on synthetic stems (`FLOAT` and `PCM_16` error < 1e-7, 0 failures).
- [ ] **Task 5: Interactive Visual HTML Inspection Report** — Add `--html report.html` to `mixforge verify` (matching SplitSafe standard).
- [ ] **Task 6: Real-Data Benchmark with MUSDB18-HQ** — Run `scripts/real_data_check.py` and commit `real_check/results.md`.
- [ ] **Task 7: Remote GitHub Push & CI Verification** — Push `main` to GitHub, confirm multi-OS CI matrix.
- [ ] **Task 8: PyPI Deployment & Release Tagging** — Deploy to PyPI, tag `v0.1.0a1`, and publish companion PR works.
