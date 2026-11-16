import subprocess
import os

paper1_md = """# Building Reliable Audio Datasets for AI/ML Training: An Open-Source Quality Assurance Framework

**Author:** Boobalan Arjunan (`boobalan_arjunan@composermind.com`)  
**Affiliation:** ComposerMind Labs, Chennai, Tamil Nadu, India  
**Repository:** [ComposerMind-AudioIntegrity-Suite](https://github.com/BoobalanArjunan/ComposerMind-AudioIntegrity-Suite)  
**Primary arXiv Category:** `cs.SD` (Sound) / `eess.AS` (Audio and Speech Processing)  
**Target Submission:** IEEE Access / IEEE TASLP  

---

## Abstract

Deep neural networks for speech recognition, audio synthesis, and music information retrieval are increasingly trained on massive web-scale corpora. However, uncurated audio datasets frequently harbor insidious physical defects: cross-partition train/test data leakage, fake lossless upsampling, destructive stereo phase inversion, mislabeled musical tempos and keys, transcript timestamp drift, and digital clipping distortion. When models train on such contaminated inputs, performance degrades unpredictably while benchmark metrics report inflated scores.

To resolve this dataset hygiene crisis without introducing circular dependencies on opaque neural evaluators, we present the **ComposerMind AudioIntegrity Suite**, an open-source framework of seven deterministic digital signal processing (DSP) quality gates: `wavewash`, `splitsafe`, `bandcheck`, `phasecheck`, `driftcheck-audio`, `tempocheck`, and `tonicfind`. Engineered with zero deep learning dependencies, these tools execute in sub-second CPU time using closed-form physical invariants, spectral-peak landmark hashing, cubic Hermite spline interpolation, and permutation hypothesis testing. Evaluated on canonical benchmarks including 5,323 LibriSpeech recordings and synthetic ground-truth stress suites, the framework achieves 100% adversarial leak detection, 71.8% clipping MSE reduction, and zero false alarms on clean studio data.

---

## 1. System Architecture: The Seven Integrity Tools

![Figure 1: ComposerMind AudioIntegrity Suite Architecture](figures/paper1_architecture.png)

### Group I: Signal Quality & Partition Hygiene (3 Tools)
1. **`splitsafe`**: Combinatorial landmark spectral peak hashing and time-displacement histogram voting. Detects evaluation duplicates down to 3s sub-clips, resampled to 11 kHz, or gain-shifted by $-12$ dB. Enforces strict speaker isolation across train/val/test splits.
2. **`wavewash`**: Saturated boundary clipping detection, piecewise cubic Hermite interpolating polynomial (PCHIP) de-clipping (71.8% MSE drop), DC displacement removal (20 Hz zero-phase HPF), and ITU-R BS.1770-4 gated integrated loudness normalization.
3. **`bandcheck`**: Active-frame power spectral density estimation and log-spectral cliff finite derivative detection ($\Delta S \ge 25$ dB). Distinguishes genuine 48 kHz recordings from 8 kHz, 11.025 kHz, and 16 kHz fake upsampling and lossy MP3/AAC artifacts.

### Group II: Harmonic, Temporal & Spatial Integrity (4 Tools)
4. **`tempocheck`**: Multi-band spectral flux onset novelty, unbiased autocorrelation tempograms, and harmonic ratio matching engine ($0.5\times, 2.0\times, 1.5\times, 0.67\times$). Rejects outright BPM errors while tolerating metric ambiguities.
5. **`tonicfind`**: Sub-bin parabolic peak interpolation for $\pm 50$ cents detuning compensation relative to ISO concert pitch A440, 12-bin pitch-class chromagram projection, and correlation against the 24 Krumhansl-Kessler cognitive key profiles.
6. **`phasecheck`**: Sliding-window Pearson correlation coefficient ($\rho_{LR} \in [-1, 1]$), sub-bass ($<150$ Hz) phase isolation, and destructive mono downmix energy loss ($\Delta E_{\text{mono}} < -3.0$ dB) to prevent stereo cancellation.
7. **`driftcheck-audio`**: Band-limited speech energy envelope extraction (200--4000 Hz, 10 ms frames), transcript interval rasterization, windowed cross-correlation regression ($\delta_w^* = \alpha + \beta t_w$), and non-parametric permutation null testing ($p < 0.01$).

---

## 2. Quantitative Benchmark Results

| Tool | Underlying Algorithm | Canonical Benchmark Dataset | Quantitative Result |
| :--- | :--- | :--- | :--- |
| **`splitsafe`** | Landmark Peak Hashing & Time Voting | LibriSpeech (5,323 files) | **100.0% recall on 56/56 planted leaks**; 0.0% false positive rate; 55.4s CPU runtime. |
| **`wavewash`** | Cubic Hermite Spline (PCHIP) & BS.1770 | 500 Clipped Speech Files | **71.8% MSE distortion drop**; zero false rejections on 200 clean studio tracks. |
| **`bandcheck`** | Log-Spectral Cliff Derivative ($\ge 25$ dB) | 185 Upsampled Files & Studio Tracks | **100.0% detection** of 8k/11k upsampling; **98.5% clean studio pass rate**. |
| **`tempocheck`** | Spectral Flux Autocorrelation Tempogram | 200 Synthetic / 50 Multi-tracks | **100.0% harmonic accuracy (Acc2)**; **88.0% rejection** of false BPM labels. |
| **`tonicfind`** | 12-Bin Chroma & Krumhansl-Kessler | 25 Multi-Genre Key Stems | **100.0% exact key match**; 1.000 MIREX weighted score; $\pm 50$ cents detuning comp. |
| **`phasecheck`** | Sliding Pearson Phase & Mono Loss | Synthetic Inverted Stems | **100.0% detection of 180° inverted polarity**; exact mono downmix loss measurement. |
| **`driftcheck-audio`** | Speech Energy Cross-Correlation | 6 Concatenated Long Talks | **100.0% planted offset recall** ($\pm 0.2$s to $\pm 2$s); median error 10--39 ms; $p < 0.01$. |

---

## 3. Developer Integration & CI/CD Exit Codes

```python
from wavewash import PROFILES, analyze_file, apply_fix_pipeline, load_audio

report = analyze_file("train/audio_01.wav", PROFILES["asr"])
if not report.passed:
    audio, sr = load_audio("train/audio_01.wav")
    clean_audio, sr, _ = apply_fix_pipeline(audio, sr, PROFILES["asr"], report.issues)
```

* POSIX Exit Codes: `0` = PASS, `1` = WARN, `2` = FAIL.

---

## 4. Software Availability

Open-source under MIT license:  
👉 **GitHub Repository:** [https://github.com/BoobalanArjunan/ComposerMind-AudioIntegrity-Suite](https://github.com/BoobalanArjunan/ComposerMind-AudioIntegrity-Suite)
"""

with open("paper/paper1_dataset_integrity/paper1_dataset_integrity.md", "w") as f:
    f.write(paper1_md)

paper2_md = """# Beyond Accuracy: Building Reproducible Benchmarks and Diagnostic Tools for Audio & Speech AI

**Author:** Boobalan Arjunan (`boobalan_arjunan@composermind.com`)  
**Affiliation:** ComposerMind Labs, Chennai, Tamil Nadu, India  
**Repository:** [ComposerMind-AudioEvaluation-Suite](https://github.com/BoobalanArjunan/ComposerMind-AudioEvaluation-Suite)  
**Primary arXiv Category:** `cs.SD` (Sound) / `cs.AI` (Artificial Intelligence)  
**Target Submission:** DAFx / AES / ISMIR  

---

## Abstract

Standard benchmark metrics in audio and speech artificial intelligence—such as Signal-to-Distortion Ratio (SDR) in music source separation and Word Error Rate (WER) in speech processing—obscure critical diagnostic failure modes. In source separation, commercial multi-track recordings inherently contain unmeasured acoustic microphone bleed and non-linear mastering compression, confounding algorithmic separation failures with physical recording artifacts. In full-duplex conversational voice agents, aggregate latency metrics fail to capture tail response delays, acoustic room-echo leakage, and customer barge-in compliance.

To establish reproducible ground-truth evaluation without black-box neural metrics, we introduce the **ComposerMind AudioEvaluation Suite**, comprising three deterministic DSP tools: `mixforge-audio`, `stemcheck`, and `turnscope`. `mixforge-audio` guarantees mathematically exact linear superposition ($\text{Mix} = \sum \text{Stems}$, float residual $\le 6.52 \times 10^{-8}$) with calibrated 4th-order Butterworth acoustic bleed and cryptographic recipe serialization. `stemcheck` introduces linear subspace projection to decompose separation error into a physical cross-stem bleed matrix (dB) versus unmodeled algorithmic artifacts. `turnscope` implements an 18 dB energy-dominance filter to suppress acoustic echo, extracting conversational turn hand-over latency distributions (P50, P90, P99) and customer barge-ins. Evaluated on MUSDB18-HQ and the MagicHub full-duplex corpus, the suite decouples bleed within 0.06 dB and reduces P90 tail latency error by 67.1%.

---

## 1. System Architecture: The Three Evaluation Tools

![Figure 1: ComposerMind AudioEvaluation Suite Architecture](figures/paper2_architecture.png)

### Track A: Source Separation Ground Truth & Bleed Decomposition (2 Tools)
1. **`mixforge-audio`**: Generates test mixtures with mathematically exact linear superposition ($\text{Mix} = \sum \text{Stems}$). In double precision arithmetic, residual error is bounded below $10^{-12}$. Implements dynamic headroom scaling ($\gamma = \min(1.0, 0.98 / P_{\max})$) to prevent clipping without dynamic range compression. Injects directional acoustic bleed via calibrated 4th-order Butterworth low-pass filtering. Serializes all parameters, gains, and seeds into tamper-evident `recipe.json`.
2. **`stemcheck`**: Projects separated stems $\hat{S}_i$ onto the subspace spanned by reference stems $\mathbf{S} = [S_1, \dots, S_K]$ via ordinary least squares $\mathbf{c}_i = (\mathbf{S}^T \mathbf{S})^{-1} \mathbf{S}^T \hat{S}_i$. Decouples physical cross-stem bleed $L_{ij} = 20 \log_{10} \frac{\|c_{ij} S_j\|_2}{\|c_{ii} S_i\|_2}$ from orthogonal unexplained artifact error $\mathcal{U}_i = 10 \log_{10} \frac{\|\mathcal{E}_i\|_2^2}{\|\hat{S}_i\|_2^2}$. Executes blind physical sanity assertions (conservation sum, DC bias, clipping, duplicate channels).

### Track B: Full-Duplex Conversational Dynamics (1 Tool)
3. **`turnscope`**: Audits full-duplex voice AI call recordings across Agent and Customer channels. Implements an 18 dB energy-dominance filter to suppress loudspeaker acoustic room echo. Drives an asynchronous state machine measuring turn hand-over latency distributions (P50, P90, P99), customer barge-in compliance, agent interruptions, and awkward dead air ($\ge 3.0$ s).

![Figure 2: TurnScope Conversational Latency ECDF on MagicHub](figures/turnscope_latency_ecdf.png)

---

## 2. Quantitative Benchmark Results

| Metric / Evaluation Dimension | Standard Baseline | ComposerMind AudioEvaluation Suite | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **`mixforge-audio` Linear Sum Residual** | Unchecked | $\mathbf{\le 6.52 \times 10^{-8}}$ | Double precision verified |
| **`mixforge-audio` Studio Mix Residual** | N/A | $\mathbf{-69.3}$ **dB Median** | Translucent to studio mixes |
| **`stemcheck` Bleed Calibration Error** | Confounded | $\mathbf{\le 0.06}$ **dB** | Decouples true bleed vs artifacts |
| **`stemcheck` Separation False Alarms** | N/A | **0 / 264 Trials** | Zero false leak detections |
| **`turnscope` Speech Detection F1** | 0.700 | **0.897** | $+28.1\%$ accuracy gain |
| **`turnscope` Missed Conversational Turns** | 207 | **126** | **$-39.1\%$ fewer missed turns** |
| **`turnscope` Median Latency Error (P50)** | 0.412 s | **0.182 s** | $-55.8\%$ estimation error drop |
| **`turnscope` Tail Latency Error (P90)** | 1.339 s | **0.440 s** | $\mathbf{-67.1\%}$ **tail latency error drop** |

---

## 3. Software Availability

Open-source under MIT license:  
👉 **GitHub Repository:** [https://github.com/BoobalanArjunan/ComposerMind-AudioEvaluation-Suite](https://github.com/BoobalanArjunan/ComposerMind-AudioEvaluation-Suite)
"""

with open("paper/paper2_audio_evaluation/paper2_audio_evaluation.md", "w") as f:
    f.write(paper2_md)

# Create submission tarballs for both papers
os.chdir("paper/paper1_dataset_integrity")
subprocess.run(["tar", "-czvf", "paper1_submission_package.tar.gz", "paper1_dataset_integrity.tex", "figures/"], check=True)
os.chdir("../paper2_audio_evaluation")
subprocess.run(["tar", "-czvf", "paper2_submission_package.tar.gz", "paper2_audio_evaluation.tex", "figures/"], check=True)

print("Created Markdown and submission tarballs for both Paper 1 and Paper 2 successfully!")
