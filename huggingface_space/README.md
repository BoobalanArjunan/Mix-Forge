---
title: Mix-Forge - Ground Truth Audio Synthesizer
emoji: 🎛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: Synthesize test mixes with exact answer keys & calibrated bleed
---

# Mix-Forge: Ground-Truth Multi-Track Audio Synthesizer

Build test audio mixes where you always know the exact answer key. Designed for evaluating Music Source Separation models (Demucs, Spleeter) and calibrated bleed detectors.

### Core Guarantees
- **Strict Superposition:** Mixture equals the exact linear sum of answer-key stems ($10^{-12}$ precision).
- **Calibrated Acoustic Bleed:** Injects physical, frequency-shaped microphone spill (e.g. drums leaking into vocal mic).
- **Self-Auditing Verification:** Mathematically proves files have not been modified or clipped.

*Created by **Boobalan Arjunan** ([boobalan.it@gmail.com](mailto:boobalan.it@gmail.com)) &bull; ComposerMind Labs*
