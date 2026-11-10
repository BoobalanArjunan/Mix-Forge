# Real-data check (MUSDB18-HQ)

This tests mixforge on real music stems. It checks that real files (different lengths, sample rates, mono or stereo) go through cleanly, that `mixforge verify` passes on everything made, and that the bleed you ask for is the bleed you get.

1. Get MUSDB18-HQ (search "MUSDB18-HQ" on Zenodo). It is free for non-commercial research. Unzip it; you want the `train` folder, which has one folder per song with `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav` and `mixture.wav`. mixforge does not ship any music.
2. Install: `pip install mixforge-audio` (or work from a project checkout to get the `scripts/` folder).
3. Quick trial: `python scripts/real_data_check.py --stems ~/musdb18hq/train --max-songs 3 --out real_check`
4. Full run: `python scripts/real_data_check.py --stems ~/musdb18hq/train --max-songs 20 --out real_check`
5. Open `real_check/results.md`.

What the numbers mean:

- **Pipeline sections:** every mix should pass `verify` (0 failures). "Largest |mixture - sum of stems|" should be tiny (about 1e-7 for FLOAT, about 1e-4 for PCM_16).
- **Bleed without low-pass:** the gap between asked and measured level should be 0.0000 dB.
- **Bleed with low-pass:** the measured level is lower than asked, because filtering removes some of the leak. That is expected.
- **Information only:** how close our plain sum is to the dataset's own `mixture.wav`. A large number means the dataset's mixture was made differently. It does not mean mixforge is wrong.

**Listen for two minutes.** Play `mixture.wav`, `stems/vocals.wav` and `leaky/vocals.wav` from a few mixes. The leaky vocal should sound like the clean vocal with faint, duller drums behind it.

Your original files are never modified. Everything is written under `real_check/`.
