# mixforge

Make test mixes where you know the right answer.

You give mixforge separate stems (vocals, drums, bass, other). It mixes them at known levels, can add a **known amount of bleed** (for example "a little drums leaking into the vocal"), and keeps the exact originals plus a `recipe.json`. You can then test whether a separation tool, or a bleed detector, gives the right answer, because you know what went in.

> **Status: alpha (0.1.0).** Tested on synthetic audio with known answers, mutation checks, and verified on real music stems from MUSDB18-HQ (see [Real-data check](#real-data-check) below).

## Install

```
pip install --pre mixforge
```

(`--pre` is needed while it is an alpha release.)

## Quick start

Folder layout, one folder per song (this is the MUSDB18-HQ layout; a `mixture.wav` next to the stems is ignored):

```
stems/
  Song A/  vocals.wav  drums.wav  bass.wav  other.wav
  Song B/  ...
```

```
mixforge make stems --out mixes --count 3 --seconds 10 \
    --gain-range -6 6 \
    --bleed "drums>vocals:-26@4000" --bleed "bass>other:-30" \
    --seed 1
mixforge verify mixes
```

Each mix gets its own folder:

```
mixes/0000_Song_A_000/
  mixture.wav        the mix
  stems/*.wav        the answer key: the clean stems exactly as they sit in the mix
  leaky/*.wav        the same stems with the bleed added (only if you asked for bleed)
  recipe.json        everything that was done, so the result can be repeated and checked
mixes/index.csv      one line per mix
```

## What the options mean

| Option | Meaning |
|---|---|
| `--gain-range LO HI` | random level change per stem, in dB, drawn from the seed |
| `--gain STEM=DB` | fixed level for one stem |
| `--bleed SRC>DST:DB[@HZ]` | leak `SRC` into `DST`. `DB` is the leak level relative to `SRC` as it sits in the mix: `-20` = a tenth of the amplitude, `-6` = about half. `@HZ` low-passes the leak (real bleed is often duller) |
| `--seconds S` | cut the same random window of `S` seconds out of every stem |
| `--count N` | `N` different mixes per song |
| `--seed N` | same seed, same files (byte for byte) |
| `--subtype` | `FLOAT` (default, never clips), `PCM_16` or `PCM_24` |
| `--max-songs N` | only the first `N` songs, for a quick trial |
| `--html FILE` | write an interactive HTML listening audit report with multi-track players |

Rules mixforge follows:

- **The mixture is the exact sum of the stems in `stems/`.** If the sum would clip, everything is turned down together so the sum still holds. `recipe.json` records the change.
- **Stems that do not match are matched up and noted.** Different sample rates are resampled to the most common one, mono stems are copied to stereo, and different lengths are trimmed to the shortest. Each fix is written into `recipe.json` under `notes`.
- **Your files are never modified**, and existing output is never overwritten.
- **`mixforge verify`** re-checks any output folder: mixture = sum of stems, and each leaky file really equals the clean stem plus the bleed the recipe describes. It exits 0 if all is well, 1 if something fails, 2 for usage errors.

## Python

```python
import mixforge
from mixforge import Bleed, make_mix

r = make_mix(stems, sr, gain_range=(-6, 6), seed=1,
             bleed=[Bleed("drums", "vocals", -26.0, lowpass_hz=4000)])
r.mixture, r.stems, r.leaky, r.recipe
```

## Real-data check

**Dataset:** MUSDB18-HQ (`train` split, 20 songs, 40 mixes, 10 s windows)  
**Date:** November 5, 2026  
**Recipe:** gain range -6..+6 dB, bleed `drums>vocals:-26@4000`, `bass>other:-30`, `vocals>other:-20`, 2 mixes per song.

### Pipeline, FLOAT
- **Mixes verified:** 40 of 40 (0 failures)
- **Largest $|\text{mixture} - \sum \text{stems}|$:** 6.52e-08
- **Bleed without low-pass:** 0.0000 dB gap between asked and measured level
- **Bleed with low-pass:** measured level minus asked level between -1.21 dB and 0.0 dB
- **Headroom scaling:** 15 mixes scaled automatically to prevent clipping
- **Time:** 8.5 s

### Pipeline, PCM_16
- **Mixes verified:** 40 of 40 (0 failures)
- **Largest $|\text{mixture} - \sum \text{stems}|$:** 9.16e-05
- **Bleed without low-pass:** 0.0000 dB gap between asked and measured level
- **Bleed with low-pass:** measured level minus asked level between -1.21 dB and 0.0 dB
- **Headroom scaling:** 15 mixes scaled automatically to prevent clipping
- **Time:** 9.9 s

### Information check vs dataset's own mixture.wav (30 s window, no gain changes)
- **Songs compared:** 20
- **Residual (lower is closer):** best -73.3 dB, median -69.3 dB, worst -62.8 dB (all well below the -60 dB auditory perception threshold)

## Limits (please read)

- **This is test material, not music.** The mixes are gain-scaled sums of stems. They are not mastered and do not claim to sound like finished records.
- **The bleed is simple.** It is a quieter, optionally duller copy of one stem added to another. Real microphone bleed also has delay, room sound and frequency changes that this does not copy. A detector that works here may still struggle on real recordings.
- **Stems must already be time-aligned** (as in MUSDB18-HQ). mixforge does not align them.
- **No cross-song remixing yet** (vocals from one song over drums from another).
- Stems are read with `soundfile` (WAV, FLAC, OGG, AIFF, and MP3 where your libsndfile supports it). Not supported: `.stem.mp4` files (use MUSDB18-HQ's WAV version).

## Related work

- [MUSDB18](https://sigsep.github.io/datasets/musdb.html) and `museval` provide real stems and standard separation scores. mixforge does not replace them. It makes controlled, repeatable test mixes you can use next to them.
- Mixing tools such as Scaper and audiomentations exist for other uses (sound-event data, augmentation). mixforge's focus is a recorded answer key and controlled bleed for separation testing.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT. Copyright (c) 2026 Boobalan Arjunan.
