# mixforge real-data check

Stems: `/Users/mymac/Desktop/Beatunzip-Test/musdb18hq/train`. Songs used for the information check: 20. Mix length 10 s. seed = 0.
Recipe: gain range -6..+6 dB, bleed drums>vocals:-26@4000, bass>other:-30, vocals>other:-20, 2 mixes per song.

## Pipeline, FLOAT

- Mixes made and re-checked by `mixforge verify`: **40**, failures: **0**
- Largest |mixture - sum of stems| found: 6.52e-08
- Bleed without low-pass: largest gap between asked and measured level: 0.0000 dB
- Bleed with low-pass: measured level minus asked level (min, max): [-1.21, -0.0]
- Things mixforge had to fix on the way (count of mixes): {'mix would clip': 15}
- Time: 8.5 s

## Pipeline, PCM_16

- Mixes made and re-checked by `mixforge verify`: **40**, failures: **0**
- Largest |mixture - sum of stems| found: 9.16e-05
- Bleed without low-pass: largest gap between asked and measured level: 0.0000 dB
- Bleed with low-pass: measured level minus asked level (min, max): [-1.21, -0.0]
- Things mixforge had to fix on the way (count of mixes): {'mix would clip': 15}
- Time: 9.9 s

## Information only: our sum vs the dataset's own mixture.wav (30 s window, no gain changes)

- Songs compared: 20. Residual (lower is closer): best -73.3 dB, median -69.3 dB, worst -62.8 dB
- Rule of thumb: below about -60 dB the two mixes are the same to the ear.

## Listen (2 minutes, do this)

Open three folders under `real_check/mixes_FLOAT/`. Play `mixture.wav`, then `stems/vocals.wav`, then `leaky/vocals.wav`.
The leaky vocal should sound like the clean vocal with faint, duller drums behind it. If it does not, tell us what you hear.
