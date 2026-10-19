"""Real-data check for mixforge using MUSDB18-HQ (or any folder of song folders with stem files).

Questions answered with numbers, on real music:
  1. Do real stems load and mix without trouble (different lengths, mono files, odd sample rates)?
  2. Does `mixforge verify` pass on everything it makes (FLOAT and PCM_16)?
  3. Does the bleed that was asked for match the bleed that was measured?
  4. INFORMATION ONLY: with no gain changes, how close is our sum of the stems to the dataset's own
     mixture.wav? (If it is not close, the dataset's mixture was made differently. That says nothing
     about mixforge being wrong, but it is worth knowing.)

Your stems are never modified. Output goes to --out (default real_check).

Usage:
  python scripts/real_data_check.py --stems ~/musdb18hq/train --max-songs 10 --out real_check
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from mixforge import Bleed, make_dataset, make_mix, verify_folder
from mixforge.report import to_html
from mixforge.data import find_songs, load_song

BLEEDS = [Bleed("drums", "vocals", -26.0, 4000.0), Bleed("bass", "other", -30.0), Bleed("vocals", "other", -20.0)]


def residual_db(mine: np.ndarray, official: np.ndarray) -> float:
    num = float(np.sum((mine - official) ** 2))
    den = float(np.sum(official ** 2)) + 1e-30
    return 10 * np.log10(num / den + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, help="folder of song folders, e.g. musdb18hq/train")
    ap.add_argument("--max-songs", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=10.0, help="length of each test mix (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_check")
    a = ap.parse_args()
    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = Path(a.stems).expanduser()

    songs, notes = find_songs(src)
    songs = songs[:a.max_songs]
    print(f"{len(songs)} song(s) from {src}")

    # ---- 4 (information): our sum vs the dataset's own mixture.wav
    resid = []
    for name, paths in songs:
        official = None
        for ext in (".wav", ".flac"):
            cand = paths[next(iter(paths))].parent / f"mixture{ext}"
            if cand.exists():
                official = cand
        if official is None:
            continue
        stems, sr, _ = load_song(paths)
        r = make_mix(stems, sr, seconds=30, seed=a.seed, peak_limit=1e9)
        info = sf.info(str(official))
        if info.samplerate != sr:
            continue
        off = r.recipe["offset_samples"]
        mix, _ = sf.read(str(official), start=off, frames=r.recipe["samples"], dtype="float64", always_2d=True)
        if mix.shape == r.mixture.shape:
            resid.append((name, residual_db(r.mixture, mix)))
        del stems, r

    # ---- 1-3: the real pipeline, twice (FLOAT, PCM_16)
    results = {}
    for subtype in ("FLOAT", "PCM_16"):
        t0 = time.time()
        d = out / f"mixes_{subtype}"
        try:
            res = make_dataset(src, d, count=2, seconds=a.seconds, gain_range=(-6, 6), bleed=BLEEDS,
                               seed=a.seed, subtype=subtype, peak_limit=0.9, max_songs=a.max_songs)
        except Exception as e:                      # keep going so the report shows what failed
            results[subtype] = {"error": str(e)}
            print(f"[{subtype}] FAILED to build: {e}")
            continue
        t_make = time.time() - t0
        ver = verify_folder(d)
        try:
            (out / f"report_{subtype}.html").write_text(to_html(ver, out_path=out / f"report_{subtype}.html", base_dir=d), encoding="utf-8")
        except Exception:
            pass
        bad = [c for c in ver.checks if not c.ok]
        meas, lows, notes_seen = [], [], {}
        for mdir in sorted(p for p in d.iterdir() if p.is_dir()):
            rec = json.loads((mdir / "recipe.json").read_text())
            for n in rec["notes"]:
                key = n.split(":")[0].split(";")[0]
                notes_seen[key] = notes_seen.get(key, 0) + 1
            for b in rec["bleed"]:
                if b["measured_db_vs_source"] is None:
                    continue
                if b["lowpass_hz"] is None:
                    meas.append(abs(b["measured_db_vs_source"] - b["requested_db"]))
                else:
                    lows.append(b["measured_db_vs_source"] - b["requested_db"])
        results[subtype] = {
            "mixes": len(ver.checks), "verify_failed": len(bad),
            "problems": [f"{c.mix}: {p}" for c in bad for p in c.problems][:10],
            "max_sum_error": max((c.max_sum_error or 0) for c in ver.checks),
            "bleed_plain_max_error_db": max(meas) if meas else None,
            "bleed_lowpass_extra_db_min_max": [round(min(lows), 2), round(max(lows), 2)] if lows else None,
            "notes": notes_seen, "seconds": round(t_make, 1),
        }
        print(f"[{subtype}] {len(ver.checks)} mixes, verify failures: {len(bad)}, time {t_make:.0f}s")

    # ---- report
    L = ["# mixforge real-data check", "",
         f"Stems: `{src}`. Songs used for the information check: {len(songs)}. Mix length {a.seconds:g} s. seed = {a.seed}.",
         f"Recipe: gain range -6..+6 dB, bleed {', '.join(f'{b.source}>{b.target}:{b.db:g}' + (f'@{b.lowpass_hz:g}' if b.lowpass_hz else '') for b in BLEEDS)}, 2 mixes per song.", ""]
    for subtype, r in results.items():
        L += [f"## Pipeline, {subtype}", ""]
        if "error" in r:
            L += [f"- **Could not build:** {r['error']}", ""]
            continue
        L += [f"- Mixes made and re-checked by `mixforge verify`: **{r['mixes']}**, failures: **{r['verify_failed']}**",
              f"- Largest |mixture - sum of stems| found: {r['max_sum_error']:.2e}",
              f"- Bleed without low-pass: largest gap between asked and measured level: "
              f"{'n/a' if r['bleed_plain_max_error_db'] is None else format(r['bleed_plain_max_error_db'], '.4f') + ' dB'}",
              f"- Bleed with low-pass: measured level minus asked level (min, max): {r['bleed_lowpass_extra_db_min_max']}",
              f"- Things mixforge had to fix on the way (count of mixes): {r['notes'] or 'none'}",
              f"- Time: {r['seconds']} s"]
        if r["problems"]:
            L += ["- Problems:"] + [f"  - {p}" for p in r["problems"]]
        L.append("")
    L += ["## Information only: our sum vs the dataset's own mixture.wav (30 s window, no gain changes)", ""]
    if resid:
        v = [x for _, x in resid]
        L += [f"- Songs compared: {len(v)}. Residual (lower is closer): best {min(v):.1f} dB, median {np.median(v):.1f} dB, worst {max(v):.1f} dB",
              "- Rule of thumb: below about -60 dB the two mixes are the same to the ear.", ""]
    else:
        L += ["- No mixture file found next to the stems, nothing to compare.", ""]
    L += ["## Listen (2 minutes, do this)", "",
          f"Open three folders under `{out}/mixes_FLOAT/`. Play `mixture.wav`, then `stems/vocals.wav`, then `leaky/vocals.wav`.",
          "The leaky vocal should sound like the clean vocal with faint, duller drums behind it. If it does not, tell us what you hear.", ""]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"pipeline": results, "residual_db": resid}, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
