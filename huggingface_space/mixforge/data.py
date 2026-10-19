"""Finding stems on disk, making them match, and writing mixes."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .core import Bleed, MixforgeError, MixResult, make_mix

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
SUBTYPES = ("FLOAT", "PCM_16", "PCM_24")
DEFAULT_EXCLUDE = ("mixture",)   # MUSDB-style folders ship a mixture.wav next to the stems; it is not a stem


def _audio_files(folder: Path, exclude) -> list[Path]:
    ex = {e.lower() for e in exclude}
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXT and p.stem.lower() not in ex)
    names = Counter(p.stem for p in files)
    dup = [k for k, v in names.items() if v > 1]
    if dup:
        raise MixforgeError(f"{folder.name}: more than one file named {dup[0]!r} (different extensions); keep one")
    return files


def find_songs(source: Path, exclude=DEFAULT_EXCLUDE):
    """Return ([(song_name, {stem_name: path})], notes).

    `source` is either one song (a folder with 2+ stem files) or a folder of such songs.
    """
    source = Path(source)
    if not source.is_dir():
        raise MixforgeError(f"{source} is not a folder")
    notes: list[str] = []
    direct = _audio_files(source, exclude)
    if len(direct) >= 2:
        return [(source.resolve().name, {p.stem: p for p in direct})], notes
    songs = []
    for sub in sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")):
        files = _audio_files(sub, exclude)
        if len(files) >= 2:
            songs.append((sub.name, {p.stem: p for p in files}))
        else:
            notes.append(f"skipped {sub.name}: {len(files)} stem file(s), need at least 2")
    if not songs:
        raise MixforgeError(f"no songs found in {source}: give a folder with 2 or more stem files, "
                            "or a folder of such folders")
    return songs, notes


def load_song(paths: dict[str, Path], sr: int | None = None):
    """Read stems and make them match: one sample rate, one channel count, one length.

    Returns (stems {name: (n, ch) float32}, sr, notes).
    """
    raw = {}
    for name, p in paths.items():
        try:
            a, rate = sf.read(str(p), dtype="float32", always_2d=True)
        except Exception as e:  # unreadable or corrupt file
            raise MixforgeError(f"cannot read {p.name}: {e}") from None
        raw[name] = (a, rate)
    notes = []
    if sr is None:
        counts = Counter(r for _, r in raw.values())
        sr = max(counts, key=lambda r: (counts[r], r))
    chans = {a.shape[1] for a, _ in raw.values()}
    ch = max(chans)
    stems, resampled, upmixed = {}, [], []
    for name, (a, rate) in raw.items():
        if rate != sr:
            g = gcd(rate, sr)
            a = signal.resample_poly(a, sr // g, rate // g, axis=0)
            resampled.append(f"{name} ({rate} Hz)")
        if a.shape[1] != ch:
            if a.shape[1] != 1:
                raise MixforgeError(f"stem {name!r} has {a.shape[1]} channels but another has {ch}; "
                                    "only mono can be matched up to a bigger channel count")
            a = np.repeat(a, ch, axis=1)
            upmixed.append(name)
        stems[name] = a
    if resampled:
        notes.append(f"resampled to {sr} Hz: {', '.join(sorted(resampled))}")
    if upmixed:
        notes.append(f"mono stem(s) copied to {ch} channels: {', '.join(sorted(upmixed))}")
    lengths = {k: v.shape[0] for k, v in stems.items()}
    n = min(lengths.values())
    if len(set(lengths.values())) > 1:
        notes.append(f"stems had different lengths; all trimmed to the shortest ({n / sr:.2f} s)")
    stems = {k: v[:n] for k, v in stems.items()}
    return stems, sr, notes


def _check_pcm_fits(result: MixResult, subtype: str):
    if subtype == "FLOAT":
        return
    arrays = [result.mixture, *result.stems.values(), *(result.leaky or {}).values()]
    peak = max(float(np.abs(a).max()) for a in arrays)
    if peak > 1.0:
        raise MixforgeError(f"the audio peaks at {peak:.2f}, which does not fit in {subtype} (limit 1.0). "
                            "Use --subtype FLOAT, lower the gains or the bleed level, or lower --peak")


def _write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_mix(result: MixResult, folder: Path, subtype: str = "FLOAT") -> Path:
    """Write mixture.wav, stems/*.wav, leaky/*.wav (if any bleed) and recipe.json. Never overwrites."""
    if subtype not in SUBTYPES:
        raise MixforgeError(f"--subtype must be one of {', '.join(SUBTYPES)}")
    folder = Path(folder)
    if folder.exists():
        raise MixforgeError(f"{folder} already exists; pick a new --out folder (nothing was overwritten)")
    _check_pcm_fits(result, subtype)
    dtype = np.float32 if subtype == "FLOAT" else np.float64
    folder.mkdir(parents=True)

    def put(path, arr):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), arr.astype(dtype), result.sr, subtype=subtype)

    put(folder / "mixture.wav", result.mixture)
    for name, a in result.stems.items():
        put(folder / "stems" / f"{name}.wav", a)
    if result.leaky is not None:
        for name, a in result.leaky.items():
            put(folder / "leaky" / f"{name}.wav", a)
    recipe = dict(result.recipe, subtype=subtype)
    _write_text(folder / "recipe.json", json.dumps(recipe, indent=2) + "\n")
    return folder


@dataclass
class DatasetResult:
    out: Path
    mixes: list[str] = field(default_factory=list)
    songs: int = 0
    notes: list[str] = field(default_factory=list)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "song"


def _append_index(idx_path: Path, rows: list):
    if idx_path.exists():
        old = idx_path.read_text(encoding="utf-8").splitlines()[1:]
        rows = [r for r in csv.reader(old)] + rows
    with open(idx_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["mix", "song", "seconds", "offset_samples", "sample_rate", "channels", "peak_scale", "bleed", "notes"])
        w.writerows(rows)


def make_dataset(source: Path, out: Path, *, count: int = 1, seconds: float | None = None,
                 gain_range=(0.0, 0.0), gains_db=None, bleed=(), seed: int = 0, subtype: str = "FLOAT",
                 sr: int | None = None, exclude=DEFAULT_EXCLUDE, peak_limit: float = 0.98,
                 max_songs: int | None = None) -> DatasetResult:
    """Make `count` mixes from every song found in `source`, written under `out`. Never touches `source`."""
    source, out = Path(source), Path(out)
    if count < 1:
        raise MixforgeError("--count must be at least 1")
    if subtype not in SUBTYPES:
        raise MixforgeError(f"--subtype must be one of {', '.join(SUBTYPES)}")
    songs, notes = find_songs(source, exclude)
    if max_songs is not None:
        if max_songs < 1:
            raise MixforgeError("--max-songs must be at least 1")
        songs = songs[:max_songs]
    src_r, out_r = source.resolve(), out.resolve()
    if out_r == src_r or src_r in out_r.parents:
        raise MixforgeError("--out must not be inside the stems folder")
    res = DatasetResult(out=out, notes=list(notes))
    rows = []
    try:
        for idx, (song, paths) in enumerate(songs):
            stems, rate, load_notes = load_song(paths, sr)
            for k in range(count):
                r = make_mix(stems, rate, gains_db=gains_db, gain_range=gain_range, bleed=bleed, seconds=seconds,
                             seed=[seed, idx, k], peak_limit=peak_limit, song=song, notes=load_notes)
                name = f"{idx:04d}_{_safe(song)}_{k:03d}"
                write_mix(r, out / name, subtype)
                res.mixes.append(name)
                rows.append([name, song, r.recipe["seconds"], r.recipe["offset_samples"], rate, r.recipe["channels"],
                             r.recipe["peak_scale"], ";".join(f"{b['source']}>{b['target']}:{b['requested_db']}"
                                                               for b in r.recipe["bleed"]), " | ".join(r.recipe["notes"])])
            res.songs += 1
    finally:
        if rows:
            _append_index(out / "index.csv", rows)
    return res
