"""Tiny synthetic stems with known properties, so every test knows the right answer."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def stem_audio(kind: str, sec: float = 3.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    r = np.random.RandomState(seed)
    n = int(sr * sec)
    t = np.arange(n) / sr
    if kind == "vocals":      # moving tone with harmonics, gated like words
        f0 = 150 + 40 * np.sin(2 * np.pi * 0.7 * t)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        x = sum(np.sin(k * ph) / k for k in range(1, 10))
        x = x * (np.sin(2 * np.pi * 1.3 * t) > -0.3)
        return (0.2 * x / np.abs(x).max()).astype(np.float32)
    if kind == "drums":       # noise bursts: lots of high-frequency energy
        env = np.exp(-((t * 4) % 1.0) * 12)
        return (0.1 * r.randn(n) * env).clip(-1, 1).astype(np.float32)
    if kind == "bass":        # low sine
        return (0.2 * np.sin(2 * np.pi * 55 * t)).astype(np.float32)
    if kind == "other":       # chord
        return (0.1 * (np.sin(2 * np.pi * 330 * t) + np.sin(2 * np.pi * 415 * t) + np.sin(2 * np.pi * 495 * t))).astype(np.float32)
    raise ValueError(kind)


STEM_KINDS = ["vocals", "drums", "bass", "other"]


def write_song(folder: Path, sec: float = 3.0, sr: int = SR, kinds=STEM_KINDS, channels=1, subtype="FLOAT", seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(kinds):
        a = stem_audio(k, sec, sr, seed + i)
        if channels == 2:
            a = np.stack([a, a * 0.8], axis=1)
        sf.write(str(folder / f"{k}.wav"), a, sr, subtype=subtype)
    return folder


def read(path):
    a, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return a, sr


def tree_hashes(root: Path):
    return {p.relative_to(root).as_posix(): hashlib.sha1(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def db(x):
    return 20 * np.log10(x)
