"""The mixing itself: gains, clipping guard, cropping, and bleed. No file access here."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ._version import __version__


class MixforgeError(ValueError):
    """A problem the user can fix (bad input or option). The CLI shows it without a traceback."""


@dataclass(frozen=True)
class Bleed:
    """Leak part of `source` into `target`.

    `db` is the leak level relative to the source stem as it sits in the mix:
    -20 means the leak is 10 times quieter than the source (amplitude), -6 about half.
    `lowpass_hz` optionally removes the highs from the leak (real bleed is often duller).
    """
    source: str
    target: str
    db: float
    lowpass_hz: float | None = None


_BLEED_RE = re.compile(
    r"\s*([^>:@]+?)\s*>\s*([^>:@]+?)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(?:@\s*(\d+(?:\.\d+)?))?\s*")


def parse_bleed(text: str) -> Bleed:
    """Parse 'drums>vocals:-26' or 'drums>vocals:-26@4000' (optional low-pass in Hz)."""
    m = _BLEED_RE.fullmatch(text)
    if not m:
        raise MixforgeError(f"bad bleed {text!r}: use SOURCE>TARGET:DB, e.g. drums>vocals:-26 "
                            "(optionally @LOWPASS_HZ, e.g. drums>vocals:-26@4000)")
    src, dst, db, lp = m.groups()
    return Bleed(src, dst, float(db), float(lp) if lp else None)


def leak_signal(source: np.ndarray, db: float, lowpass_hz: float | None, sr: int) -> np.ndarray:
    """The audio that leaks: the source, scaled by `db`, optionally low-passed. Shape (samples, channels)."""
    leak = np.asarray(source, dtype=np.float64) * (10.0 ** (db / 20.0))
    if lowpass_hz is not None:
        if not (0 < lowpass_hz < sr / 2):
            raise MixforgeError(f"bleed low-pass {lowpass_hz} Hz must be between 0 and half the sample rate ({sr / 2:g})")
        sos = signal.butter(4, lowpass_hz, btype="low", fs=sr, output="sos")
        leak = signal.sosfilt(sos, leak, axis=0)
    return leak


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _db(x: float) -> float | None:
    return round(20.0 * math.log10(x), 4) if x > 1e-12 else None


@dataclass
class MixResult:
    mixture: np.ndarray                  # (samples, channels)
    stems: dict[str, np.ndarray]         # the answer key: clean stems exactly as they sit in the mixture
    leaky: dict[str, np.ndarray] | None  # same stems with the requested bleed added (None if no bleed)
    recipe: dict
    sr: int


def make_mix(stems: dict[str, np.ndarray], sr: int, *, gains_db: dict[str, float] | None = None,
             gain_range: tuple[float, float] = (0.0, 0.0), bleed=(), seconds: float | None = None,
             seed=0, peak_limit: float = 0.98, song: str = "", notes=()) -> MixResult:
    """Mix `stems` (name -> array of shape (samples,) or (samples, channels), all the same shape).

    The mixture is the exact sum of the returned stems. If that sum would be louder than `peak_limit`,
    every stem and the mixture are scaled down together, so the sum still holds.
    """
    if len(stems) < 2:
        raise MixforgeError("need at least 2 stems to make a mix")
    arrs = {}
    for name, a in stems.items():
        a = np.asarray(a)                        # converted to float64 only after cropping (saves memory on long songs)
        arrs[name] = a[:, None] if a.ndim == 1 else a
    shapes = {a.shape for a in arrs.values()}
    if len(shapes) != 1:
        raise MixforgeError(f"all stems must have the same length and channels, got {sorted(shapes)}")
    names = sorted(arrs)
    n, ch = next(iter(shapes))
    lo, hi = gain_range
    if lo > hi:
        raise MixforgeError(f"gain range {lo}..{hi} is backwards")
    gains_db = dict(gains_db or {})
    for k in gains_db:
        if k not in arrs:
            raise MixforgeError(f"gain given for unknown stem {k!r} (stems: {', '.join(names)})")
    bleed = list(bleed)
    seen = set()
    for b in bleed:
        for end in (b.source, b.target):
            if end not in arrs:
                raise MixforgeError(f"bleed uses unknown stem {end!r} (stems: {', '.join(names)})")
        if b.source == b.target:
            raise MixforgeError(f"bleed from {b.source!r} into itself makes no sense")
        if (b.source, b.target) in seen:
            raise MixforgeError(f"bleed {b.source}>{b.target} given twice")
        seen.add((b.source, b.target))
        if b.lowpass_hz is not None and not (0 < b.lowpass_hz < sr / 2):
            raise MixforgeError(f"bleed low-pass {b.lowpass_hz} Hz must be between 0 and half the sample rate ({sr / 2:g})")

    rng = np.random.default_rng(seed)
    notes = list(notes)

    # crop one aligned window out of every stem
    offset, n_use = 0, n
    if seconds is not None:
        want = int(round(seconds * sr))
        if want <= 0:
            raise MixforgeError("--seconds must be positive")
        if n > want:
            offset = int(rng.integers(0, n - want + 1))
            n_use = want
        else:
            notes.append(f"song is {n / sr:.2f} s, shorter than the {seconds:g} s asked for; used all of it")

    # gains (drawn in sorted-name order so a seed always gives the same result)
    gain_used = {}
    for name in names:
        g = rng.uniform(lo, hi)
        gain_used[name] = float(gains_db[name]) if name in gains_db else round(float(g), 4)
    gained = {name: arrs[name][offset:offset + n_use].astype(np.float64) * (10.0 ** (gain_used[name] / 20.0))
              for name in names}

    mixture = sum(gained[name] for name in names)
    peak = float(np.abs(mixture).max()) if mixture.size else 0.0
    scale = min(1.0, peak_limit / peak) if peak > 0 else 1.0
    if scale < 1.0:
        gained = {k: v * scale for k, v in gained.items()}
        mixture = sum(gained[name] for name in names)
        notes.append(f"mix would clip; everything scaled by {20 * math.log10(scale):.2f} dB")

    leaky, bleed_rec = None, []
    if bleed:
        leaky = {name: gained[name].copy() for name in names}
        for b in bleed:
            leak = leak_signal(gained[b.source], b.db, b.lowpass_hz, sr)
            leaky[b.target] = leaky[b.target] + leak
            bleed_rec.append({
                "source": b.source, "target": b.target, "requested_db": b.db, "lowpass_hz": b.lowpass_hz,
                "measured_db_vs_source": _db(_rms(leak) / _rms(gained[b.source])) if _rms(gained[b.source]) > 0 else None,
                "leak_vs_target_db": _db(_rms(leak) / _rms(gained[b.target])) if _rms(gained[b.target]) > 0 else None,
            })

    recipe = {
        "mixforge": __version__, "song": song, "seed": list(seed) if isinstance(seed, (list, tuple)) else seed,
        "sample_rate": sr, "channels": ch, "samples": int(n_use), "seconds": round(n_use / sr, 6),
        "offset_samples": offset, "source_samples": int(n),
        "peak_limit": peak_limit, "peak_scale": scale,
        "stems": {name: {"gain_db": gain_used[name],
                         "rms_dbfs": _db(_rms(gained[name])), "silent": _rms(gained[name]) == 0.0}
                  for name in names},
        "bleed": bleed_rec,
        "notes": notes,
    }
    return MixResult(mixture, gained, leaky, recipe, sr)
