"""Re-check a folder of mixes: is the mixture really the sum of the stems, and is the bleed really what the recipe says?"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .core import MixforgeError, leak_signal

_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}


@dataclass
class MixCheck:
    mix: str
    ok: bool
    problems: list[str] = field(default_factory=list)
    max_sum_error: float | None = None
    path: Path | None = None


@dataclass
class VerifyResult:
    checks: list[MixCheck]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _tolerance(subtype: str, n_items: int) -> float:
    """Biggest rounding error the file format can explain."""
    bits = _BITS.get(subtype)
    if bits is None:
        return 1e-5
    return max(1e-5, n_items / 2 ** (bits - 1))


def _read(path: Path):
    a, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return a, sr


def verify_mix(folder: Path) -> MixCheck:
    folder = Path(folder)
    problems: list[str] = []
    try:
        recipe = json.loads((folder / "recipe.json").read_text(encoding="utf-8"))
        names = list(recipe["stems"])
        sr, n, ch = int(recipe["sample_rate"]), int(recipe["samples"]), int(recipe["channels"])
    except Exception as e:
        return MixCheck(folder.name, False, [f"recipe.json missing or unreadable: {e}"], path=folder)

    def load(path: Path):
        if not path.exists():
            problems.append(f"missing file: {path.relative_to(folder)}")
            return None
        try:
            a, rate = _read(path)
        except Exception as e:
            problems.append(f"cannot read {path.relative_to(folder)}: {e}")
            return None
        if rate != sr or a.shape != (n, ch):
            problems.append(f"{path.relative_to(folder)} is {a.shape[0]} samples x {a.shape[1]} ch at {rate} Hz, "
                            f"recipe says {n} x {ch} at {sr}")
            return None
        return a

    mixture = load(folder / "mixture.wav")
    stems = {name: load(folder / "stems" / f"{name}.wav") for name in names}
    max_err = None
    if mixture is not None and all(a is not None for a in stems.values()):
        try:
            subtype = sf.info(str(folder / "mixture.wav")).subtype
        except Exception:
            subtype = "FLOAT"
        tol = _tolerance(subtype, len(names) + 1)
        max_err = float(np.abs(mixture - sum(stems.values())).max())
        if max_err > tol:
            problems.append(f"mixture is not the sum of the stems (largest difference {max_err:.2e}, allowed {tol:.2e})")

        bleed = recipe.get("bleed") or []
        if bleed:
            gsum = sum(10.0 ** (b["requested_db"] / 20.0) for b in bleed)
            tol_l = tol * (2 + 2 * gsum) + 1e-6
            expected = {name: a.copy() for name, a in stems.items()}
            try:
                for b in bleed:
                    expected[b["target"]] = expected[b["target"]] + leak_signal(
                        stems[b["source"]], b["requested_db"], b.get("lowpass_hz"), sr)
            except (KeyError, MixforgeError) as e:
                problems.append(f"recipe bleed entry is invalid: {e}")
                expected = None
            if expected is not None:
                for name in names:
                    got = load(folder / "leaky" / f"{name}.wav")
                    if got is None:
                        continue
                    err = float(np.abs(got - expected[name]).max())
                    if err > tol_l:
                        problems.append(f"leaky/{name}.wav does not match the recipe's bleed (largest difference {err:.2e})")
    return MixCheck(folder.name, not problems, problems, max_err, path=folder)


def verify_folder(folder: Path) -> VerifyResult:
    """`folder` is one mix (has recipe.json) or a folder of mixes (as written by `mixforge make`)."""
    folder = Path(folder)
    if not folder.is_dir():
        raise MixforgeError(f"{folder} is not a folder")
    if (folder / "recipe.json").exists():
        dirs = [folder]
    else:
        dirs = sorted(p for p in folder.iterdir() if p.is_dir() and (p / "recipe.json").exists())
    if not dirs:
        raise MixforgeError(f"no mixes found in {folder} (looking for folders that contain recipe.json)")
    return VerifyResult([verify_mix(d) for d in dirs])
