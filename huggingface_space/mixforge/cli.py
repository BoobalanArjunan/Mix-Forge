from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .core import MixforgeError, parse_bleed
from .data import DEFAULT_EXCLUDE, SUBTYPES, make_dataset
from .report import to_html
from .verify import verify_folder


def _gains(items):
    out = {}
    for it in items or []:
        name, sep, val = it.partition("=")
        try:
            if not sep or not name:
                raise ValueError
            out[name] = float(val)
        except ValueError:
            raise MixforgeError(f"bad --gain {it!r}: use STEM=DB, e.g. vocals=-3") from None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mixforge", description="Build test mixes with a known answer key.")
    ap.add_argument("--version", action="version", version=f"mixforge {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    m = sub.add_parser("make", help="mix stems into test songs, with an answer key and a recipe")
    m.add_argument("source", help="a folder of stem files (one song) or a folder of song folders")
    m.add_argument("--out", required=True, help="new folder to write the mixes into")
    m.add_argument("--count", type=int, default=1, help="mixes per song (default 1)")
    m.add_argument("--seconds", type=float, help="cut a random window of this length (same window for every stem)")
    m.add_argument("--gain-range", type=float, nargs=2, metavar=("LO", "HI"), default=(0.0, 0.0),
                   help="random level change per stem in dB, e.g. -6 6 (default: none)")
    m.add_argument("--gain", action="append", metavar="STEM=DB", help="fixed level for one stem; can repeat")
    m.add_argument("--bleed", action="append", metavar="SRC>DST:DB[@HZ]",
                   help="leak SRC into DST at DB (negative = quieter), optional low-pass; can repeat")
    m.add_argument("--max-songs", type=int, help="only use the first N songs (alphabetical), for a quick trial")
    m.add_argument("--seed", type=int, default=0, help="same seed gives the same mixes (default 0)")
    m.add_argument("--subtype", choices=SUBTYPES, default="FLOAT", help="audio format (default FLOAT, no clipping)")
    m.add_argument("--sr", type=int, help="sample rate to use (default: the most common one among the stems)")
    m.add_argument("--exclude", action="append", default=None,
                   help=f"file name to ignore (default: {', '.join(DEFAULT_EXCLUDE)}); can repeat")
    m.add_argument("--peak", type=float, default=0.98, help="loudest allowed peak before turning everything down (default 0.98)")
    m.add_argument("--json", action="store_true", help="print JSON instead of text")
    m.add_argument("--html", metavar="FILE", help="write an interactive HTML listening audit report")

    v = sub.add_parser("verify", help="re-check mixes: mixture = sum of stems, bleed = recipe")
    v.add_argument("folder", help="a folder written by 'mixforge make' (or one mix)")
    v.add_argument("--json", action="store_true", help="print JSON instead of text")
    v.add_argument("--html", metavar="FILE", help="write an interactive HTML listening audit report")

    a = ap.parse_args(argv)
    if a.cmd not in ("make", "verify"):
        ap.print_help()
        return 2
    try:
        if a.cmd == "make":
            res = make_dataset(
                Path(a.source), Path(a.out), count=a.count, seconds=a.seconds, gain_range=tuple(a.gain_range),
                gains_db=_gains(a.gain), bleed=[parse_bleed(b) for b in (a.bleed or [])], seed=a.seed,
                subtype=a.subtype, sr=a.sr, exclude=tuple(a.exclude) if a.exclude else DEFAULT_EXCLUDE,
                peak_limit=a.peak, max_songs=a.max_songs)
            if a.html:
                v_res = verify_folder(Path(a.out))
                html_content = to_html(v_res, out_path=Path(a.html), base_dir=Path(a.out))
                Path(a.html).parent.mkdir(parents=True, exist_ok=True)
                Path(a.html).write_text(html_content, encoding="utf-8")
            if a.json:
                print(json.dumps({"out": str(res.out), "songs": res.songs, "mixes": res.mixes, "notes": res.notes}, indent=2))
            else:
                print(f"wrote {len(res.mixes)} mix(es) from {res.songs} song(s) to {res.out}")
                for n in res.notes:
                    print(f"note: {n}")
                if a.html:
                    print(f"wrote HTML report to {a.html}")
                print(f"check them with: mixforge verify {res.out}")
            return 0
        res = verify_folder(Path(a.folder))
    except (MixforgeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.html:
        html_content = to_html(res, out_path=Path(a.html), base_dir=Path(a.folder))
        Path(a.html).parent.mkdir(parents=True, exist_ok=True)
        Path(a.html).write_text(html_content, encoding="utf-8")
    if a.json:
        print(json.dumps({"ok": res.ok, "checks": [{k: v for k, v in vars(c).items() if k != "path"} for c in res.checks]}, indent=2))
    else:
        for c in res.checks:
            print(f"{'PASS' if c.ok else 'FAIL'}  {c.mix}")
            for p in c.problems:
                print(f"      {p}")
        bad = sum(not c.ok for c in res.checks)
        print(f"{len(res.checks) - bad} of {len(res.checks)} mixes OK")
        if a.html:
            print(f"wrote HTML report to {a.html}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
