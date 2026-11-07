import json
import shutil

import numpy as np
import pytest
import soundfile as sf

import mixforge
from mixforge import Bleed, MixforgeError, make_dataset, make_mix, parse_bleed, verify_folder
from mixforge.cli import main
from conftest import STEM_KINDS, SR, db, read, rms, stem_audio, tree_hashes, write_song


def stems_dict(sec=3.0, sr=SR):
    return {k: stem_audio(k, sec, sr, i) for i, k in enumerate(STEM_KINDS)}


def recipe_of(out, mix):
    return json.loads((out / mix / "recipe.json").read_text())


# ---------- the core promise: mixture == sum of the answer-key stems ----------

def test_mixture_is_exact_sum_of_stems():
    r = make_mix(stems_dict(), SR, seed=1)
    assert np.abs(r.mixture - sum(r.stems.values())).max() < 1e-12


def test_default_gains_leave_stems_untouched():
    s = stems_dict()
    r = make_mix(s, SR)
    for k in s:
        assert np.allclose(r.stems[k][:, 0], s[k], atol=1e-7)
    assert r.recipe["peak_scale"] == 1.0
    assert all(v["gain_db"] == 0.0 for v in r.recipe["stems"].values())


def test_fixed_gain_changes_level_by_exactly_that_many_db():
    s = stems_dict()
    r = make_mix(s, SR, gains_db={"vocals": -6.0})
    assert db(rms(r.stems["vocals"]) / rms(s["vocals"])) == pytest.approx(-6.0, abs=1e-4)
    assert db(rms(r.stems["drums"]) / rms(s["drums"])) == pytest.approx(0.0, abs=1e-4)
    assert r.recipe["stems"]["vocals"]["gain_db"] == -6.0


def test_random_gains_stay_in_range_and_follow_seed():
    s = stems_dict()
    a = make_mix(s, SR, gain_range=(-6, 6), seed=5)
    b = make_mix(s, SR, gain_range=(-6, 6), seed=5)
    c = make_mix(s, SR, gain_range=(-6, 6), seed=6)
    ga = {k: v["gain_db"] for k, v in a.recipe["stems"].items()}
    gc = {k: v["gain_db"] for k, v in c.recipe["stems"].items()}
    assert all(-6 <= g <= 6 for g in ga.values())
    assert ga == {k: v["gain_db"] for k, v in b.recipe["stems"].items()}
    assert ga != gc
    assert len(set(ga.values())) > 1          # not all the same
    assert np.array_equal(a.mixture, b.mixture)


def test_gain_in_recipe_is_the_gain_actually_applied():
    s = stems_dict()
    r = make_mix(s, SR, gain_range=(-9, 3), seed=2)
    for k, v in r.recipe["stems"].items():
        assert db(rms(r.stems[k]) / rms(s[k])) == pytest.approx(v["gain_db"], abs=1e-3)


def test_clipping_guard_scales_everything_together():
    loud = {k: np.full(SR, 0.6, dtype=np.float32) * (1 if k != "bass" else -0.2) for k in ["vocals", "drums", "bass", "other"]}
    loud["vocals"] = np.sin(np.linspace(0, 200, SR)).astype(np.float32) * 0.9
    loud["drums"] = np.sin(np.linspace(0, 77, SR)).astype(np.float32) * 0.9
    loud["other"] = np.sin(np.linspace(0, 31, SR)).astype(np.float32) * 0.9
    raw_peak = np.abs(sum(loud.values())).max()
    assert raw_peak > 1.0                       # the test only means something if it would clip
    r = make_mix(loud, SR, peak_limit=0.98)
    assert np.abs(r.mixture).max() <= 0.98 + 1e-9
    assert np.abs(r.mixture - sum(r.stems.values())).max() < 1e-12     # sum still holds
    scale = r.recipe["peak_scale"]
    assert scale == pytest.approx(0.98 / raw_peak, rel=1e-9)
    for k in loud:                                                      # same scale on every stem
        assert rms(r.stems[k]) / rms(loud[k]) == pytest.approx(scale, rel=1e-6)
    assert any("scaled" in n for n in r.recipe["notes"])


# ---------- bleed ----------

def test_bleed_amount_matches_request_exactly():
    r = make_mix(stems_dict(), SR, bleed=[Bleed("drums", "vocals", -20.0)])
    expect = r.stems["vocals"] + 0.1 * r.stems["drums"]
    assert np.abs(r.leaky["vocals"] - expect).max() < 1e-12
    b = r.recipe["bleed"][0]
    assert b["measured_db_vs_source"] == pytest.approx(-20.0, abs=0.001)
    assert b["leak_vs_target_db"] == pytest.approx(db(0.1 * rms(r.stems["drums"]) / rms(r.stems["vocals"])), abs=0.001)


def test_bleed_only_touches_the_target_and_not_the_answer_key():
    r = make_mix(stems_dict(), SR, bleed=[Bleed("drums", "vocals", -12.0)])
    for k in ("drums", "bass", "other"):
        assert np.array_equal(r.leaky[k], r.stems[k])
    assert not np.allclose(r.leaky["vocals"], r.stems["vocals"])
    assert np.abs(r.mixture - sum(r.stems.values())).max() < 1e-12     # mixture built from CLEAN stems


def test_bleed_follows_gain_and_clip_scaling():
    r = make_mix(stems_dict(), SR, gains_db={"drums": -10.0}, bleed=[Bleed("drums", "vocals", -6.0)])
    leak = r.leaky["vocals"] - r.stems["vocals"]
    assert rms(leak) / rms(r.stems["drums"]) == pytest.approx(10 ** (-6 / 20), rel=1e-6)


def test_lowpass_bleed_removes_highs_and_lowers_measured_level():
    s = stems_dict()
    plain = make_mix(s, SR, bleed=[Bleed("drums", "bass", -10.0)])
    dull = make_mix(s, SR, bleed=[Bleed("drums", "bass", -10.0, lowpass_hz=1000.0)])
    leak_plain = plain.leaky["bass"] - plain.stems["bass"]
    leak_dull = dull.leaky["bass"] - dull.stems["bass"]

    def high_share(x):                      # fraction of energy above 3 kHz
        spec = np.abs(np.fft.rfft(x[:, 0])) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return spec[f > 3000].sum() / spec.sum()
    assert high_share(leak_plain) > 0.2                  # drums really have highs
    assert high_share(leak_dull) < 0.001                 # and the low-pass removed them
    assert dull.recipe["bleed"][0]["measured_db_vs_source"] < plain.recipe["bleed"][0]["measured_db_vs_source"] - 1
    assert dull.recipe["bleed"][0]["lowpass_hz"] == 1000.0


def test_several_bleeds_into_one_target_add_up():
    r = make_mix(stems_dict(), SR, bleed=[Bleed("drums", "vocals", -20.0), Bleed("bass", "vocals", -14.0)])
    expect = r.stems["vocals"] + 0.1 * r.stems["drums"] + 10 ** (-14 / 20) * r.stems["bass"]
    assert np.abs(r.leaky["vocals"] - expect).max() < 1e-12
    assert len(r.recipe["bleed"]) == 2


def test_bleed_from_silent_source_is_reported_as_none():
    s = stems_dict()
    s["bass"] = np.zeros_like(s["bass"])
    r = make_mix(s, SR, bleed=[Bleed("bass", "vocals", -20.0)])
    assert r.recipe["bleed"][0]["measured_db_vs_source"] is None
    assert r.recipe["stems"]["bass"]["silent"] is True
    assert np.array_equal(r.leaky["vocals"], r.stems["vocals"])


@pytest.mark.parametrize("text,expected", [
    ("drums>vocals:-26", Bleed("drums", "vocals", -26.0)),
    ("drums > vocals : -26.5 @ 4000", Bleed("drums", "vocals", -26.5, 4000.0)),
    ("bass>other:+3", Bleed("bass", "other", 3.0)),
])
def test_parse_bleed(text, expected):
    assert parse_bleed(text) == expected


@pytest.mark.parametrize("text", ["", "drums vocals -26", "drums>vocals", "drums>vocals:abc", ">vocals:-3", "a>b>c:-3"])
def test_parse_bleed_rejects_nonsense(text):
    with pytest.raises(MixforgeError):
        parse_bleed(text)


def test_bleed_validation():
    s = stems_dict()
    with pytest.raises(MixforgeError, match="unknown stem"):
        make_mix(s, SR, bleed=[Bleed("guitar", "vocals", -20)])
    with pytest.raises(MixforgeError, match="itself"):
        make_mix(s, SR, bleed=[Bleed("drums", "drums", -20)])
    with pytest.raises(MixforgeError, match="twice"):
        make_mix(s, SR, bleed=[Bleed("drums", "vocals", -20), Bleed("drums", "vocals", -10)])
    with pytest.raises(MixforgeError, match="low-pass"):
        make_mix(s, SR, bleed=[Bleed("drums", "vocals", -20, lowpass_hz=SR)])
    with pytest.raises(MixforgeError, match="unknown stem"):
        make_mix(s, SR, gains_db={"guitar": 3})
    with pytest.raises(MixforgeError, match="backwards"):
        make_mix(s, SR, gain_range=(3, -3))
    with pytest.raises(MixforgeError, match="at least 2"):
        make_mix({"vocals": s["vocals"]}, SR)
    with pytest.raises(MixforgeError, match="same length"):
        make_mix({"a": s["vocals"], "b": s["drums"][:100]}, SR)


# ---------- cropping ----------

def test_crop_takes_the_same_window_from_every_stem(tmp_path):
    song = write_song(tmp_path / "song", sec=5.0)
    out = tmp_path / "out"
    make_dataset(song, out, seconds=2.0, seed=3)
    rec = recipe_of(out, "0000_song_000")
    off, n = rec["offset_samples"], rec["samples"]
    assert n == 2 * SR and rec["seconds"] == 2.0
    assert 0 < off <= 3 * SR                               # actually moved (seed 3 is not offset 0)
    for k in STEM_KINDS:
        orig, _ = read(song / f"{k}.wav")
        got, _ = read(out / "0000_song_000" / "stems" / f"{k}.wav")
        assert np.allclose(got, orig[off:off + n], atol=1e-6)


def test_song_shorter_than_requested_uses_everything_and_says_so(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0)
    out = tmp_path / "out"
    make_dataset(song, out, seconds=5.0)
    rec = recipe_of(out, "0000_song_000")
    assert rec["samples"] == SR and rec["offset_samples"] == 0
    assert any("shorter" in n for n in rec["notes"])


def test_identical_songs_in_different_folders_still_get_different_random_choices(tmp_path):
    write_song(tmp_path / "src" / "a", sec=4.0)
    write_song(tmp_path / "src" / "b", sec=4.0)            # exactly the same audio as "a"
    out = tmp_path / "out"
    make_dataset(tmp_path / "src", out, seconds=1.0, gain_range=(-6, 6), seed=1)
    ra, rb = recipe_of(out, "0000_a_000"), recipe_of(out, "0001_b_000")
    assert ra["seed"] != rb["seed"]
    assert (ra["offset_samples"], ra["stems"]) != (rb["offset_samples"], rb["stems"])


def test_different_mixes_of_one_song_use_different_windows_and_gains(tmp_path):
    song = write_song(tmp_path / "song", sec=6.0)
    out = tmp_path / "out"
    res = make_dataset(song, out, count=3, seconds=1.5, gain_range=(-6, 6), seed=1)
    recs = [recipe_of(out, m) for m in res.mixes]
    assert len({r["offset_samples"] for r in recs}) > 1
    assert len({json.dumps(r["stems"], sort_keys=True) for r in recs}) == 3


# ---------- matching up real-world stems ----------

def test_mixed_sample_rates_are_matched_to_the_most_common(tmp_path):
    song = write_song(tmp_path / "song", sec=2.0, sr=16000)
    sf.write(str(song / "bass.wav"), stem_audio("bass", 2.0, 8000), 8000)
    out = tmp_path / "out"
    make_dataset(song, out)
    rec = recipe_of(out, "0000_song_000")
    assert rec["sample_rate"] == 16000
    assert any("resampled" in n and "bass" in n for n in rec["notes"])
    bass, sr = read(out / "0000_song_000" / "stems" / "bass.wav")
    assert sr == 16000 and abs(len(bass) - 2 * 16000) <= 1
    spec = np.abs(np.fft.rfft(bass[:, 0]))
    assert np.fft.rfftfreq(len(bass), 1 / 16000)[spec.argmax()] == pytest.approx(55, abs=1.0)   # still the same 55 Hz tone
    assert verify_folder(out).ok


def test_mono_and_stereo_stems_are_matched_to_stereo(tmp_path):
    song = write_song(tmp_path / "song", sec=2.0, channels=2)
    sf.write(str(song / "bass.wav"), stem_audio("bass", 2.0), SR)           # one mono stem
    out = tmp_path / "out"
    make_dataset(song, out)
    rec = recipe_of(out, "0000_song_000")
    assert rec["channels"] == 2 and any("mono" in n and "bass" in n for n in rec["notes"])
    bass, _ = read(out / "0000_song_000" / "stems" / "bass.wav")
    assert bass.shape[1] == 2 and np.array_equal(bass[:, 0], bass[:, 1])
    original, _ = read(song / "bass.wav")
    assert np.allclose(bass[:, 0], original[:, 0], atol=1e-6) and rms(bass) > 0.05   # same sound, in both channels
    vocals, _ = read(out / "0000_song_000" / "stems" / "vocals.wav")
    assert not np.array_equal(vocals[:, 0], vocals[:, 1])                   # real stereo untouched
    assert verify_folder(out).ok


def test_two_and_six_channels_cannot_be_matched(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0, channels=2)
    sf.write(str(song / "bass.wav"), np.zeros((SR, 6), dtype=np.float32), SR)
    with pytest.raises(MixforgeError, match="channels"):
        make_dataset(song, tmp_path / "out")


def test_different_lengths_are_trimmed_to_the_shortest(tmp_path):
    song = write_song(tmp_path / "song", sec=3.0)
    sf.write(str(song / "bass.wav"), stem_audio("bass", 2.0), SR)
    out = tmp_path / "out"
    make_dataset(song, out)
    rec = recipe_of(out, "0000_song_000")
    assert rec["samples"] == 2 * SR
    assert any("different lengths" in n for n in rec["notes"])
    assert verify_folder(out).ok


def test_silent_stem_is_allowed(tmp_path):
    song = write_song(tmp_path / "song", sec=2.0)
    sf.write(str(song / "vocals.wav"), np.zeros(2 * SR, dtype=np.float32), SR)
    out = tmp_path / "out"
    make_dataset(song, out, bleed=[Bleed("drums", "vocals", -20)])
    rec = recipe_of(out, "0000_song_000")
    assert rec["stems"]["vocals"]["silent"] is True
    assert rec["bleed"][0]["leak_vs_target_db"] is None          # nothing to compare the leak with
    assert verify_folder(out).ok
    leaky, _ = read(out / "0000_song_000" / "leaky" / "vocals.wav")
    assert rms(leaky) > 0                                        # the leak is the only thing in that track


# ---------- folders, names, safety ----------

def test_musdb_layout_ignores_the_mixture_file(tmp_path):
    song = write_song(tmp_path / "src" / "Artist - Song", sec=2.0)
    sf.write(str(song / "mixture.wav"), stem_audio("vocals", 2.0), SR)
    out = tmp_path / "out"
    res = make_dataset(tmp_path / "src", out)
    assert res.songs == 1
    assert sorted(recipe_of(out, res.mixes[0])["stems"]) == sorted(STEM_KINDS)
    assert res.mixes[0] == "0000_Artist_-_Song_000"                 # odd characters made safe


def test_exclude_option_and_custom_names(tmp_path):
    song = write_song(tmp_path / "song", sec=2.0)
    out = tmp_path / "out"
    make_dataset(song, out, exclude=("other", "bass"))
    assert sorted(recipe_of(out, "0000_song_000")["stems"]) == ["drums", "vocals"]


def test_folder_of_songs_gives_every_song_its_mixes(tmp_path):
    for i, name in enumerate(["a", "b", "c"]):
        write_song(tmp_path / "src" / name, sec=2.0, seed=10 * i)
    out = tmp_path / "out"
    res = make_dataset(tmp_path / "src", out, count=2, gain_range=(-3, 3))
    assert res.songs == 3 and len(res.mixes) == 6 and len(set(res.mixes)) == 6
    lines = (out / "index.csv").read_text().splitlines()
    assert len(lines) == 7 and lines[0].startswith("mix,song")
    a, _ = read(out / res.mixes[0] / "mixture.wav")
    c, _ = read(out / res.mixes[4] / "mixture.wav")
    assert not np.allclose(a, c)
    assert verify_folder(out).ok


def test_max_songs_limits_the_run(tmp_path):
    for name in ["a", "b", "c"]:
        write_song(tmp_path / "src" / name, sec=1.0)
    res = make_dataset(tmp_path / "src", tmp_path / "out", max_songs=2)
    assert res.songs == 2 and [m.split("_")[1] for m in res.mixes] == ["a", "b"]
    with pytest.raises(MixforgeError, match="max-songs"):
        make_dataset(tmp_path / "src", tmp_path / "out2", max_songs=0)


def test_folders_with_too_few_stems_are_skipped_with_a_note(tmp_path):
    write_song(tmp_path / "src" / "good", sec=1.0)
    write_song(tmp_path / "src" / "lonely", sec=1.0, kinds=["vocals"])
    res = make_dataset(tmp_path / "src", tmp_path / "out")
    assert res.songs == 1 and any("lonely" in n for n in res.notes)


def test_no_songs_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MixforgeError, match="no songs found"):
        make_dataset(tmp_path / "empty", tmp_path / "out")
    with pytest.raises(MixforgeError, match="not a folder"):
        make_dataset(tmp_path / "missing", tmp_path / "out")


def test_unreadable_stem_is_a_clear_error(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0)
    (song / "bass.wav").write_bytes(b"this is not audio")
    with pytest.raises(MixforgeError, match="cannot read bass.wav"):
        make_dataset(song, tmp_path / "out")


def test_source_files_are_never_modified(tmp_path):
    song = write_song(tmp_path / "src" / "song", sec=2.0)
    before = tree_hashes(tmp_path / "src")
    make_dataset(tmp_path / "src", tmp_path / "out", count=2, gain_range=(-6, 6), bleed=[Bleed("drums", "vocals", -20)])
    assert tree_hashes(tmp_path / "src") == before


def test_out_inside_source_is_refused(tmp_path):
    write_song(tmp_path / "src" / "song", sec=1.0)
    with pytest.raises(MixforgeError, match="inside"):
        make_dataset(tmp_path / "src", tmp_path / "src" / "out")
    with pytest.raises(MixforgeError, match="inside"):
        make_dataset(tmp_path / "src" / "song", tmp_path / "src" / "song" / "out")


def test_existing_output_is_never_overwritten(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0)
    out = tmp_path / "out"
    make_dataset(song, out)
    before = tree_hashes(out)
    with pytest.raises(MixforgeError, match="already exists"):
        make_dataset(song, out)
    assert tree_hashes(out) == before


def test_same_seed_gives_identical_files_in_a_new_folder(tmp_path):
    song = write_song(tmp_path / "song", sec=3.0)
    kw = dict(count=2, seconds=1.0, gain_range=(-6, 6), seed=9, bleed=[Bleed("drums", "vocals", -20, 2000.0)])
    make_dataset(song, tmp_path / "one", **kw)
    make_dataset(song, tmp_path / "two", **kw)
    files_one = {p.relative_to(tmp_path / "one").as_posix(): p for p in sorted((tmp_path / "one").rglob("*")) if p.is_file()}
    files_two = {p.relative_to(tmp_path / "two").as_posix(): p for p in sorted((tmp_path / "two").rglob("*")) if p.is_file()}
    assert set(files_one.keys()) == set(files_two.keys())
    for rel, p1 in files_one.items():
        p2 = files_two[rel]
        if rel.endswith(".wav"):
            a1, sr1 = read(p1)
            a2, sr2 = read(p2)
            assert sr1 == sr2
            assert np.allclose(a1, a2, atol=1e-5)
        else:
            assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    make_dataset(song, tmp_path / "three", **{**kw, "seed": 10})
    files_three = {p.relative_to(tmp_path / "three").as_posix(): p for p in sorted((tmp_path / "three").rglob("*")) if p.is_file()}
    # Different seeds produce different mixtures
    a1, _ = read(files_one["0000_song_000/mixture.wav"])
    a3, _ = read(files_three["0000_song_000/mixture.wav"])
    assert not np.allclose(a1, a3, atol=1e-4)


# ---------- formats ----------

def test_pcm16_files_verify_within_rounding(tmp_path):
    song = write_song(tmp_path / "song", sec=2.0)
    out = tmp_path / "out"
    make_dataset(song, out, subtype="PCM_16", gain_range=(-6, 0), bleed=[Bleed("drums", "vocals", -20, 3000.0)])
    assert sf.info(str(out / "0000_song_000" / "mixture.wav")).subtype == "PCM_16"
    assert recipe_of(out, "0000_song_000")["subtype"] == "PCM_16"
    assert verify_folder(out).ok


def test_pcm_refuses_audio_that_would_clip_but_float_allows_it(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0)
    sf.write(str(song / "vocals.wav"), np.full(SR, 0.9, dtype=np.float32), SR)
    sf.write(str(song / "drums.wav"), np.full(SR, 0.05, dtype=np.float32), SR)
    bleed = [Bleed("vocals", "drums", 6.0)]                       # leak louder than the source: leaky drums exceeds 1.0
    with pytest.raises(MixforgeError, match="PCM_16"):
        make_dataset(song, tmp_path / "bad", subtype="PCM_16", bleed=bleed, peak_limit=0.99)
    assert not (tmp_path / "bad" / "0000_song_000").exists()      # nothing half-written
    make_dataset(song, tmp_path / "ok", subtype="FLOAT", bleed=bleed, peak_limit=0.99)
    assert verify_folder(tmp_path / "ok").ok


# ---------- verify catches tampering ----------

def _made(tmp_path, **kw):
    song = write_song(tmp_path / "song", sec=2.0)
    out = tmp_path / "out"
    make_dataset(song, out, bleed=kw.pop("bleed", [Bleed("drums", "vocals", -20, 3000.0)]), **kw)
    return out, out / "0000_song_000"


def test_verify_passes_on_untouched_output(tmp_path):
    out, _ = _made(tmp_path, count=2, gain_range=(-6, 6))
    res = verify_folder(out)
    assert res.ok and len(res.checks) == 2 and all(c.max_sum_error < 1e-6 for c in res.checks)


def test_verify_catches_a_changed_mixture(tmp_path):
    out, mix = _made(tmp_path)
    a, sr = read(mix / "mixture.wav")
    a[1000:1100] += 0.01
    sf.write(str(mix / "mixture.wav"), a.astype(np.float32), sr, subtype="FLOAT")
    res = verify_folder(out)
    assert not res.ok and "not the sum" in res.checks[0].problems[0]


def test_verify_catches_a_small_change_in_a_pcm16_mixture(tmp_path):
    out, mix = _made(tmp_path, subtype="PCM_16")
    assert verify_folder(out).ok
    a, sr = read(mix / "mixture.wav")
    a[2000:2200] += 0.002                                  # about 65 steps of 16-bit rounding: far more than rounding noise
    sf.write(str(mix / "mixture.wav"), a, sr, subtype="PCM_16")
    res = verify_folder(out)
    assert not res.ok and "not the sum" in res.checks[0].problems[0]


def test_verify_catches_a_changed_stem(tmp_path):
    out, mix = _made(tmp_path)
    a, sr = read(mix / "stems" / "bass.wav")
    sf.write(str(mix / "stems" / "bass.wav"), (a * 0.5).astype(np.float32), sr, subtype="FLOAT")
    assert not verify_folder(out).ok


def test_verify_catches_a_changed_leaky_file(tmp_path):
    out, mix = _made(tmp_path)
    a, sr = read(mix / "leaky" / "vocals.wav")
    sf.write(str(mix / "leaky" / "vocals.wav"), (a + 0.001).astype(np.float32), sr, subtype="FLOAT")
    res = verify_folder(out)
    assert not res.ok and any("leaky/vocals.wav" in p for p in res.checks[0].problems)


def test_verify_catches_a_recipe_that_lies_about_bleed(tmp_path):
    out, mix = _made(tmp_path)
    rec = json.loads((mix / "recipe.json").read_text())
    rec["bleed"][0]["requested_db"] = -10.0              # audio was made at -20
    (mix / "recipe.json").write_text(json.dumps(rec))
    res = verify_folder(out)
    assert not res.ok and any("recipe's bleed" in p for p in res.checks[0].problems)


def test_verify_catches_missing_files_and_bad_recipe(tmp_path):
    out, mix = _made(tmp_path)
    (mix / "stems" / "drums.wav").unlink()
    assert any("missing file" in p for p in verify_folder(out).checks[0].problems)
    (mix / "recipe.json").write_text("{not json")
    assert "recipe.json" in verify_folder(out).checks[0].problems[0]


def test_verify_on_empty_folder_is_a_clear_error(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(MixforgeError, match="no mixes"):
        verify_folder(tmp_path / "e")


# ---------- command line ----------

def test_cli_make_then_verify(tmp_path, capsys):
    song = write_song(tmp_path / "song", sec=2.0)
    out = tmp_path / "out"
    rc = main(["make", str(song), "--out", str(out), "--count", "2", "--gain-range", "-6", "6",
               "--bleed", "drums>vocals:-26@4000", "--gain", "bass=-3", "--seed", "4", "--seconds", "1"])
    assert rc == 0 and "wrote 2 mix(es)" in capsys.readouterr().out
    rec = recipe_of(out, "0000_song_000")
    assert rec["stems"]["bass"]["gain_db"] == -3.0 and rec["bleed"][0]["lowpass_hz"] == 4000.0
    assert main(["verify", str(out)]) == 0
    assert "2 of 2 mixes OK" in capsys.readouterr().out


def test_cli_verify_exit_code_1_on_tamper(tmp_path, capsys):
    out, mix = _made(tmp_path)
    (mix / "stems" / "bass.wav").write_bytes((mix / "stems" / "drums.wav").read_bytes())
    assert main(["verify", str(out)]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    song = write_song(tmp_path / "song", sec=1.0)
    assert main(["make", str(song), "--out", str(tmp_path / "out"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["mixes"] == ["0000_song_000"]
    assert main(["verify", str(tmp_path / "out"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


@pytest.mark.parametrize("argv", [
    [],
    ["make"],
    ["make", "nope", "--out", "x"],
    ["verify", "nope"],
])
def test_cli_usage_errors_exit_2_without_traceback(argv, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        rc = main(argv)
    except SystemExit as e:        # argparse's own "missing argument" exit
        rc = e.code
    assert rc == 2


def test_cli_bad_options_exit_2(tmp_path, capsys):
    song = write_song(tmp_path / "song", sec=1.0)
    base = ["make", str(song), "--out", str(tmp_path / "out")]
    for extra in (["--bleed", "garbage"], ["--gain", "vocals"], ["--bleed", "x>y:-3"], ["--count", "0"],
                  ["--seconds", "-1"]):
        assert main(base + extra) == 2, extra
        assert "error:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and mixforge.__version__ in capsys.readouterr().out


def test_cli_html_report_generation(tmp_path):
    song = write_song(tmp_path / "song", sec=1.0)
    out = tmp_path / "out"
    html_out = tmp_path / "reports" / "mix_report.html"
    rc = main(["make", str(song), "--out", str(out), "--bleed", "drums>vocals:-20", "--html", str(html_out)])
    assert rc == 0
    assert html_out.exists()
    content = html_out.read_text(encoding="utf-8")
    assert "Mix-Forge Inspection &amp; Verification Report" in content
    assert "mixture.wav" in content
    assert "PASS" in content
    assert "Leaky Version (with Injected Bleed)" in content

    # Test verify --html
    verify_html = tmp_path / "reports" / "verify_report.html"
    rc_v = main(["verify", str(out), "--html", str(verify_html)])
    assert rc_v == 0
    assert verify_html.exists()
    v_content = verify_html.read_text(encoding="utf-8")
    assert "ALL MIXES MATHEMATICALLY VERIFIED" in v_content
