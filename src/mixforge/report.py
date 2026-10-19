"""HTML listening and verification report generator for mixforge."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .verify import VerifyResult


def to_html(res: VerifyResult, out_path: Path | None = None, base_dir: Path | None = None) -> str:
    total_mixes = len(res.checks)
    passed_mixes = sum(1 for c in res.checks if c.ok)
    failed_mixes = total_mixes - passed_mixes
    overall_ok = res.ok

    if overall_ok:
        badge_style = "background: linear-gradient(135deg, #10b981, #059669); color: white;"
        badge_text = "PASS &mdash; ALL MIXES MATHEMATICALLY VERIFIED"
        summary_desc = "Every mixture matches the exact linear sum of its answer-key stems within numerical precision."
    else:
        badge_style = "background: linear-gradient(135deg, #ef4444, #dc2626); color: white;"
        badge_text = "FAIL &mdash; VERIFICATION ANOMALIES DETECTED"
        summary_desc = f"{failed_mixes} mix(es) failed mathematical verification or bleed matching criteria."

    cards_html = []
    for c in res.checks:
        mix_dir = c.path
        if mix_dir is None and base_dir is not None:
            cand = base_dir / c.mix
            if cand.is_dir():
                mix_dir = cand
            elif (base_dir / "recipe.json").exists() and base_dir.name == c.mix:
                mix_dir = base_dir

        recipe = {}
        if mix_dir and (mix_dir / "recipe.json").exists():
            try:
                recipe = json.loads((mix_dir / "recipe.json").read_text(encoding="utf-8"))
            except Exception:
                recipe = {}

        def _rel_src(p: Path | None) -> str:
            if p is None or not p.exists():
                return ""
            if out_path:
                try:
                    return os.path.relpath(p, Path(out_path).parent)
                except Exception:
                    pass
            try:
                return p.as_uri()
            except Exception:
                return str(p)

        if c.ok:
            mix_status_badge = '<span style="background:#10b981;color:white;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">PASS</span>'
        else:
            mix_status_badge = '<span style="background:#ef4444;color:white;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">FAIL</span>'

        mix_wav = mix_dir / "mixture.wav" if mix_dir else None
        mix_src = _rel_src(mix_wav)

        sr = recipe.get("sample_rate", "N/A")
        sec = recipe.get("seconds", "N/A")
        sec_str = f"{sec:.2f}s" if isinstance(sec, (int, float)) else str(sec)
        ch = recipe.get("channels", "N/A")
        ch_str = "Stereo" if ch == 2 else ("Mono" if ch == 1 else str(ch))
        peak_scale = recipe.get("peak_scale", 1.0)
        peak_scale_str = f"{peak_scale:.4f}" if isinstance(peak_scale, (int, float)) else str(peak_scale)

        max_err_str = f"{c.max_sum_error:.2e}" if c.max_sum_error is not None else "N/A"

        stems_dict = recipe.get("stems", {})
        bleed_list = recipe.get("bleed", [])

        stems_html = []
        for stem_name, stem_info in stems_dict.items():
            clean_p = mix_dir / "stems" / f"{stem_name}.wav" if mix_dir else None
            clean_src = _rel_src(clean_p)
            gain_val = stem_info.get("gain_db", 0.0) if isinstance(stem_info, dict) else 0.0

            target_bleeds = [b for b in bleed_list if b.get("target") == stem_name]
            leaky_p = mix_dir / "leaky" / f"{stem_name}.wav" if mix_dir else None
            leaky_src = _rel_src(leaky_p)

            bleed_badges = []
            for b in target_bleeds:
                lp = f" @ {b.get('lowpass_hz')}Hz LP" if b.get("lowpass_hz") else ""
                bleed_badges.append(f'<span style="background:#3b82f6;color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;">+ {b.get("source")} bleed ({b.get("requested_db"):.1f} dB{lp})</span>')

            leaky_ui = ""
            if leaky_src and leaky_p and leaky_p.exists():
                badges_str = " ".join(bleed_badges)
                leaky_ui = f"""
                <div style="margin-top:10px;padding:8px;background:rgba(59,130,246,0.08);border-radius:6px;border-left:3px solid #3b82f6;">
                  <div style="font-size:0.78rem;font-weight:600;color:#93c5fd;margin-bottom:4px;">
                    Leaky Version (with Injected Bleed)
                    {badges_str}
                  </div>
                  <audio controls preload="none" src="{html.escape(leaky_src)}" style="width:100%;height:32px;"></audio>
                  <div style="font-size:0.7rem;color:#94a3b8;margin-top:2px;">Compare directly against clean {html.escape(stem_name)} above to audit bleed suppression.</div>
                </div>
                """

            stems_html.append(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;color:#f1f5f9;font-size:0.9rem;">{html.escape(stem_name.upper())}</span>
                <span style="font-size:0.75rem;color:#94a3b8;">Gain: {gain_val:+.1f} dB</span>
              </div>
              <div style="font-size:0.75rem;color:#cbd5e1;margin-bottom:4px;">Clean Answer-Key Stem:</div>
              <audio controls preload="none" src="{html.escape(clean_src)}" style="width:100%;height:32px;"></audio>
              {leaky_ui}
            </div>
            """)

        problems_html = ""
        if c.problems:
            prob_items = "".join(f"<li>{html.escape(p)}</li>" for p in c.problems)
            problems_html = f"""
            <div style="margin-top:12px;padding:10px;background:#450a0a;border-left:4px solid #ef4444;border-radius:6px;">
              <strong style="color:#fca5a5;font-size:0.85rem;">Verification Problems:</strong>
              <ul style="color:#fecaca;font-size:0.8rem;margin:4px 0 0 18px;">{prob_items}</ul>
            </div>
            """

        recipe_raw = html.escape(json.dumps(recipe, indent=2)) if recipe else "N/A"
        stems_joined = "".join(stems_html)

        card = f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.3);">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;border-bottom:1px solid #1e293b;padding-bottom:12px;margin-bottom:16px;">
            <div>
              <div style="display:flex;align-items:center;gap:10px;">
                <h3 style="margin:0;font-size:1.15rem;color:#f8fafc;">{html.escape(c.mix)}</h3>
                {mix_status_badge}
              </div>
              <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">
                Duration: <strong>{sec_str}</strong> &bull; Sample Rate: <strong>{sr} Hz</strong> &bull; Channels: <strong>{ch_str}</strong> &bull; Peak Scale: <strong>{peak_scale_str}</strong>
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:0.75rem;color:#94a3b8;">Max Sum Error:</div>
              <div style="font-family:monospace;font-size:0.85rem;color:#38bdf8;">{max_err_str}</div>
            </div>
          </div>

          <div style="margin-bottom:16px;background:#1e293b;border-radius:8px;padding:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <strong style="color:#f8fafc;font-size:0.9rem;">Full Mix (Sum of Stems):</strong>
              <span style="font-size:0.75rem;color:#64748b;">mixture.wav</span>
            </div>
            <audio controls preload="none" src="{html.escape(mix_src)}" style="width:100%;height:36px;"></audio>
          </div>

          <div style="margin-bottom:16px;">
            <h4 style="margin:0 0 10px 0;font-size:0.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">Isolated Stems & Bleed Inspection</h4>
            <div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(280px, 1fr));gap:12px;">
              {stems_joined}
            </div>
          </div>

          {problems_html}

          <details style="margin-top:12px;background:#090d16;border:1px solid #1e293b;border-radius:6px;padding:8px 12px;">
            <summary style="cursor:pointer;font-size:0.8rem;color:#64748b;font-weight:600;">Inspect Recipe JSON</summary>
            <pre style="margin:8px 0 0 0;font-family:monospace;font-size:0.75rem;color:#94a3b8;overflow-x:auto;">{recipe_raw}</pre>
          </details>
        </div>
        """
        cards_html.append(card)

    cards_joined = "".join(cards_html) if cards_html else '<div style="color:#94a3b8;padding:40px;text-align:center;">No mixes to display.</div>'
    failed_color = "#ef4444" if failed_mixes > 0 else "#64748b"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mix-Forge Verification &amp; Listening Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #020617;
      color: #f1f5f9;
      margin: 0;
      padding: 30px 20px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 25px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 20px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 15px;
    }}
    .badge {{
      display: inline-block;
      padding: 6px 16px;
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.85rem;
      letter-spacing: 0.05em;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .stat-card {{
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 8px;
      padding: 12px 16px;
    }}
    .stat-label {{
      font-size: 0.75rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .stat-value {{
      font-size: 1.4rem;
      font-weight: 700;
      color: #38bdf8;
      margin-top: 2px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="hero">
        <div>
          <h1 style="margin:0;font-size:1.75rem;color:#f8fafc;">Mix-Forge Inspection &amp; Verification Report</h1>
          <p style="margin:4px 0 0 0;color:#94a3b8;font-size:0.9rem;">Ground-Truth Multi-Track Listening &amp; Bleed Integrity Audit</p>
        </div>
        <div class="badge" style="{badge_style}">{badge_text}</div>
      </div>
      <p style="margin:12px 0 0 0;color:#cbd5e1;font-size:0.9rem;">{summary_desc}</p>
      
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-label">Total Mixes Verified</div>
          <div class="stat-value">{total_mixes}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Mathematical Passes</div>
          <div class="stat-value" style="color:#10b981;">{passed_mixes}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Failures / Anomalies</div>
          <div class="stat-value" style="color:{failed_color};">{failed_mixes}</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Answer-Key Guarantee</div>
          <div class="stat-value" style="font-size:1.1rem;color:#a855f7;">Mixture = &Sigma; Stems</div>
        </div>
      </div>
    </header>

    <main>
      {cards_joined}
    </main>

    <footer style="margin-top:40px;padding-top:20px;border-top:1px solid #1e293b;text-align:center;font-size:0.8rem;color:#64748b;">
      Generated by <strong>mixforge</strong> &bull; Developed by <strong>Boobalan Arjunan</strong> &bull; ComposerMind Labs
    </footer>
  </div>
</body>
</html>
"""
    return doc
