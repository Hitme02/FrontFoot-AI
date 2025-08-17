# advanced_addons.py
"""
Lightweight "bonus" helpers:
- reference loading + comparison
- phase segmentation (heuristic)
- contact-moment detection (heuristic)
- temporal smoothness + chart export
- grade prediction
- HTML report

These are intentionally simple so they run everywhere without extra models.
"""

from __future__ import annotations
import os, json, math
from typing import Dict, List, Tuple, Optional

import numpy as np

# Use a headless backend for servers/CI
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt


# -----------------------------
# Reference config
# -----------------------------
def load_reference(path: str) -> Dict[str, Tuple[float, float]]:
    """
    Load reference angle ranges. If not found, return sane defaults.
    """
    defaults = {
        "elbow_deg": [90, 135],
        "spine_lean_deg": [10, 25],
        "head_knee_gap_norm": [0.0, 0.30],   # smaller is better
        "foot_angle_deg": [0, 20]
    }
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
                return cfg or defaults
        except Exception:
            return defaults
    return defaults


# -----------------------------
# Phase segmentation (heuristic)
# -----------------------------
def _safe_diff(x: List[float]) -> np.ndarray:
    a = np.array([np.nan if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in x], dtype=np.float32)
    good = np.nan_to_num(a, nan=np.nanmedian(a) if not np.isnan(np.nanmedian(a)) else 0.0)
    d = np.abs(np.diff(good))
    return d


def segment_phases(series: Dict[str, List[float]], fps: float) -> List[Dict[str, int]]:
    """
    Very rough segmentation based on elbow/spine velocity.
    Returns list of {name, start, end} in frame indices.
    """
    n = max(len(v) for v in series.values())
    if n < 5:
        return [{"name":"Unknown", "start":0, "end":max(0,n-1)}]

    elbow_d = _safe_diff(series.get("elbow_deg", [np.nan]*n))
    spine_d = _safe_diff(series.get("spine_lean_deg", [np.nan]*n))
    vel = (elbow_d + spine_d) / 2.0
    thr = np.percentile(vel, 60)

    # simple states by velocity level
    states = []
    cur = "Stance"
    s = 0
    for i, v in enumerate(vel):
        label = "Downswing" if v >= thr else "Stride"
        if i < 3: label = "Stance"
        if i > len(vel)-3: label = "Follow-through"
        if i == 0:
            cur = label
            s = 0
            continue
        if label != cur:
            states.append({"name":cur, "start":s, "end":i})
            cur = label
            s = i
    states.append({"name":cur, "start":s, "end":len(vel)})

    # clean up names + a "Recovery" tail
    if states:
        states[0]["name"] = "Stance"
        states[-1]["name"] = "Follow-through"
        # add a short Recovery if video goes beyond follow-through
        ft_end = states[-1]["end"]
        if ft_end < n-1:
            states.append({"name":"Recovery", "start":ft_end, "end":n-1})
    return states


# -----------------------------
# Contact-moment detection (heuristic)
# -----------------------------
def detect_contact(series: Dict[str, List[float]], fps: float) -> Optional[int]:
    """
    Pick the frame with minimum head-knee gap (approx 'stacked' moment),
    else max elbow velocity.
    """
    n = max(len(v) for v in series.values())
    gap = np.array(series.get("head_knee_gap_norm", [np.nan]*n), dtype=np.float32)
    if np.any(~np.isnan(gap)):
        idx = int(np.nanargmin(gap))
        return idx

    elbow = series.get("elbow_deg", [np.nan]*n)
    vel = _safe_diff(elbow)
    if len(vel):
        return int(np.argmax(vel))
    return None


# -----------------------------
# Smoothness + chart
# -----------------------------
def export_timechart(series: Dict[str, List[float]], fps: float, out_dir: str) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Plot elbow & spine over time; compute simple smoothness:
      mean(|Δangle|) for each signal (lower is smoother).
    """
    os.makedirs(out_dir, exist_ok=True)
    n = max(len(v) for v in series.values())
    t = np.arange(n) / (fps if fps else 30.0)

    elbow = np.array(series.get("elbow_deg", [np.nan]*n))
    spine = np.array(series.get("spine_lean_deg", [np.nan]*n))

    e_diff = np.nanmean(np.abs(np.diff(elbow))) if np.any(~np.isnan(elbow)) else np.nan
    s_diff = np.nanmean(np.abs(np.diff(spine))) if np.any(~np.isnan(spine)) else np.nan
    smooth = {"elbow_mean_abs_delta": float(e_diff if not np.isnan(e_diff) else 0.0),
              "spine_mean_abs_delta": float(s_diff if not np.isnan(s_diff) else 0.0)}

    try:
        plt.figure(figsize=(8, 3))
        if np.any(~np.isnan(elbow)):
            plt.plot(t, elbow, label="Elbow (deg)")
        if np.any(~np.isnan(spine)):
            plt.plot(t, spine, label="Spine lean (deg)")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle")
        plt.legend(loc="best")
        plt.tight_layout()
        out = os.path.join(out_dir, "timechart.png")
        plt.savefig(out)
        plt.close()
        return out, smooth
    except Exception:
        return None, smooth


# -----------------------------
# Reference comparison
# -----------------------------
def reference_compare(medians: Dict[str, float], ref: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for k, v in medians.items():
        rng = ref.get(k)
        if rng and not (v is None or (isinstance(v, float) and math.isnan(v))):
            lo, hi = float(rng[0]), float(rng[1])
            within = 1.0 if (lo <= v <= hi) else 0.0
            delta = 0.0 if within else min(abs(v-lo), abs(v-hi))
            out[k] = {"value": float(v), "target_low": lo, "target_high": hi, "within": within, "delta": float(delta)}
    return out


# -----------------------------
# Grade
# -----------------------------
def grade_prediction(overall_score: float) -> str:
    if overall_score >= 9.0: return "Advanced"
    if overall_score >= 7.5: return "Intermediate"
    return "Beginner"


# -----------------------------
# HTML report
# -----------------------------
def write_report(out_dir: str, context: Dict, chart_path: Optional[str]) -> str:
    html = os.path.join(out_dir, "report.html")
    ev = context.get("evaluation", {})
    ref = context.get("reference", {})
    phases = context.get("phases", [])
    contact = context.get("contact_frame", None)
    avg_fps = context.get("avg_fps", 0.0)
    vid = context.get("annotated_video", "")

    def _row(k, v):
        return f"<tr><td>{k}</td><td>{v.get('score','')}</td><td>{v.get('feedback','')}</td></tr>"

    rows = "\n".join(_row(k, v) for k, v in ev.items())

    ref_rows = ""
    for k, d in ref.items():
        ref_rows += f"<tr><td>{k}</td><td>{d['value']:.2f}</td><td>{d['target_low']}–{d['target_high']}</td><td>{'Yes' if d['within'] else 'No'}</td></tr>"

    phase_rows = ""
    for p in phases:
        phase_rows += f"<tr><td>{p['name']}</td><td>{p['start']}</td><td>{p['end']}</td></tr>"

    with open(html, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><html><head>
<meta charset="utf-8"/><title>Cover Drive Report</title>
<style>body{{font-family:system-ui,Arial,sans-serif;background:#0b0f14;color:#e6f0ff;padding:24px}}
h1,h2{{color:#c2f6ff}} table{{border-collapse:collapse;width:100%;margin:12px 0}} 
td,th{{border:1px solid #1f2a36;padding:6px}} .muted{{opacity:.8}}</style>
</head><body>
<h1>Cover Drive – Analysis Report</h1>
<p class="muted">Avg FPS: {avg_fps:.2f} • Contact frame: {contact if contact is not None else '—'}</p>

<h2>Scores</h2>
<table><tr><th>Category</th><th>Score</th><th>Feedback</th></tr>
{rows}
</table>

<h2>Reference Comparison</h2>
<table><tr><th>Metric</th><th>Measured</th><th>Target</th><th>Within</th></tr>
{ref_rows}
</table>

<h2>Phases</h2>
<table><tr><th>Name</th><th>Start</th><th>End</th></tr>
{phase_rows}
</table>

{"<h2>Timeline</h2><img src='timechart.png' style='max-width:100%'>" if chart_path else ""}

<h2>Annotated Video</h2>
<p class="muted">{vid}</p>

</body></html>""")
    return html
