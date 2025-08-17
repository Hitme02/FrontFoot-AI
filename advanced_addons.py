# advanced_addons.py
"""
Lightweight "bonus" helpers:
- reference loading + comparison
- phase segmentation (heuristic, with explicit Impact + Recovery)
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
# Utilities
# -----------------------------
def _nanmed(a: np.ndarray, default: float = 0.0) -> float:
    m = np.nanmedian(a)
    return default if np.isnan(m) else float(m)

def _safe_array(x: List[float], default: float = 0.0) -> np.ndarray:
    arr = np.array(
        [np.nan if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in x],
        dtype=np.float32
    )
    if np.all(np.isnan(arr)):
        arr = np.full_like(arr, default, dtype=np.float32)
    else:
        arr = np.nan_to_num(arr, nan=_nanmed(arr, default))
    return arr

def _safe_diff(x: List[float]) -> np.ndarray:
    a = _safe_array(x)
    return np.abs(np.diff(a))

def _smooth1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    """Simple moving-average smoothing; keeps length identical (centered)."""
    win = max(1, int(win))
    if win == 1:
        return x.astype(np.float32)
    k = np.ones(win, dtype=np.float32) / float(win)
    pad = win // 2
    xpad = np.pad(x, (pad, pad - (win % 2 == 0)), mode="edge")
    y = np.convolve(xpad, k, mode="valid")
    return y.astype(np.float32)


# -----------------------------
# Contact-moment detection (Impact)
# -----------------------------
def detect_contact(series: Dict[str, List[float]], fps: float) -> Optional[int]:
    """
    Heuristic 'impact' frame:
      - Combine elbow velocity (abs Δ) and inverse head–knee gap.
      - Take the global maximum of a smoothed score.
    """
    n = max((len(v) for v in series.values()), default=0)
    if n < 5:
        return None

    elbow = np.array(series.get("elbow_deg", [np.nan]*n), dtype=np.float32)
    gap   = np.array(series.get("head_knee_gap_norm", [np.nan]*n), dtype=np.float32)

    # elbow velocity (length n-1)
    def _vel(x):
        x = np.array(x, dtype=np.float32)
        if not np.any(~np.isnan(x)):
            return np.zeros(max(n-1, 1), dtype=np.float32)
        x = np.nan_to_num(x, nan=np.nanmedian(x))
        v = np.abs(np.diff(x))
        return v

    v_elbow = _vel(elbow)
    # z-score
    v_elbow = (v_elbow - np.nanmean(v_elbow)) / (np.nanstd(v_elbow) + 1e-6)

    # inverse gap (smaller gap → bigger score) aligned to n-1
    gap_c = gap[:-1] if len(gap) > 1 else gap
    if np.any(~np.isnan(gap_c)):
        inv_gap = -np.nan_to_num(gap_c, nan=np.nanmedian(gap_c))
        inv_gap = (inv_gap - np.nanmean(inv_gap)) / (np.nanstd(inv_gap) + 1e-6)
    else:
        inv_gap = np.zeros_like(v_elbow)

    score = 0.65 * v_elbow + 0.35 * inv_gap
    if len(score) >= 5:
        score = np.convolve(score, np.ones(5)/5.0, mode="same")

    # avoid edges
    lo = max(2, int(0.02 * len(score)))
    hi = min(len(score) - 3, int(0.98 * len(score)))
    idx = lo + int(np.argmax(score[lo:hi])) if hi > lo else int(np.argmax(score))

    # map back to full-frame index
    impact = int(np.clip(idx, 1, n-2))
    return impact


# -----------------------------
# Phase segmentation with explicit Impact/Recovery
# -----------------------------
def _merge_short_segments(segments: List[Dict[str, int]], min_len: int = 5) -> List[Dict[str, int]]:
    """
    Merge segments shorter than min_len into neighbors; collapse duplicates.
    Works on dicts containing {'label','start','end'}. Keeps a 'name' alias.
    """
    if not segments:
        return segments

    # collapse immediate duplicates first
    out: List[Dict[str, int]] = []
    for seg in segments:
        if out and seg["label"] == out[-1]["label"]:
            out[-1]["end"] = seg["end"]
        else:
            out.append(dict(seg))

    i = 0
    while i < len(out):
        seg = out[i]
        length = seg["end"] - seg["start"]
        if length < min_len and len(out) > 1:
            left_len  = out[i-1]["end"] - out[i-1]["start"] if i-1 >= 0 else -1
            right_len = out[i+1]["end"] - out[i+1]["start"] if i+1 < len(out) else -1
            if right_len >= left_len and i+1 < len(out):
                out[i+1]["start"] = min(out[i+1]["start"], seg["start"])
                del out[i]
                continue
            elif i-1 >= 0:
                out[i-1]["end"] = max(out[i-1]["end"], seg["end"])
                del out[i]
                i -= 1
                continue
        i += 1

    # final duplicate collapse
    merged: List[Dict[str, int]] = []
    for seg in out:
        if merged and seg["label"] == merged[-1]["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    for m in merged:
        m["name"] = m["label"]
    return merged

def _enforce_order(
    segments: List[Dict[str, int]],
    allowed=("Stance","Stride","Downswing","Impact","Follow-through","Recovery")
) -> List[Dict[str, int]]:
    """Clamp progression to the canonical order."""
    if not segments:
        return segments
    order = {name:i for i, name in enumerate(allowed)}
    current_max = -1
    cleaned: List[Dict[str, int]] = []
    for seg in segments:
        lbl = seg["label"]
        idx = order.get(lbl, current_max)
        if idx < current_max:
            lbl = allowed[current_max]
        else:
            current_max = idx
        cleaned.append({"label": lbl, "name": lbl, "start": seg["start"], "end": seg["end"]})
    return _merge_short_segments(cleaned, min_len=1)

def segment_phases(
    series: Dict[str, List[float]],
    fps: float,
    contact_frame: Optional[int] = None
) -> List[Dict[str, int]]:
    """
    Produce ordered phases:
      Stance → Stride → Downswing → Impact → Follow-through → Recovery
    based on velocity heuristics and an 'impact' frame.

    Returns list of dicts: {label,name,start,end} in FRAME indices.
    """
    n = max((len(v) for v in series.values()), default=0)
    if n < 5:
        return [{"label":"Stance","name":"Stance", "start":0, "end":max(0,n-1)}]

    # velocity proxy from elbow + spine (length n-1)
    vel = ( _safe_diff(series.get("elbow_deg", [np.nan]*n)) +
            _safe_diff(series.get("spine_lean_deg", [np.nan]*n)) ) / 2.0
    vel = _smooth1d(vel, win=5)
    vmax = float(np.max(vel)) if np.max(vel) > 1e-6 else 1.0
    vel = (vel / vmax)

    thr_hi = float(np.percentile(vel, 65))  # enter Downswing
    thr_lo = float(np.percentile(vel, 45))  # exit Downswing → calm
    thr_lo = min(thr_lo, thr_hi*0.9)

    # impact
    impact = contact_frame if (contact_frame is not None) else detect_contact(series, fps)
    impact = int(np.clip(impact if impact is not None else int(np.argmax(vel)), 1, n-2))

    labels = np.array(["Stride"] * n, dtype=object)

    # Stance (stable pre-move)
    stance_end = int(max(3, min(impact - max(1, int(0.06*n)), int(0.2*n))))
    labels[:max(0, stance_end)] = "Stance"

    # Pre-impact: Downswing vs Stride by velocity
    for i in range(stance_end, impact):
        v = vel[i-1]  # vel is between frames
        labels[i] = "Downswing" if v >= thr_hi else "Stride"

    # Short Impact window
    win = max(1, int(0.015 * n))
    imp_s = max(impact - win//2, stance_end + 1)
    imp_e = min(impact + win//2 + 1, n-1)
    labels[imp_s:imp_e] = "Impact"

    # After impact → Follow-through while motion stays above calm threshold, then Recovery
    calm_len = max(3, int(0.04 * n))
    ft_start = imp_e
    ft_end = ft_start
    for i in range(ft_start, n-1):
        v = vel[i-1]
        ft_end = i+1
        if v < thr_lo:
            # need a short block of calm frames to exit FT
            calm_ok = True
            for j in range(i, min(i+calm_len, n-1)):
                if vel[j-1] >= thr_lo:
                    calm_ok = False
                    break
            if calm_ok:
                ft_end = i+1
                break
    labels[ft_start:ft_end] = "Follow-through"
    if ft_end < n:
        labels[ft_end:n] = "Recovery"

    # compress to segments
    segs: List[Dict[str, int]] = []
    cur = labels[0]; s = 0
    for i in range(1, n):
        if labels[i] != cur:
            segs.append({"label": str(cur), "name": str(cur), "start": int(s), "end": int(i)})
            cur, s = labels[i], i
    segs.append({"label": str(cur), "name": str(cur), "start": int(s), "end": int(n-1)})

    # Guarantee a visible Stride between Stance and first Downswing
    has_stride   = any(x["label"] == "Stride"     for x in segs)
    first_ds_idx = next((i for i,x in enumerate(segs) if x["label"] == "Downswing"), None)
    stance_idx   = 0 if segs and segs[0]["label"] == "Stance" else None
    if (not has_stride) and (first_ds_idx is not None):
        ds_s = segs[first_ds_idx]["start"]
        st_e = segs[stance_idx]["end"] if stance_idx is not None else 0
        stride_min = max(2, int(0.03 * n))
        stride_s = max(st_e, 0)
        stride_e = max(min(ds_s, n-2), stride_s + stride_min)
        if stride_e > ds_s:
            segs[first_ds_idx]["start"] = stride_e
        if stance_idx is not None:
            segs[stance_idx]["end"] = stride_s
        segs.insert(first_ds_idx, {"label":"Stride", "name":"Stride", "start": int(stride_s), "end": int(stride_e)})

    # Merge shorts, enforce order, and ensure Stance at head
    segs = _merge_short_segments(segs, min_len=max(3, int(0.05 * n)))
    segs = _enforce_order(segs)

    if segs:
        stance_min = max(3, int(0.04 * n))
        if segs[0]["label"] != "Stance":
            segs[0]["label"] = segs[0]["name"] = "Stance"
        if segs[0]["start"] != 0:
            segs[0]["start"] = 0
        if len(segs) > 1 and (segs[0]["end"] - segs[0]["start"]) < stance_min:
            new_end = min(segs[0]["start"] + stance_min, segs[1]["end"] - 1)
            if new_end > segs[0]["end"]:
                segs[1]["start"] = new_end
                segs[0]["end"] = new_end

    # final clipping
    for s in segs:
        s["start"] = int(max(0, min(s["start"], n-2)))
        s["end"]   = int(max(s["start"]+1, min(s["end"], n-1)))

    return segs


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
    smooth = {"elbow_mean_abs_delta": float(0.0 if np.isnan(e_diff) else e_diff),
              "spine_mean_abs_delta": float(0.0 if np.isnan(s_diff) else s_diff)}

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
        ref_rows += (
            f"<tr><td>{k}</td><td>{d['value']:.2f}</td>"
            f"<td>{d['target_low']}–{d['target_high']}</td>"
            f"<td>{'Yes' if d['within'] else 'No'}</td></tr>"
        )

    phase_rows = ""
    for p in phases:
        lbl = p.get("label") or p.get("name", "")
        phase_rows += f"<tr><td>{lbl}</td><td>{p['start']}</td><td>{p['end']}</td></tr>"

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
