#!/usr/bin/env python3
# AthleteRise – Real-Time Cover Drive Analysis (MediaPipe)
# Full pipeline with robust video opening, FFmpeg normalization, and fallback pipe decoding.

import argparse
import os
import json
import math
import time
import tempfile
import shutil
import subprocess
import sys
from typing import Dict, Tuple, Optional, Generator

import numpy as np
import cv2

# --------------------- Lazy import mediapipe ---------------------
try:
    import mediapipe as mp
except ImportError:
    mp = None

# ================================================================
#                        UTILITIES / IO
# ================================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def download_video(url: str, out_mp4: str) -> str:
    """
    Downloads a video using yt-dlp. Requires yt-dlp installed in the venv.
    """
    tmpdir = tempfile.mkdtemp(prefix="ar_dl_")
    try:
        cmd = [
            sys.executable, "-m", "yt_dlp",
            # Prefer MP4 first, else best that is easy to re-encode
            "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "-o", os.path.join(tmpdir, "vid.%(ext)s"),
            url
        ]
        subprocess.check_call(cmd)
        files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)]
        if not files:
            raise RuntimeError("Download failed, no file produced.")
        src = sorted(files, key=os.path.getsize)[-1]
        shutil.copy(src, out_mp4)
        return out_mp4
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def check_ffmpeg_available() -> bool:
    try:
        subprocess.check_output(["ffmpeg", "-version"])
        subprocess.check_output(["ffprobe", "-version"])
        return True
    except Exception:
        return False

def normalize_mp4(in_path: str, out_path: str, target_fps=30, target_height=720):
    """
    Normalize to safe H.264 MP4 (yuv420p), CFR, resized to 720p height (keep AR), faststart.
    """
    vf = f"fps={target_fps},scale=-2:{target_height}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        out_path
    ]
    print("[INFO] Normalizing to safe MP4…")
    subprocess.check_call(cmd)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError("MP4 normalization produced empty file")

def normalize_avi_mjpeg(in_path: str, out_path: str, target_fps=30, target_height=720):
    """
    Normalize to MJPEG AVI (very compatible with OpenCV on Windows). Drops audio.
    """
    vf = f"fps={target_fps},scale=-2:{target_height}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "mjpeg", "-q:v", "3", "-an",
        out_path
    ]
    print("[INFO] Normalizing to MJPEG AVI…")
    subprocess.check_call(cmd)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError("AVI normalization produced empty file")

def ffmpeg_frame_pipe(in_path: str, target_fps=30, target_height=720) -> Tuple[Generator[np.ndarray, None, None], Tuple[int, int], float]:
    """
    Last-resort decode: stream frames via FFmpeg -> rawvideo pipe (RGB24).
    Returns: (frame generator yielding BGR frames), (width, height), fps
    """
    # Probe original to compute aspect for scaled width
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=nokey=1:noprint_wrappers=1", in_path
    ], text=True).strip().splitlines()
    w0, h0 = int(probe[0]), int(probe[1])
    aspect = w0 / max(h0, 1)
    w_est = int(round(target_height * aspect))
    if w_est % 2:
        w_est += 1  # even width for safety
    w, h = w_est, target_height
    fps = float(target_fps)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", f"fps={target_fps},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ]
    print("[INFO] Decoding via FFmpeg rawvideo pipe…")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    frame_size = w * h * 3  # rgb24

    def gen():
        while True:
            data = proc.stdout.read(frame_size)
            if not data or len(data) < frame_size:
                break
            frame_rgb = np.frombuffer(data, np.uint8).reshape((h, w, 3))
            # convert to BGR for OpenCV drawing functions
            yield cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        proc.stdout.close()
        proc.wait()

    return gen(), (w, h), fps

# ================================================================
#                       GEOMETRY / METRICS
# ================================================================
def np_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b defined by a-b-c in degrees."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float("nan")
    cosv = np.dot(ba, bc) / (nba * nbc)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def vec_angle_with_vertical(v: np.ndarray) -> float:
    """Angle between vector v and vertical (y+ down image)."""
    vert = np.array([0.0, 1.0], dtype=np.float32)
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return float("nan")
    cosv = np.dot(v / nv, vert)
    cosv = np.clip(cosv, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def line_angle_vs_xaxis(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """Angle of vector (from->to) w.r.t +x axis in [0,180] degrees."""
    d = p_to - p_from
    if np.linalg.norm(d) < 1e-6:
        return float("nan")
    ang = math.degrees(math.atan2(d[1], d[0]))
    ang = abs(ang) % 180.0
    return float(ang)

# --------------------- MediaPipe landmark indices ---------------------
POSE_LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

def lm_xy(landmarks, idx: int, w: int, h: int) -> Optional[np.ndarray]:
    lm = landmarks[idx]
    if lm.x is None or lm.y is None:
        return None
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def center(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0

def compute_metrics(lms, w: int, h: int, front_side: str) -> Dict[str, float]:
    """
    Returns dict: elbow_deg, spine_lean_deg, head_knee_gap_norm, foot_angle_deg
    """
    if front_side.lower() == "right":
        sh, el, wr = "right_shoulder", "right_elbow", "right_wrist"
        knee, heel, toe = "right_knee", "right_heel", "right_foot_index"
    else:
        sh, el, wr = "left_shoulder", "left_elbow", "left_wrist"
        knee, heel, toe = "left_knee", "left_heel", "left_foot_index"

    def P(name):
        return lm_xy(lms, POSE_LM[name], w, h) if lms else None

    p_sh = P(sh)
    p_el = P(el)
    p_wr = P(wr)
    p_lsh = P("left_shoulder")
    p_rsh = P("right_shoulder")
    p_lhip = P("left_hip")
    p_rhip = P("right_hip")
    p_nose = P("nose")
    p_knee = P(knee)
    p_heel = P(heel)
    p_toe = P(toe)

    elbow_deg = np_angle(p_sh, p_el, p_wr) if (p_sh is not None and p_el is not None and p_wr is not None) else float("nan")

    spine_lean_deg = float("nan")
    if all(x is not None for x in [p_lsh, p_rsh, p_lhip, p_rhip]):
        sh_c = center(p_lsh, p_rsh)
        hip_c = center(p_lhip, p_rhip)
        v = sh_c - hip_c
        spine_lean_deg = 180.0 - vec_angle_with_vertical(v)

    head_knee_gap_norm = float("nan")
    if p_nose is not None and p_knee is not None and p_lsh is not None and p_rsh is not None:
        shoulder_w = np.linalg.norm(p_lsh - p_rsh)
        if shoulder_w > 1e-6:
            gap = abs(p_nose[0] - p_knee[0])  # horizontal distance
            head_knee_gap_norm = float(gap / shoulder_w)

    foot_angle_deg = line_angle_vs_xaxis(p_toe, p_heel) if (p_toe is not None and p_heel is not None) else float("nan")

    return {
        "elbow_deg": elbow_deg,
        "spine_lean_deg": spine_lean_deg,
        "head_knee_gap_norm": head_knee_gap_norm,
        "foot_angle_deg": foot_angle_deg,
    }

# --------------------- Thresholds / feedback ---------------------
THRESH = {
    "elbow_good_min": 80.0,    # more forgiving & coach-friendly
    "elbow_good_max": 140.0,
    "spine_lean_good_min": 10.0,
    "spine_lean_good_max": 30.0,
    "head_knee_good_max": 0.35,
    "foot_angle_good_max": 25.0,
}

def frame_feedback(m: Dict[str, float]) -> Tuple[str, Tuple[int, int, int]]:
    msgs = []

    # Per-metric messages
    if not math.isnan(m["elbow_deg"]) and (THRESH["elbow_good_min"] <= m["elbow_deg"] <= THRESH["elbow_good_max"]):
        msgs.append("✅ Good elbow elevation")
    else:
        msgs.append("❌ Raise front elbow")

    if not math.isnan(m["spine_lean_deg"]) and (THRESH["spine_lean_good_min"] <= m["spine_lean_deg"] <= THRESH["spine_lean_good_max"]):
        msgs.append("✅ Balanced spine lean")
    else:
        msgs.append("❌ Lean slightly forward")

    if not math.isnan(m["head_knee_gap_norm"]) and (m["head_knee_gap_norm"] <= THRESH["head_knee_good_max"]):
        msgs.append("✅ Head over front knee")
    else:
        msgs.append("❌ Head not over front knee")

    if not math.isnan(m["foot_angle_deg"]) and (m["foot_angle_deg"] <= THRESH["foot_angle_good_max"] or (180 - m["foot_angle_deg"]) <= THRESH["foot_angle_good_max"]):
        msgs.append("✅ Front foot aligned")
    else:
        msgs.append("❌ Open/closed front foot")

    # Cue color is green only if ALL metrics are good
    all_ok = (
        (not math.isnan(m["elbow_deg"]) and THRESH["elbow_good_min"] <= m["elbow_deg"] <= THRESH["elbow_good_max"]) and
        (not math.isnan(m["spine_lean_deg"]) and THRESH["spine_lean_good_min"] <= m["spine_lean_deg"] <= THRESH["spine_lean_good_max"]) and
        (not math.isnan(m["head_knee_gap_norm"]) and m["head_knee_gap_norm"] <= THRESH["head_knee_good_max"]) and
        (not math.isnan(m["foot_angle_deg"]) and (m["foot_angle_deg"] <= THRESH["foot_angle_good_max"] or (180 - m["foot_angle_deg"]) <= THRESH["foot_angle_good_max"]))
    )
    color = (0, 200, 0) if all_ok else (0, 0, 255)

    return " | ".join(msgs), color

def score_from_series(series: Dict[str, list]) -> Dict[str, Dict[str, object]]:
    def median_clean(vals):
        good = [v for v in vals if not (v is None or math.isnan(v))]
        return float(np.median(good)) if good else float("nan")

    med = {k: median_clean(vs) for k, vs in series.items()}

    footwork = 10 if (not math.isnan(med["foot_angle_deg"]) and (med["foot_angle_deg"] <= THRESH["foot_angle_good_max"] or (180 - med["foot_angle_deg"]) <= THRESH["foot_angle_good_max"])) else 6
    headpos  = 10 if (not math.isnan(med["head_knee_gap_norm"]) and med["head_knee_gap_norm"] <= THRESH["head_knee_good_max"]) else 6
    balance  = 10 if (not math.isnan(med["spine_lean_deg"]) and THRESH["spine_lean_good_min"] <= med["spine_lean_deg"] <= THRESH["spine_lean_good_max"]) else 6
    swing    = 8   # placeholder – could use elbow velocity smoothness
    follow   = 7   # placeholder – post-impact posture window

    feedback = {
        "Footwork": {
            "score": footwork,
            "feedback": "Front foot points down the line; square slightly if angle > 20°."
        },
        "Head Position": {
            "score": headpos,
            "feedback": "Keep head stacked over front knee through contact."
        },
        "Swing Control": {
            "score": swing,
            "feedback": "Smooth V-shaped arc; avoid abrupt elbow snaps."
        },
        "Balance": {
            "score": balance,
            "feedback": "Maintain 10–30° forward spine lean; avoid falling off-side."
        },
        "Follow-through": {
            "score": follow,
            "feedback": "Finish high with control; stable base after impact."
        }
    }
    return feedback

# --------------------- Drawing helpers ---------------------
def draw_pose_skeleton(frame, lms, w, h):
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_drawing.draw_landmarks(
        frame,
        lms,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

# --- Neon / Glass HUD helpers ---------------------------------------------
def draw_neon_text(img, text, org, color=(120, 255, 190), scale=0.9, thickness=2):
    x, y = org
    glow_layer = np.zeros_like(img)
    cv2.putText(glow_layer, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness * 2, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow_layer, (0, 0), 7, 7)
    cv2.addWeighted(glow, 0.55, img, 1.0, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def draw_glass_hud(img, lines, topleft=(16, 16), pad=12, width=520):
    """
    lines: list of (text, (b,g,r))
    """
    x, y = topleft
    line_h = 28
    h = pad * 2 + line_h * len(lines)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + h), (24, 24, 32), -1)  # glass bg
    cv2.rectangle(overlay, (x, y), (x + width, y + h), (64, 64, 80), 1)   # subtle border
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    ty = y + pad + 20
    for text, color in lines:
        draw_neon_text(img, text, (x + pad, ty), color=color, scale=0.8, thickness=2)
        ty += line_h

# ================================================================
#                         CORE ANALYSIS
# ================================================================
def analyze(video_path: str, output_dir: str, front_side: str = "right") -> Dict[str, object]:
    """
    Robust open (original -> normalized MP4 -> normalized AVI -> raw pipe),
    run MediaPipe Pose, overlay metrics + cues, save annotated video + evaluation.json.
    """
    if mp is None:
        raise ImportError("mediapipe not installed. pip install mediapipe")
    ensure_dir(output_dir)

    # --- Try multiple open strategies in order ---
    used_path = video_path
    backend_used = "opencv:original"
    cap = cv2.VideoCapture(used_path, cv2.CAP_FFMPEG)

    def cap_is_ok(c: cv2.VideoCapture) -> bool:
        if not c.isOpened():
            return False
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if w <= 0 or h <= 0:
            return False
        if n > 0:
            return True
        # Some containers report 0 frames; do a safe probe read and restore position
        pos = c.get(cv2.CAP_PROP_POS_FRAMES)
        ret, _ = c.read()
        if ret:
            c.set(cv2.CAP_PROP_POS_FRAMES, pos)
        return ret

    if not cap_is_ok(cap):
        if not check_ffmpeg_available():
            raise RuntimeError("FFmpeg not found on PATH. Install via winget: `winget install Gyan.FFmpeg` then reopen terminal.")
        # 1) Normalize to MP4
        norm_mp4 = os.path.join(output_dir, "input_normalized.mp4")
        try:
            normalize_mp4(used_path, norm_mp4)
            cap.release()
            cap = cv2.VideoCapture(norm_mp4, cv2.CAP_FFMPEG)
            used_path = norm_mp4
            backend_used = "opencv:mp4"
        except Exception as e:
            print(f"[WARN] MP4 normalization failed or still unreadable: {e}")

    if not cap_is_ok(cap):
        # 2) Normalize to AVI MJPEG
        norm_avi = os.path.join(output_dir, "input_mjpeg.avi")
        try:
            normalize_avi_mjpeg(video_path, norm_avi)
            cap.release()
            cap = cv2.VideoCapture(norm_avi)  # MJPEG often fine with default backend
            used_path = norm_avi
            backend_used = "opencv:avi"
        except Exception as e:
            print(f"[WARN] AVI normalization failed or still unreadable: {e}")

    use_pipe = False
    if not cap_is_ok(cap):
        # 3) Last resort: FFmpeg rawvideo pipe
        use_pipe = True
        backend_used = "ffmpeg:pipe"
        print("[INFO] Falling back to FFmpeg frame pipe (rawvideo).")

    print(f"[INFO] Video backend: {backend_used}")

    # -------------------- Path A: OpenCV capture --------------------
    if not use_pipe:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # optional

        # Writer (MP4 then AVI fallback)
        out_path = os.path.join(output_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not out.isOpened():
            print("[WARN] MP4 writer failed, falling back to AVI (XVID).")
            out_path = os.path.join(output_dir, "annotated_video.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            if not out.isOpened():
                cap.release()
                raise RuntimeError("Could not open any video writer (MP4/AVI).")

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        series = {
            "elbow_deg": [],
            "spine_lean_deg": [],
            "head_knee_gap_norm": [],
            "foot_angle_deg": [],
        }

        processed = 0
        t0 = time.time()

        with pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    draw_pose_skeleton(frame, results.pose_landmarks, w, h)

                m = {
                    "elbow_deg": float("nan"),
                    "spine_lean_deg": float("nan"),
                    "head_knee_gap_norm": float("nan"),
                    "foot_angle_deg": float("nan")
                }
                try:
                    if results.pose_landmarks:
                        lms = results.pose_landmarks.landmark
                        m = compute_metrics(lms, w, h, front_side)
                except Exception:
                    pass

                for k in series.keys():
                    series[k].append(m[k])

                hud_lines = [
                    (f"Elbow: {m['elbow_deg']:.1f}°" if not math.isnan(m['elbow_deg']) else "Elbow: --", (160, 255, 200)),
                    (f"Spine lean: {m['spine_lean_deg']:.1f}°" if not math.isnan(m['spine_lean_deg']) else "Spine: --", (160, 255, 200)),
                    (f"Head–knee gap: {m['head_knee_gap_norm']:.2f}" if not math.isnan(m['head_knee_gap_norm']) else "Head–knee: --", (160, 255, 200)),
                    (f"Foot angle: {m['foot_angle_deg']:.1f}°" if not math.isnan(m['foot_angle_deg']) else "Foot: --", (160, 255, 200)),
                ]
                draw_glass_hud(frame, hud_lines, topleft=(16, 16), width=520)

                cue, cue_color = frame_feedback(m)
                draw_glass_hud(frame, [(cue, cue_color)],
                               topleft=(16, frame.shape[0] - 16 - 28 - 12),
                               width=700)

                out.write(frame)
                processed += 1

        cap.release()
        out.release()

        elapsed = max(time.time() - t0, 1e-6)
        avg_fps = processed / elapsed if processed else 0.0
        print(f"[INFO] Processed {processed} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS)")

        evaluation = score_from_series(series)
        eval_path = os.path.join(output_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        return {"annotated_video": out_path, "evaluation": eval_path, "avg_fps": avg_fps}

    # -------------------- Path B: FFmpeg pipe --------------------
    else:
        frames, (w, h), fps = ffmpeg_frame_pipe(used_path, target_fps=30, target_height=720)

        out_path = os.path.join(output_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if not out.isOpened():
            print("[WARN] MP4 writer failed in pipe path, falling back to AVI.")
            out_path = os.path.join(output_dir, "annotated_video.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
            if not out.isOpened():
                raise RuntimeError("Could not open any video writer in pipe path.")

        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        series = {
            "elbow_deg": [],
            "spine_lean_deg": [],
            "head_knee_gap_norm": [],
            "foot_angle_deg": []
        }
        processed = 0
        t0 = time.time()

        with pose:
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    draw_pose_skeleton(frame, results.pose_landmarks, w, h)

                m = {
                    "elbow_deg": float("nan"),
                    "spine_lean_deg": float("nan"),
                    "head_knee_gap_norm": float("nan"),
                    "foot_angle_deg": float("nan")
                }
                try:
                    if results.pose_landmarks:
                        lms = results.pose_landmarks.landmark
                        m = compute_metrics(lms, w, h, front_side)
                except Exception:
                    pass

                for k in series.keys():
                    series[k].append(m[k])

                hud_lines = [
                    (f"Elbow: {m['elbow_deg']:.1f}°" if not math.isnan(m['elbow_deg']) else "Elbow: --", (160, 255, 200)),
                    (f"Spine lean: {m['spine_lean_deg']:.1f}°" if not math.isnan(m['spine_lean_deg']) else "Spine: --", (160, 255, 200)),
                    (f"Head–knee gap: {m['head_knee_gap_norm']:.2f}" if not math.isnan(m['head_knee_gap_norm']) else "Head–knee: --", (160, 255, 200)),
                    (f"Foot angle: {m['foot_angle_deg']:.1f}°" if not math.isnan(m['foot_angle_deg']) else "Foot: --", (160, 255, 200)),
                ]
                draw_glass_hud(frame, hud_lines, topleft=(16, 16), width=520)

                cue, cue_color = frame_feedback(m)
                draw_glass_hud(frame, [(cue, cue_color)],
                               topleft=(16, frame.shape[0] - 16 - 28 - 12),
                               width=700)

                out.write(frame)
                processed += 1

            out.release()

        elapsed = max(time.time() - t0, 1e-6)
        avg_fps = processed / elapsed if processed else 0.0
        print(f"[INFO] Processed {processed} frames in {elapsed:.2f}s (avg {avg_fps:.2f} FPS)")

        evaluation = score_from_series(series)
        eval_path = os.path.join(output_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(evaluation, f, indent=2)

        return {"annotated_video": out_path, "evaluation": eval_path, "avg_fps": avg_fps}

# ================================================================
#                           CLI WRAPPER
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="AthleteRise – Real-Time Cover Drive Analysis (MediaPipe)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-url", type=str, help="YouTube URL (e.g., Shorts link)")
    src.add_argument("--video-path", type=str, help="Local video file path")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--front-side", type=str, default="right", choices=["right", "left"], help="Front side (elbow/leg to evaluate)")
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.video_url:
        local_video = os.path.join(args.output_dir, "input.mp4")
        print("[INFO] Downloading video...")
        local_video = download_video(args.video_url, local_video)
        print("[INFO] Downloaded to:", local_video)
    else:
        local_video = args.video_path
        if not os.path.exists(local_video):
            raise FileNotFoundError(local_video)

    results = analyze(local_video, args.output_dir, front_side=args.front_side)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
