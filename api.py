# api.py
import os, time, json, subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from cover_drive_analysis_realtime import analyze, ensure_dir, download_video

APP_ROOT    = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR  = os.path.join(APP_ROOT, "frontend", "public")   # Vite serves this
OUTPUT_ROOT = os.path.join(PUBLIC_DIR, "output")
ensure_dir(OUTPUT_ROOT)

ALLOWED = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MIME_MAP = {
    ".mp4":  "video/mp4",
    ".avi":  "video/x-msvideo",
    ".webm": "video/webm",
    ".mov":  "video/quicktime",
    ".mkv":  "video/x-matroska",
}

app = Flask(__name__)

def _bool(v: str) -> bool:
    return str(v).lower().strip() in {"1", "true", "yes", "on"}

@app.post("/api/analyze")
def api_analyze():
    # Inputs
    front_side = request.form.get("front_side", "right")
    fast_flag  = _bool(request.form.get("fast", "false"))
    url        = (request.form.get("video_url") or "").strip()
    file       = request.files.get("video_file")

    # Per-run output dir served by Vite
    run_id  = f"run_{int(time.time())}"
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    ensure_dir(run_dir)

    # ---------- Input handling ----------
    if url:
        local_video = os.path.join(run_dir, "input.mp4")
        try:
            download_video(url, local_video)
        except Exception as e:
            return jsonify({"error": f"yt-dlp download failed: {e}"}), 400
    elif file and os.path.splitext(file.filename)[1].lower() in ALLOWED:
        local_video = os.path.join(run_dir, secure_filename(file.filename))
        file.save(local_video)
    else:
        return jsonify({"error": "Provide a video_file or a video_url"}), 400

    # ---------- Run analyzer ----------
    # Prefer the advanced signature (fast=...), but fall back if your analyze() doesn’t accept it yet.
    try:
        res = analyze(local_video, run_dir, front_side=front_side, fast=fast_flag)
    except TypeError:
        res = analyze(local_video, run_dir, front_side=front_side)

    # ---------- Read evaluation ----------
    try:
        with open(res["evaluation"], "r", encoding="utf-8") as f:
            evaluation = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read evaluation.json: {e}"}), 500

    # ---------- Pick a browser-safe MP4 ----------
    # Prefer analyzer-provided browser copy; else enforce H.264/yuv420p here.
    final_video = res.get("browser_video") or res.get("annotated_video")
    if final_video is None:
        return jsonify({"error": "Analyzer did not return an output video path"}), 500

    if not final_video.lower().endswith(".mp4") or "browser" not in os.path.basename(final_video):
        mp4_path = os.path.join(run_dir, "annotated_browser.mp4")
        try:
            subprocess.check_call([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", final_video,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                mp4_path
            ])
            final_video = mp4_path
        except Exception as e:
            print("[WARN] Final MP4 convert failed; returning original:", e)

    ext = os.path.splitext(final_video)[1].lower()
    video_mime = MIME_MAP.get(ext, "video/mp4")

    # ---------- Optional extras from analyzer ----------
    # These keys exist if you’re using the advanced_addons-powered analyze().
    phases        = res.get("phases")
    contact_frame = res.get("contact_frame")
    grade         = res.get("grade")

    chart_name  = res.get("chart_path")
    report_name = res.get("report_html")

    payload = {
        "ok": True,
        "runId": run_id,
        "video_url": f"/output/{run_id}/{os.path.basename(final_video)}",
        "video_mime": video_mime,
        "eval_url":  f"/output/{run_id}/{os.path.basename(res['evaluation'])}",
        "evaluation": evaluation,
        "avg_fps": res.get("avg_fps", 0.0),

        # Extras (may be None if you’re on the basic analyzer)
        "phases": phases,
        "contact_frame": contact_frame,
        "grade": grade,
        "chart_url":  f"/output/{run_id}/{chart_name}"  if chart_name  else None,
        "report_url": f"/output/{run_id}/{report_name}" if report_name else None,
    }
    return jsonify(payload)

@app.get("/api/ping")
def ping():
    return jsonify({"pong": True})

if __name__ == "__main__":
    # Flask on :7861
    app.run(host="127.0.0.1", port=7861, debug=True)
