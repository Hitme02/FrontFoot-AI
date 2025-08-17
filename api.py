# api.py
import os, time, json, subprocess
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from cover_drive_analysis_realtime import analyze, ensure_dir, download_video

APP_ROOT    = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR  = os.path.join(APP_ROOT, "frontend", "public")
OUTPUT_ROOT = os.path.join(PUBLIC_DIR, "output")
ensure_dir(OUTPUT_ROOT)

ALLOWED = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MIME_MAP = {
    ".mp4": "video/mp4",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}

app = Flask(__name__)

@app.post("/api/analyze")
def api_analyze():
    front_side = request.form.get("front_side", "right")
    url        = (request.form.get("video_url") or "").strip()
    file       = request.files.get("video_file")

    run_id  = f"run_{int(time.time())}"
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    ensure_dir(run_dir)

    # Input handling
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

    # Run analyzer
    res = analyze(local_video, run_dir, front_side=front_side)

    # Read evaluation
    try:
        with open(res["evaluation"], "r", encoding="utf-8") as f:
            evaluation = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Failed to read evaluation.json: {e}"}), 500

    # Force a browser-safe MP4 (H.264 + yuv420p + faststart)
    src_path = res["annotated_video"]
    mp4_path = os.path.join(run_dir, "annotated_browser.mp4")
    final_video = src_path
    try:
        subprocess.check_call([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src_path,
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

    return jsonify({
        "ok": True,
        "runId": run_id,
        "video_url": f"/output/{run_id}/{os.path.basename(final_video)}",
        "video_mime": video_mime,
        "eval_url":  f"/output/{run_id}/evaluation.json",
        "evaluation": evaluation,
        "avg_fps": res.get("avg_fps", 0.0),
    })

@app.get("/api/ping")
def ping():
    return jsonify({"pong": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7861, debug=True)
