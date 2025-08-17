import { useRef, useState } from "react";
import "./index.css";
import FXBackground from "./FXBackground";

export default function App() {
  const fileRef = useRef(null);
  const videoRef = useRef(null); // keep for seeking
  const [frontSide, setFrontSide] = useState("right");
  const [videoUrl, setVideoUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [result, setResult] = useState(null);
  const [fxOn, setFxOn] = useState(true);

  const submit = async () => {
    setBusy(true);
    setMsg("Analyzing… bringing the neon.");
    try {
      const fd = new FormData();
      fd.append("front_side", frontSide);
      if (fileRef.current?.files?.[0]) fd.append("video_file", fileRef.current.files[0]);
      if (videoUrl.trim()) fd.append("video_url", videoUrl.trim());

      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "Analyzer failed");

      setResult(data);
      setMsg("Done! Scroll for results ↓");
    } catch (e) {
      console.error(e);
      setMsg(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  // shared pill style
  const pill =
    "bg-white/10 hover:bg-white/20 transition px-4 py-2 rounded-lg text-sm";

  // ---------------- Helpers for phases / seeking ----------------
  // Pick the most reliable FPS the API provided, else 30.
  const fps = Number(
    result?.time_fps ??
    result?.fps_used ??
    result?.source_fps ??
    30
  );

  const fmtTime = (secs) => {
    // mm:ss.mmm
    const mm = Math.floor(secs / 60);
    const ss = Math.floor(secs % 60);
    const ms = Math.round((secs - Math.floor(secs)) * 1000);
    return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
  };

  const seekToFrame = (frame) => {
    if (!videoRef.current || !Number.isFinite(fps)) return;
    const t = Math.max(0, frame / fps - 0.05); // slight lead-in
    try {
      videoRef.current.currentTime = t;
      videoRef.current.play().catch(() => {});
    } catch {}
  };

  const contactBadge = (frame) => (
    <span className="text-xs bg-white/10 rounded px-2 py-0.5 ml-2">
      Contact @ F{frame} ({fmtTime((frame || 0) / fps)})
    </span>
  );

  // Normalize any phase row into {label, start, end}
  const normalizePhase = (p) => {
    if (Array.isArray(p)) {
      // e.g. ["Stride", 3, 6]
      const [label, a, b] = p;
      return {
        label: String(label ?? "Phase"),
        start: Number(a ?? 0),
        end: Number(b ?? a ?? 0),
      };
    }
    // e.g. {label:"Stride", start:3, end:6} or {name:"Stride", ...}
    return {
      label: String(p?.label ?? p?.name ?? "Phase"),
      start: Number(p?.start ?? 0),
      end: Number(p?.end ?? p?.start ?? 0),
    };
  };
  // --------------------------------------------------------------

  return (
    <main className="min-h-dvh text-white font-sans">
      {/* FX layers */}
      <FXBackground enabled={fxOn} paused={false} />
      <div className="hero-shade"></div>
      <div className="grid-overlay"></div>
      <div className="bg-neon"></div>

      {/* Content */}
      <section className="layer-content max-w-6xl mx-auto px-6 py-14">
        <div className="flex items-center justify-between gap-4">
          {/* CRISP title */}
          <h1 className="font-display title-crisp tracking-tight text-5xl md:text-7xl font-extrabold leading-tight">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-green to-neon-pink drop-shadow-[0_1px_0_rgba(0,0,0,0.6)]"></span>{" "}
            Cover Drive Analyzer
          </h1>

          {/* FX toggle */}
          <button
            onClick={() => setFxOn((v) => !v)}
            className={`${pill} hidden md:inline-block`}
            title="Toggle background effects (use if video FPS dips)"
          >
            FX: {fxOn ? "ON" : "OFF"}
          </button>
        </div>

        <p className="mt-3 text-slate-300 max-w-2xl">
          Upload a file or paste a YouTube Short. We’ll annotate every frame in style and score your shot
          with coach-like feedback.
        </p>

        <div className="grid md:grid-cols-2 gap-6 mt-10">
          {/* Input Card */}
          <div className="glass rounded-3xl p-6">
            <label className="block text-sm text-slate-400 mb-2">Front side</label>
            <select
              value={frontSide}
              onChange={(e) => setFrontSide(e.target.value)}
              className="w-full bg-black/40 rounded-xl p-3 border border-white/10"
            >
              <option value="right">Right (default)</option>
              <option value="left">Left</option>
            </select>

            <div className="mt-4">
              <label className="block text-sm text-slate-400 mb-2">Upload video (mp4/mov/avi/webm)</label>
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                className="w-full bg-black/40 rounded-xl p-3 border border-white/10"
              />
            </div>

            <div className="mt-4">
              <label className="block text-sm text-slate-400 mb-2">or paste YouTube URL</label>
              <input
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="https://youtube.com/shorts/..."
                className="w-full bg-black/40 rounded-xl p-3 border border-white/10"
              />
            </div>

            <button
              onClick={submit}
              disabled={busy}
              className="mt-6 w-full py-3 rounded-xl font-semibold bg-gradient-to-r from-neon-green to-neon-pink hover:opacity-90 transition disabled:opacity-50"
            >
              {busy ? "Processing…" : "Analyze"}
            </button>
            {!!msg && <p className="text-xs text-slate-400 mt-2">{msg}</p>}
          </div>

          {/* What you’ll see */}
          <div className="glass rounded-3xl p-6">
            <h3 className="text-lg font-semibold mb-2">What you’ll see</h3>
            <ul className="text-slate-300 text-sm space-y-2">
              <li>• Neon HUD on video (elbow, spine, foot, head–knee)</li>
              <li>• ✅/❌ cues with tuned thresholds</li>
              <li>• Final JSON scores + downloads</li>
              <li>• If available: phases, contact frame, chart &amp; HTML report</li>
            </ul>
          </div>
        </div>

        {/* Results */}
        {result && (
          <section className="grid md:grid-cols-2 gap-6 mt-10">
            {/* Left column */}
            <div className="space-y-6">
              <div className="glass rounded-3xl p-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold mb-2">Annotated video</h2>
                  {result.grade && (
                    <span className="text-xs bg-white/10 rounded px-2 py-1">
                      Grade: {result.grade}
                    </span>
                  )}
                </div>

                {/* Video */}
                {result.video_url ? (
                  <video
                    ref={videoRef}
                    key={result.video_url}
                    className="w-full rounded-xl border border-white/10 bg-black/40 aspect-video"
                    controls
                    playsInline
                    preload="metadata"
                    onPlay={() => setFxOn(false)}
                    onPause={() => setFxOn(true)}
                    onEnded={() => setFxOn(true)}
                    onLoadedData={() => setMsg("")}
                    onError={() =>
                      setMsg("Video couldn't inline-play here. Use 'Open in new tab' or 'Download Video'.")
                    }
                  >
                    <source
                      src={`${result.video_url}?t=${Date.now()}`}
                      type={result.video_mime || "video/mp4"}
                    />
                  </video>
                ) : (
                  <div className="w-full rounded-xl border border-white/10 bg-black/40 aspect-video grid place-items-center">
                    <span className="text-slate-400 text-sm">Waiting for video…</span>
                  </div>
                )}

                {/* Actions */}
                <div className="flex flex-wrap gap-3 mt-3">
                  <a href={result.video_url} download className={pill}>
                    Download Video
                  </a>
                  <a href={result.eval_url} download className={pill}>
                    evaluation.json
                  </a>
                  <a href={result.video_url} target="_blank" rel="noreferrer" className={pill}>
                    Open in new tab
                  </a>
                  {result.report_url && (
                    <a href={result.report_url} className={pill} target="_blank" rel="noreferrer">
                      report.html
                    </a>
                  )}
                  {result.chart_url && (
                    <a href={result.chart_url} className={pill} download>
                      timechart.png
                    </a>
                  )}
                </div>
              </div>

              {/* Phases */}
              <div className="glass rounded-3xl p-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-semibold">Phases (frames & times)</h2>
                  {Number.isFinite(result?.contact_frame) && contactBadge(result.contact_frame)}
                </div>

                {!result?.phases?.length ? (
                  <p className="text-slate-400 text-sm mt-2">No phases available for this run.</p>
                ) : (
                  <div className="mt-3 max-h-64 overflow-auto rounded-lg border border-white/10">
                    <table className="w-full text-sm">
                      <thead className="bg-white/5">
                        <tr className="text-left">
                          <th className="px-3 py-2 font-semibold">Phase</th>
                          <th className="px-3 py-2 font-semibold">Frames</th>
                          <th className="px-3 py-2 font-semibold">Start</th>
                          <th className="px-3 py-2 font-semibold">End</th>
                          <th className="px-3 py-2 font-semibold">Go</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(result.phases || []).map((raw, i) => {
                          const p = normalizePhase(raw);
                          const startT = p.start / fps;
                          const endT = p.end / fps;
                          return (
                            <tr key={`${p.label}-${p.start}-${p.end}-${i}`} className="border-t border-white/10 hover:bg-white/5">
                              <td className="px-3 py-2">{p.label}</td>
                              <td className="px-3 py-2">[{p.start}–{p.end}]</td>
                              <td className="px-3 py-2 font-mono">{fmtTime(startT)}</td>
                              <td className="px-3 py-2 font-mono">{fmtTime(endT)}</td>
                              <td className="px-3 py-2">
                                <button
                                  className="text-xs bg-white/10 hover:bg-white/20 px-2 py-1 rounded"
                                  onClick={() => seekToFrame(p.start)}
                                >
                                  Seek
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
                <p className="text-xs text-slate-500 mt-2">
                  FPS used for timestamps: {fps.toFixed(2)}
                </p>
              </div>
            </div>

            {/* Right column: scores */}
            <div className="glass rounded-3xl p-4">
              <h2 className="text-xl font-semibold mb-3">Scores & feedback</h2>
              <div className="space-y-3">
                {Object.entries(result.evaluation || {}).map(([k, v]) => (
                  <div key={k} className="bg-black/25 rounded-xl p-3 border border-white/10">
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">{k}</div>
                      <div className="text-neon-green font-mono">{v.score}/10</div>
                    </div>
                    <div className="text-slate-300 text-sm mt-1 leading-relaxed">
                      {v.feedback}
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-3">
                Avg FPS: {Number(result.avg_fps).toFixed(2)}
              </p>
            </div>
          </section>
        )}
      </section>
    </main>
  );
}
