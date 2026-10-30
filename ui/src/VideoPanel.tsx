import { useEffect, useRef, useState } from "react";
import { h264Supported, playH264 } from "./h264";

interface Props {
  /** MJPEG path on the server, e.g. "video/follow.mjpg". */
  src: string;
  /** H.264 path of the same view, e.g. "video/h264/follow/0", played instead where the browser can. */
  h264Src?: string;
  title: string;
  /** Changing this reloads the stream; use it to force a refresh after a switch. */
  streamKey: string | number;
  /** Shown when the stream errors (no bridge, kinematic backend, bridge down). */
  placeholder: string;
  className?: string;
  /** Camera control: wheel steps (> 0 zooms out), drag in pixels, double-click resets. */
  onZoom?: (steps: number) => void;
  onDrag?: (dx: number, dy: number) => void;
  onReset?: () => void;
  /** Short state of the camera control, shown under the title, e.g. "42 m, 45 deg". */
  hint?: string;
}

/** False while the page is in a background tab or minimized: the video panels then stop their streams. */
function usePageVisible() {
  const [visible, setVisible] = useState(() => document.visibilityState !== "hidden");
  useEffect(() => {
    const on = () => setVisible(document.visibilityState !== "hidden");
    document.addEventListener("visibilitychange", on);
    return () => document.removeEventListener("visibilitychange", on);
  }, []);
  return visible;
}

/**
 * The H.264 stream on a canvas where the browser decodes it (h264.ts), else an MJPEG <img>
 * that swaps to a clean placeholder when the stream is unavailable.
 * Browsers give no load event for MJPEG, so availability comes from the `onerror`
 * of a probe request the server answers with 404 when there is no bridge.
 * Both stop while the page is hidden.
 */
export default function VideoPanel({ src, h264Src, title, streamKey, placeholder, className, onZoom, onDrag, onReset, hint }: Props) {
  const [ok, setOk] = useState(true);
  const [h264, setH264] = useState(h264Supported && !!h264Src);
  const [retry, setRetry] = useState(0);
  const visible = usePageVisible();
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => setH264(h264Supported && !!h264Src), [h264Src, streamKey]);

  // A broken H.264 stream is tried again after 2 s; one that cannot play here falls back to MJPEG.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!h264 || !visible || !h264Src || !canvas) return;
    let timer = 0;
    const stop = playH264(h264Src, canvas, (fatal) => {
      if (fatal) setH264(false);
      else timer = window.setTimeout(() => setRetry((k) => k + 1), 2000);
    });
    return () => {
      stop();
      clearTimeout(timer);
    };
  }, [h264, visible, h264Src, streamKey, retry]);

  // A 404 from the server means there is no stream; anything else is treated as live.
  // The probe is aborted once its status is in: an MJPEG body never ends, and open probes
  // fill the browser's six connections per host, so a newly selected stream never loads.
  useEffect(() => {
    if (h264) return;
    let stop = false;
    setOk(true);
    const ctl = new AbortController();
    const url = `${src}?probe=${encodeURIComponent(String(streamKey))}`;
    fetch(url, { method: "GET", headers: { Range: "bytes=0-1" }, signal: ctl.signal })
      .then((r) => {
        if (!stop) setOk(r.ok || r.status === 206);
        ctl.abort();
      })
      .catch(() => {
        if (!stop && !ctl.signal.aborted) setOk(false);
      });
    return () => {
      stop = true;
      ctl.abort();
    };
  }, [src, streamKey, h264]);

  // React's onWheel is passive, so the page would scroll too; a native listener can prevent that.
  useEffect(() => {
    const el = boxRef.current;
    if (!el || !onZoom) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      onZoom(Math.sign(e.deltaY));
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [onZoom]);

  const control = !!(onZoom || onDrag);
  return (
    <div
      ref={boxRef}
      className={`video ${control ? "video-control" : ""} ${className ?? ""}`}
      onPointerDown={(e) => {
        if (!onDrag || e.button !== 0) return;
        drag.current = { x: e.clientX, y: e.clientY };
        e.currentTarget.setPointerCapture(e.pointerId);
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d || !onDrag) return;
        onDrag(e.clientX - d.x, e.clientY - d.y);
        drag.current = { x: e.clientX, y: e.clientY };
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
      onDoubleClick={onReset}
    >
      <div className="video-title">
        {title}
        {control && <span className="video-hint">{hint ? `${hint} · ` : ""}scroll to zoom, drag to turn, double-click to reset</span>}
      </div>
      {!visible ? (
        <div className="video-placeholder">Video paused while the tab is hidden</div>
      ) : h264 ? (
        <canvas key={streamKey} ref={canvasRef} className="video-img" />
      ) : ok ? (
        <img
          key={streamKey}
          ref={imgRef}
          className="video-img"
          src={src}
          alt={title}
          draggable={false}
          onError={() => setOk(false)}
        />
      ) : (
        <div className="video-placeholder">{placeholder}</div>
      )}
    </div>
  );
}
