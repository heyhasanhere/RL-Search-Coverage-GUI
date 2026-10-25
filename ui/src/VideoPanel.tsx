import { useEffect, useRef, useState } from "react";

interface Props {
  /** MJPEG path on the server, e.g. "video/follow.mjpg". */
  src: string;
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

/**
 * An MJPEG <img> that swaps to a clean placeholder when the stream is unavailable.
 * Browsers give no load event for MJPEG, so availability comes from the `onerror`
 * of a probe request the server answers with 404 when there is no bridge.
 */
export default function VideoPanel({ src, title, streamKey, placeholder, className, onZoom, onDrag, onReset, hint }: Props) {
  const [ok, setOk] = useState(true);
  const imgRef = useRef<HTMLImageElement>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);

  // A 404 from the server means there is no stream; anything else is treated as live.
  // The probe is aborted once its status is in: an MJPEG body never ends, and open probes
  // fill the browser's six connections per host, so a newly selected stream never loads.
  useEffect(() => {
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
  }, [src, streamKey]);

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
      {ok ? (
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
