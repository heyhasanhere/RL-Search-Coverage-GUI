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
}

/**
 * An MJPEG <img> that swaps to a clean placeholder when the stream is unavailable.
 * Browsers give no load event for MJPEG, so availability comes from the `onerror`
 * of a probe request the server answers with 404 when there is no bridge.
 */
export default function VideoPanel({ src, title, streamKey, placeholder, className }: Props) {
  const [ok, setOk] = useState(true);
  const imgRef = useRef<HTMLImageElement>(null);

  // A 404 from the server means there is no stream; anything else is treated as live.
  useEffect(() => {
    let stop = false;
    setOk(true);
    const url = `${src}?probe=${encodeURIComponent(String(streamKey))}`;
    fetch(url, { method: "GET", headers: { Range: "bytes=0-1" } })
      .then((r) => {
        if (!stop) setOk(r.ok || r.status === 206);
      })
      .catch(() => {
        if (!stop) setOk(false);
      });
    return () => {
      stop = true;
    };
  }, [src, streamKey]);

  return (
    <div className={`video ${className ?? ""}`}>
      <div className="video-title">{title}</div>
      {ok ? (
        <img
          key={streamKey}
          ref={imgRef}
          className="video-img"
          src={src}
          alt={title}
          onError={() => setOk(false)}
        />
      ) : (
        <div className="video-placeholder">{placeholder}</div>
      )}
    </div>
  );
}