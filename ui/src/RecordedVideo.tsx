import { useEffect, useRef, useState } from "react";
import type { ReplayVideo } from "./types";

/** Browsers do not play faster than 16x; past that the video catches up by seeking. */
const MAX_RATE = 16;

/** The video time at replay time `t`, and the video seconds per replay second there. */
export function videoAt(points: [number, number][], t: number): { vt: number; slope: number } {
  if (!points.length) return { vt: 0, slope: 1 };
  if (points.length === 1 || t <= points[0][0]) {
    const slope = points.length > 1 ? (points[1][1] - points[0][1]) / Math.max(points[1][0] - points[0][0], 1e-6) : 1;
    return { vt: points[0][1], slope };
  }
  for (let i = 1; i < points.length; i++) {
    const [t1, v1] = points[i];
    if (t <= t1) {
      const [t0, v0] = points[i - 1];
      const slope = (v1 - v0) / Math.max(t1 - t0, 1e-6);
      return { vt: v0 + (t - t0) * slope, slope };
    }
  }
  return { vt: points[points.length - 1][1], slope: 0 };
}

interface Props {
  /** The recorded view, or undefined when this view was not recorded. */
  video: ReplayVideo | undefined;
  /** The replay time in seconds (the shared slider). */
  t: number;
  playing: boolean;
  /** Replay speed (1x, 4x, 16x). */
  speed: number;
  title: string;
  placeholder: string;
}

/**
 * A recorded view of a packed replay, in the same frame as the live video panels,
 * kept in step with the time slider. While playing the video runs at its own rate
 * (replay speed times the recorded slope) and is only re-seeked when it drifts;
 * while paused it follows the slider.
 */
export default function RecordedVideo({ video, t, playing, speed, title, placeholder }: Props) {
  // Bumped when the video has its metadata, so the first sync waits for it.
  const [ready, setReady] = useState(0);
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !video || el.readyState < 1) return;
    const { vt, slope } = videoAt(video.points, t);
    const rate = speed * slope;
    if (!playing || rate <= 0) {
      if (!el.paused) el.pause();
      if (Math.abs(el.currentTime - vt) > 0.05) el.currentTime = vt;
      return;
    }
    const r = Math.min(Math.max(rate, 0.0625), MAX_RATE);
    if (Math.abs(el.playbackRate - r) > 1e-3) el.playbackRate = r;
    // Allow more drift at high rates, so the video is not re-seeked every frame.
    if (Math.abs(el.currentTime - vt) > Math.max(0.75, rate * 0.25)) el.currentTime = vt;
    if (el.paused) void el.play().catch(() => undefined);
  }, [t, playing, speed, video, ready]);

  return (
    <div className="video">
      <div className="video-title">
        {title}
        {video?.approx && <span className="muted" title="No clock was recorded; the video is stretched evenly over the mission"> - timing approximate</span>}
      </div>
      {video ? (
        <video
          key={video.url} ref={ref} className="video-img" src={video.url} muted playsInline preload="auto"
          onLoadedMetadata={() => setReady((r) => r + 1)}
        />
      ) : (
        <div className="video-placeholder">{placeholder}</div>
      )}
    </div>
  );
}
