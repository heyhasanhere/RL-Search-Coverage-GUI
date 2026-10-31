import type { ReplayInfo } from "./types";
import { fmtTime } from "./types";

export const SPEEDS = [1, 4, 16];

function label(info: ReplayInfo): string {
  const bits = [info.mode ?? "?", info.scenario ?? "?"];
  if (info.recording) bits.unshift(info.recording);
  if (info.seed != null) bits.push(`seed ${info.seed}`);
  if (info.duration != null) bits.push(fmtTime(info.duration));
  if (info.n_videos) bits.push("3D");
  return bits.join(" - ");
}

interface Props {
  /** The missions saved with the Record button. */
  list: ReplayInfo[];
  id: string;
  onPick: (id: string) => void;
  t: number;
  duration: number;
  playing: boolean;
  speed: number;
  onSeek: (t: number) => void;
  onPlay: (playing: boolean) => void;
  onSpeed: (speed: number) => void;
  error: string | null;
}

/** Replay mode: takes the place of the mission panel (pick a recording, play, seek). */
export default function ReplayControls({ list, id, onPick, t, duration, playing, speed, onSeek, onPlay, onSpeed, error }: Props) {
  return (
    <section className="panel replay-controls-panel">
      <h2>Replay</h2>
      <select value={id} onChange={(e) => onPick(e.target.value)} title="Recorded mission">
        {list.length === 0 && <option value="">no recordings</option>}
        {list.map((r) => (
          <option key={r.id} value={r.id}>{label(r)}</option>
        ))}
      </select>
      <div className="row">
        <button className="ok" onClick={() => onPlay(!playing)} disabled={!id}>{playing ? "Pause" : "Play"}</button>
        {SPEEDS.map((s) => (
          <button key={s} className={speed === s ? "active" : ""} onClick={() => onSpeed(s)}>{s}x</button>
        ))}
        <span className="muted small">{fmtTime(t)} / {fmtTime(duration)}</span>
      </div>
      <input
        className="replay-slider" type="range" min={0} max={Math.max(duration, 0.1)} step={0.1}
        value={Math.min(t, duration)} onChange={(e) => onSeek(Number(e.target.value))}
      />
      {error && <div className="err small">{error}</div>}
    </section>
  );
}
