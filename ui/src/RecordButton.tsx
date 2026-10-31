import { useCallback, useEffect, useState } from "react";
import { fmtTime } from "./types";

interface RecordStatus {
  state: "idle" | "recording" | "packing";
  name: string;
  videos: boolean;
  elapsed: number;
  last: string;
  error: string;
}

/**
 * Status bar control: record the running mission into a replay (`/api/record`).
 * The server records every 3D view; `onSaved` gets the new replay's name.
 */
export default function RecordButton({ disabled, onSaved }: { disabled: boolean; onSaved: (name: string) => void }) {
  const [st, setSt] = useState<RecordStatus | null>(null);
  const [error, setError] = useState("");

  const poll = useCallback(async () => {
    try {
      const r = await fetch("api/record");
      if (r.ok) setSt(await r.json());
    } catch {
      // the next poll tries again
    }
  }, []);

  useEffect(() => {
    void poll();
    const id = setInterval(poll, 1000);
    return () => clearInterval(id);
  }, [poll]);

  // A finished recording: refresh the replay list once.
  const last = st?.last ?? "";
  const [seen, setSeen] = useState<string | null>(null);
  useEffect(() => {
    if (seen === null) setSeen(last);
    else if (last && last !== seen) {
      setSeen(last);
      onSaved(last);
    }
  }, [last, seen, onSaved]);

  const call = async (what: "start" | "stop") => {
    setError("");
    const r = await fetch(`api/record/${what}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const js = await r.json();
    if (!r.ok) setError(js.error ?? `HTTP ${r.status}`);
    else setSt(js);
  };

  if (!st) return null;
  const err = error || st.error;
  return (
    <span className="record">
      {st.state === "idle" && (
        <button onClick={() => call("start")} disabled={disabled}
          title={st.videos ? "Record the mission with every 3D view for the replay" : "Record the mission log for the replay (no sim bridge: no 3D views)"}>
          <span className="rec-dot" /> Record
        </button>
      )}
      {st.state === "recording" && (
        <button className="rec-on" onClick={() => call("stop")} title={`Stop and save as ${st.name}`}>
          <span className="rec-dot" /> Stop {fmtTime(st.elapsed)}
        </button>
      )}
      {st.state === "packing" && <span className="muted small">Saving {st.name} ...</span>}
      {st.state === "idle" && !err && st.last && <span className="muted small" title="Last recording">saved {st.last}</span>}
      {err && <span className="err small" title={err}>record failed</span>}
    </span>
  );
}
