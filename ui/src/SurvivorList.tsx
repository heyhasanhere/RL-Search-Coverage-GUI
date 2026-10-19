import { useState } from "react";
import type { ClientMsg, Drone, Survivor } from "./types";
import { SURVIVOR_COLORS, fmtTime } from "./types";

interface Props {
  survivors: Survivor[];
  drones: Drone[];
  selected: number | null;
  onSelect: (survivor: number) => void;
  send: (m: ClientMsg) => void;
  /** Replay: show the decisions, without the command buttons. */
  readOnly?: boolean;
}

function SurvivorRow({ s, drones, selected, onSelect, send, readOnly }: {
  s: Survivor; drones: Drone[]; selected: boolean; onSelect: (sid: number) => void; send: (m: ClientMsg) => void;
  readOnly?: boolean;
}) {
  const defaultDrone = drones.find((d) => d.role === "rescue") ?? drones[0];
  const [drone, setDrone] = useState<string>(defaultDrone ? String(defaultDrone.id) : "");
  const assignedName = s.assigned != null ? drones.find((d) => d.id === s.assigned)?.name ?? `#${s.assigned}` : null;
  return (
    <li className={`card survivor ${selected ? "sel" : ""}`} onClick={() => onSelect(s.id)}>
      {s.has_snapshot && (
        <a href={`api/snapshot/${s.id}.jpg`} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}>
          <img className="thumb" src={`api/snapshot/${s.id}.jpg?v=${s.views}`} alt={`survivor ${s.id}`} />
        </a>
      )}
      <div className="grow">
        <div className="card-head">
          <b>#{s.id}</b>
          <span className="badge dark" style={{ background: SURVIVOR_COLORS[s.status] }}>{s.status}</span>
          <span className="muted small">score {s.score.toFixed(2)} - {s.views} views</span>
        </div>
        <div className="muted small">
          det {fmtTime(s.t_detected)} - conf {fmtTime(s.t_confirmed)} - reached {fmtTime(s.t_reached)}
          {assignedName && <> - assigned {assignedName}</>}
        </div>
        {!readOnly && (
        <div className="row wrap" onClick={(e) => e.stopPropagation()}>
          {s.status === "candidate" && (
            <>
              <button className="ok" onClick={() => send({ type: "confirm", survivor: s.id })}>Confirm</button>
              <button onClick={() => send({ type: "reject", survivor: s.id })}>Reject</button>
            </>
          )}
          {s.status !== "rejected" && s.status !== "reached" && (
            <>
              <select value={drone} onChange={(e) => setDrone(e.target.value)}>
                {drones.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}</option>
                ))}
              </select>
              <button disabled={!drone} onClick={() => send({ type: "dispatch", drone: Number(drone), survivor: s.id })}>
                Dispatch
              </button>
            </>
          )}
        </div>
        )}
      </div>
    </li>
  );
}

// Candidates first (they need a decision), then the rest newest first.
const ORDER: Record<Survivor["status"], number> = {
  candidate: 0, confirmed: 1, assigned: 2, reached: 3, rejected: 4,
};

export default function SurvivorList({ survivors, drones, selected, onSelect, send, readOnly }: Props) {
  const sorted = [...survivors].sort((a, b) => ORDER[a.status] - ORDER[b.status] || b.t_detected - a.t_detected);
  return (
    <section className="panel">
      <h2>Survivors ({survivors.length})</h2>
      {survivors.length === 0 && <div className="muted small">None detected</div>}
      <ul className="list">
        {sorted.map((s) => (
          <SurvivorRow key={s.id} s={s} drones={drones} selected={selected === s.id} onSelect={onSelect} send={send} readOnly={readOnly} />
        ))}
      </ul>
    </section>
  );
}