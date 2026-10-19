import { useState } from "react";
import type { ClientMsg, Drone, Survivor } from "./types";
import { DRONE_COLORS, LEVEL_COLORS, droneColor, fmtTime } from "./types";

interface Props {
  drones: Drone[];
  survivors: Survivor[];
  pickingFor: number | null;
  selected: number | null;
  onSelect: (drone: number) => void;
  onPickStandby: (drone: number | null) => void;
  send: (m: ClientMsg) => void;
  /** Replay: no commands, and no battery (it is not in the mission log). */
  readOnly?: boolean;
}

function DroneRow({ d, survivors, picking, selected, onSelect, onPickStandby, send, readOnly }: {
  d: Drone; survivors: Survivor[]; picking: boolean; selected: boolean;
  onSelect: (drone: number) => void; onPickStandby: (drone: number | null) => void; send: (m: ClientMsg) => void;
  readOnly?: boolean;
}) {
  const [rescueTarget, setRescueTarget] = useState("");
  const pct = Math.round(d.battery_frac * 100);
  const battColor = d.battery_frac > 0.5 ? "#22c55e" : d.battery_frac > 0.25 ? "#facc15" : "#ef4444";
  const rescuable = survivors.filter((s) => s.status === "confirmed" || s.status === "candidate");
  const launchRescue = () =>
    send({ type: "launch", drone: d.id, task: "rescue", ...(rescueTarget ? { survivor: Number(rescueTarget) } : {}) });
  const agl = d.alt_agl ?? d.alt;
  const swapping = d.swap_remaining != null && d.swap_remaining > 0;
  const onGround = d.state === "LANDED" || d.state === "IDLE" || (d.state === "STANDBY" && d.task.startsWith("landed"));
  return (
    <li className={`card drone ${selected ? "sel" : ""}`} onClick={() => onSelect(d.id)}>
      <div className="card-head">
        <b><span className="drone-tag" style={{ background: droneColor(d.id) }} />{d.name}</b>
        <span className="muted">{d.role}</span>
        {d.level && (
          <span className="lvl" style={{ background: LEVEL_COLORS[d.level], color: "#0b0f14" }}>
            {d.level}
          </span>
        )}
        <span className="badge" style={{ background: DRONE_COLORS[d.state] }}>{d.state}</span>
      </div>
      {Number.isNaN(d.battery_frac) ? (
        <div className="muted small">{agl.toFixed(0)} m altitude</div>
      ) : (
        <>
          <div className="batt" title={`${pct}%`}>
            <div style={{ width: `${pct}%`, background: battColor }} />
          </div>
          <div className="muted small">
            {pct}% - {fmtTime(d.battery_s)} left - {agl.toFixed(0)} m AGL
            {d.sortie != null && d.sortie > 1 && <> - sortie {d.sortie}</>}
          </div>
        </>
      )}
      {swapping && <div className="small swap">Battery swap {fmtTime(d.swap_remaining)}</div>}
      {d.task && <div className="small">{d.task}</div>}
      {!readOnly && (
      <>
      <div className="row wrap" onClick={(e) => e.stopPropagation()}>
        <button onClick={() => send({ type: "recall", drone: d.id })}>Recall</button>
        {picking ? (
          <>
            <button onClick={() => { send({ type: "standby", drone: d.id }); onPickStandby(null); }}>At home</button>
            <button onClick={() => onPickStandby(null)}>Cancel</button>
          </>
        ) : (
          <button onClick={() => onPickStandby(d.id)} title="Then click the map, or choose At home">Standby</button>
        )}
        <button onClick={() => send({ type: "launch", drone: d.id, task: "search" })}>Launch search</button>
        <button disabled={!onGround} onClick={() => send({ type: "recharge", drone: d.id })}
          title="Full battery; the drone must be landed">Recharge</button>
      </div>
      <div className="row" onClick={(e) => e.stopPropagation()}>
        <select value={rescueTarget} onChange={(e) => setRescueTarget(e.target.value)}>
          <option value="">auto target</option>
          {rescuable.map((s) => (
            <option key={s.id} value={s.id}>survivor #{s.id}</option>
          ))}
        </select>
        <button onClick={launchRescue}>Launch rescue</button>
      </div>
      </>
      )}
    </li>
  );
}

export default function DroneList({ drones, survivors, pickingFor, selected, onSelect, onPickStandby, send, readOnly }: Props) {
  return (
    <section className="panel">
      <h2>Drones</h2>
      {drones.length === 0 && <div className="muted small">No drone state yet</div>}
      <ul className="list">
        {drones.map((d) => (
          <DroneRow
            key={d.id} d={d} survivors={survivors} picking={pickingFor === d.id} selected={selected === d.id}
            onSelect={onSelect} onPickStandby={onPickStandby} send={send} readOnly={readOnly}
          />
        ))}
      </ul>
    </section>
  );
}