import { useEffect, useMemo, useState } from "react";
import area from "@turf/area";
import bbox from "@turf/bbox";
import distance from "@turf/distance";
import type { ClientMsg, Config, LonLat, Options, Region, Scenario, StateMsg } from "./types";

interface Props {
  config: Config;
  state: StateMsg | null;
  polygon: LonLat[] | null;
  lkp: LonLat | null;
  regions: Region[];
  send: (m: ClientMsg) => void;
}

const OPTION_LABELS: [keyof Options, string][] = [
  ["auto_dispatch", "Auto dispatch"],
  ["auto_replace", "Auto replace"],
  ["auto_stage", "Auto stage"],
];

const MODE_HINTS: Record<string, string> = {
  rl: "Trained MAPPO policy (main method)",
  rl_search: "Learned policy with tree search (main method)",
  mcts: "MCTS planner (strongest baseline)",
  planner: "Two-segment look-ahead planner",
  greedy: "Highest immediate expected detections",
  lawnmower: "Boustrophedon sweep (baseline)",
};

/** Area in km2 and the largest bounding-box side in meters. */
function measure(ring: LonLat[]) {
  const poly: GeoJSON.Feature<GeoJSON.Polygon> = {
    type: "Feature",
    properties: {},
    geometry: { type: "Polygon", coordinates: [ring] },
  };
  const [w, s, e, n] = bbox(poly);
  const midLat = (s + n) / 2;
  const width = distance([w, midLat], [e, midLat], { units: "meters" });
  const height = distance([w, s], [w, n], { units: "meters" });
  return { km2: area(poly) / 1e6, side: Math.max(width, height) };
}

export default function MissionPanel({ config, state, polygon, lkp, regions, send }: Props) {
  const running = state?.mission.status === "running";
  const scenarios = config.scenarios ?? ["wilderness", "flood"];
  const [scenario, setScenario] = useState<Scenario>((scenarios[0] as Scenario) ?? "wilderness");
  const [mode, setMode] = useState(config.modes[0] ?? "");
  const nDrones = Math.max(config.drones.length, 1);
  const [nSearch, setNSearch] = useState(Math.min(2, nDrones));
  const [expected, setExpected] = useState(3);
  const [lkpTimeH, setLkpTimeH] = useState(6);
  const [waterLevel, setWaterLevel] = useState(10.5);   // m above the normal river; floods the demo town
  const [options, setOptions] = useState<Options>({ auto_dispatch: true, auto_replace: false, auto_stage: false });
  const [speed, setSpeed] = useState(1);
  const [droneSpeed, setDroneSpeed] = useState<number | null>(null);
  const shownSpeed = droneSpeed ?? state?.drone_speed ?? 10;
  const sendDroneSpeed = () => send({ type: "set_drone_speed", speed: shownSpeed });
  const hasMission = !!state && state.mission.status !== "idle";

  // Follow the server while a mission runs (another client may change things).
  const srvMode = state?.mission.mode;
  const srvOpts = state?.options;
  useEffect(() => {
    if (running && srvMode) setMode(srvMode);
  }, [running, srvMode]);
  useEffect(() => {
    if (srvOpts) setOptions(srvOpts);
  }, [srvOpts?.auto_dispatch, srvOpts?.auto_replace, srvOpts?.auto_stage]); // eslint-disable-line react-hooks/exhaustive-deps

  const m = useMemo(() => (polygon && polygon.length >= 4 ? measure(polygon) : null), [polygon]);
  const maxSide = config.max_grid * config.cell_m;
  const isFlood = scenario === "flood";
  let problem = "";
  if (!m) problem = "Draw a search area";
  else if (m.km2 > config.max_area_km2) problem = `Area exceeds ${config.max_area_km2.toFixed(2)} km2`;
  else if (m.side > maxSide) problem = `Bounding box side ${(m.side / 1000).toFixed(2)} km exceeds ${(maxSide / 1000).toFixed(2)} km`;
  else if (!isFlood && !lkp) problem = "Wilderness needs a last-seen point";
  else if (isFlood && waterLevel < 0) problem = "Water level must be 0 m or more";

  const start = () => {
    if (problem || !polygon) return;
    send({
      type: "start_mission",
      polygon,
      ...(lkp ? { lkp } : {}),
      mode,
      n_search: nSearch,
      expected_survivors: expected,
      options,
      scenario,
      ...(!isFlood ? { lkp_time_h: lkpTimeH } : {}),
      ...(isFlood ? { water_level_m: waterLevel } : {}),
      ...(regions.length ? { regions } : {}),
    });
  };

  const changeMode = (v: string) => {
    setMode(v);
    if (running) send({ type: "set_mode", mode: v });
  };

  const toggle = (k: keyof Options) => {
    const v = !options[k];
    setOptions({ ...options, [k]: v });
    if (running) send({ type: "set_options", [k]: v });
  };

  return (
    <section className="panel">
      <h2>Mission</h2>
      <div className="area-info">
        {m ? (
          <>
            Area <b>{m.km2.toFixed(2)}</b> / {config.max_area_km2.toFixed(2)} km2, side {(m.side / 1000).toFixed(2)} km
          </>
        ) : (
          "No area drawn"
        )}
        {lkp && <div>LKP {lkp[1].toFixed(5)}, {lkp[0].toFixed(5)}</div>}
        {regions.length > 0 && <div>{regions.length} operator region(s), weight {regions[0].weight.toFixed(2)}</div>}
        {problem && m && <div className="err">{problem}</div>}
      </div>
      <div className="grid2">
        <label>
          Scenario
          <select value={scenario} disabled={running} onChange={(e) => setScenario(e.target.value as Scenario)}>
            {scenarios.map((x) => (
              <option key={x} value={x}>{x === "wilderness" ? "Wilderness" : x === "flood" ? "Flood" : x}</option>
            ))}
          </select>
        </label>
        <label>
          Mode
          <select value={mode} onChange={(e) => changeMode(e.target.value)} title={MODE_HINTS[mode]}>
            {config.modes.map((x) => (
              <option key={x} value={x}>{x}</option>
            ))}
          </select>
        </label>
        {isFlood ? (
          <label>
            Water level (m)
            <input
              type="number" min={0} step={0.1} value={waterLevel} disabled={running}
              onChange={(e) => setWaterLevel(Number(e.target.value))}
            />
          </label>
        ) : (
          <label>
            Hours since seen
            <input
              type="number" min={0.5} step={0.5} value={lkpTimeH} disabled={running}
              onChange={(e) => setLkpTimeH(Math.max(0.5, Number(e.target.value)))}
            />
          </label>
        )}
        <label>
          Drones
          <select value={nSearch} disabled={running} onChange={(e) => setNSearch(Number(e.target.value))}
            title="Drones that start searching; the others wait on their pads for rescue">
            {Array.from({ length: nDrones }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <label>
          Expected survivors
          <input
            type="number" min={0} step={1} value={expected} disabled={running}
            onChange={(e) => setExpected(Math.max(0, Number(e.target.value)))}
          />
        </label>
      </div>
      <div className="checks">
        {OPTION_LABELS.map(([k, label]) => (
          <label key={k}>
            <input type="checkbox" checked={options[k]} onChange={() => toggle(k)} /> {label}
          </label>
        ))}
      </div>
      {config.backend === "kinematic" && (
        <label className="speed">
          Sim speed {(state?.speed ?? speed).toFixed(1)}x
          <input
            type="range" min={0.5} max={20} step={0.5} value={speed}
            onChange={(e) => {
              const v = Number(e.target.value);
              setSpeed(v);
              send({ type: "set_speed", speed: v });
            }}
          />
        </label>
      )}
      <label className="speed" title="Horizontal speed cap for all drones">
        Drone speed {shownSpeed.toFixed(0)} m/s
        <input
          type="range" min={2} max={15} step={1} value={shownSpeed}
          onChange={(e) => setDroneSpeed(Number(e.target.value))}
          onMouseUp={sendDroneSpeed} onTouchEnd={sendDroneSpeed} onKeyUp={sendDroneSpeed}
        />
      </label>
      <div className="row">
        <button className="primary" disabled={running || !!problem} title={problem} onClick={start}>
          Start
        </button>
        <button className="danger" disabled={!running} onClick={() => send({ type: "stop_mission" })}>
          Stop
        </button>
        <button disabled={!hasMission} onClick={() => send({ type: "restart_mission" })}
          title="Start the last mission again: drones back on their pads, batteries full">
          Restart
        </button>
      </div>
    </section>
  );
}