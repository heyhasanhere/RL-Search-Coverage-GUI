import { useEffect, useMemo, useState } from "react";
import area from "@turf/area";
import bbox from "@turf/bbox";
import distance from "@turf/distance";
import type { ClientMsg, Config, LonLat, Options, Region, Scenario, StateMsg, SurvivorMix } from "./types";

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
  rl_unity: "A trained FloodUAV agent (PPO, TRPO or DQN) flies D1; other search drones use rl",
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

// Survivor terrain mix in percent, the default of sar/prior/flood_bank.py SHARES.
const MIX_DEFAULT: SurvivorMix = { roofs: 50, high_ground: 25, bridges: 10, water: 10, uniform: 5 };
const MIX_LABELS: [keyof SurvivorMix, string, string][] = [
  ["roofs", "Roof", "Flat roofs of flooded buildings, above the water"],
  ["high_ground", "High ground", "Dry ground near the flood edge"],
  ["bridges", "Bridge", "Bridge decks above the water"],
  ["water", "Water", "In open flood water"],
  ["uniform", "Random", "Anywhere outdoors in the area"],
];

export default function MissionPanel({ config, state, polygon, lkp, regions, send }: Props) {
  const running = state?.mission.status === "running";
  const scenarios = config.scenarios ?? ["wilderness", "flood"];
  const [scenario, setScenario] = useState<Scenario>((scenarios[0] as Scenario) ?? "wilderness");
  const [mode, setMode] = useState(config.modes[0] ?? "");
  const nDrones = Math.max(config.drones.length, 1);
  // rl_unity: the agent flies D1 alone and the other drone stands by for rescue
  const defaultSearch = (m: string) => (m === "rl_unity" ? 1 : Math.min(2, nDrones));
  const [nSearch, setNSearch] = useState(defaultSearch(config.modes[0] ?? ""));
  // Trained agents for rl_unity; models dropped into the models folder show up when the list is opened.
  const [agents, setAgents] = useState(config.agents ?? []);
  const refreshAgents = () => {
    fetch("api/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((c: Config | null) => c?.agents && setAgents(c.agents))
      .catch(() => undefined);
  };
  const [expected, setExpected] = useState(3);
  const [spawn, setSpawn] = useState(15);
  const [mix, setMix] = useState<SurvivorMix>(MIX_DEFAULT);
  const mixTotal = Object.values(mix).reduce((a, b) => a + b, 0);
  const [lkpTimeH, setLkpTimeH] = useState(6);
  const [waterLevel, setWaterLevel] = useState(10.5);   // m above the normal river; floods the demo town
  const [options, setOptions] = useState<Options>({ auto_dispatch: true, auto_replace: false, auto_stage: false });
  const [speed, setSpeed] = useState(1);
  const [droneSpeed, setDroneSpeed] = useState<number | null>(null);
  const shownSpeed = droneSpeed ?? state?.drone_speed ?? 10;
  const sendDroneSpeed = () => send({ type: "set_drone_speed", speed: shownSpeed });
  // rl_unity: the agent keeps the flight limits it was trained with unless the operator overrides them
  const trainingLimits = mode === "rl_unity" && (state?.agent_limits ?? "training") === "training";
  const hasMission = !!state && state.mission.status !== "idle";
  const stopped = hasMission && (state.mission.status === "finished" || !!state.mission.stopping);

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
    if (!running) setNSearch(defaultSearch(v));
    if (v === "rl_unity") {
      send({ type: "set_agent_limits", limits: "training" });
      refreshAgents();
    }
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
        {config.survivor_layouts && (
          <label title="How many survivors New survivors places in the town (1-100)">
            Survivors in map
            <input
              type="number" min={1} max={100} step={1} value={spawn}
              onChange={(e) => setSpawn(Math.min(100, Math.max(1, Math.round(Number(e.target.value)) || 1)))}
            />
          </label>
        )}
      </div>
      {config.survivor_layouts && (
        <details className="mix" title="How likely each new survivor lands on each terrain kind; New survivors uses it and so does the search prior. Normalized to 100%">
          <summary>
            Survivor terrain mix <span className="mix-sum">{MIX_LABELS.map(([k]) => mix[k]).join(" / ")}</span>
          </summary>
          <div className="mix-head">
            <span className={mixTotal === 100 ? "" : "warn"}>total {mixTotal}%</span>
            <button onClick={() => setMix(MIX_DEFAULT)} title="Back to 50 / 25 / 10 / 10 / 5">Default</button>
          </div>
          <div className="mix-fields">
            {MIX_LABELS.map(([k, label, tip]) => (
              <label key={k} title={tip}>
                {label}
                <input
                  type="number" min={0} max={100} step={1} value={mix[k]}
                  onChange={(e) => setMix((m) => ({ ...m, [k]: Math.min(100, Math.max(0, Math.round(Number(e.target.value)) || 0)) }))}
                />
              </label>
            ))}
          </div>
        </details>
      )}
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
      {mode === "rl_unity" && (
        <label title="The trained agent that flies D1 on Unity's observation; drop new models into the models folder. It can change during a mission">
          Agent model
          <select value={state?.agent ?? ""} onFocus={refreshAgents} onMouseDown={refreshAgents}
            onChange={(e) => send({ type: "set_agent", agent: e.target.value })}>
            {agents.length === 0 && <option value="">No agents in models/</option>}
            {agents.map((a) => (
              <option key={a.id} value={a.id} title={a.id}>{a.label}</option>
            ))}
          </select>
        </label>
      )}
      {mode === "rl_unity" && (
        <div className="checks">
          <label title="The agent flies with the speed and tilt limits it was trained with (the build's 10 m/s, 3 m/s climb, 20 degree tilt); untick to use the Drone speed cap.">
            <input type="checkbox" checked={trainingLimits}
              onChange={() => send({ type: "set_agent_limits", limits: trainingLimits ? "bridge" : "training" })} />{" "}
            Training flight limits
          </label>
        </div>
      )}
      <label className="speed" title={trainingLimits ? "Off while the agent uses its training flight limits" : "Horizontal speed cap for all drones"}>
        Drone speed {trainingLimits ? "off" : `${shownSpeed.toFixed(0)} m/s`}
        <input
          type="range" min={2} max={15} step={1} value={shownSpeed} disabled={trainingLimits}
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
        <button disabled={!stopped} onClick={() => send({ type: "continue_mission" })}
          title="Resume the stopped mission: searched area, belief and survivors are kept; search drones take off again">
          Continue
        </button>
        <button disabled={!hasMission} onClick={() => send({ type: "restart_mission" })}
          title="Start the last mission again: drones back on their pads, batteries full">
          Restart
        </button>
        <button onClick={() => send({ type: "reset_mission" })}
          title="Clear the mission: drones back on their pads, batteries full, survivors cleared; nothing starts">
          Reset
        </button>
        {config.survivor_layouts && (
          <button disabled={mixTotal <= 0} onClick={() => send({ type: "new_survivors", count: spawn, mix })}
            title="Reset, then place 'Survivors in map' survivors at new random positions by the survivor terrain mix; Reset keeps them">
            New survivors
          </button>
        )}
      </div>
    </section>
  );
}