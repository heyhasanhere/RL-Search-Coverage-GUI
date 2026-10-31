import { useEffect, useState } from "react";
import type { ClientMsg, Config, Options, StateMsg, SurvivorMix } from "./types";

const OPTION_LABELS: [keyof Options, string][] = [
  ["auto_dispatch", "Auto dispatch"],
  ["auto_replace", "Auto replace"],
  ["auto_stage", "Auto stage"],
  ["auto_switch", "Auto switch to rescue"],
];

const MODE_HINTS: Record<string, string> = {
  rl: "Trained MAPPO policy (main method)",
  rl_unity: "A trained FloodUAV agent (PPO, TRPO or DQN) flies D1; other search drones use rl",
  mcts: "MCTS planner (strongest baseline)",
  greedy: "Highest immediate expected detections",
  lawnmower: "Boustrophedon sweep (baseline)",
};

// Survivor terrain mix in percent, the default of sar/prior/flood_bank.py SHARES.
const MIX_DEFAULT: SurvivorMix = { roofs: 50, high_ground: 25, bridges: 10, water: 10, uniform: 5 };
const MIX_LABELS: [keyof SurvivorMix, string, string][] = [
  ["roofs", "Roof", "Flat roofs of flooded buildings, above the water"],
  ["high_ground", "High ground", "Dry ground near the flood edge"],
  ["bridges", "Bridge", "Bridge decks above the water"],
  ["water", "Water", "In open flood water"],
  ["uniform", "Random", "Anywhere outdoors in the area"],
];

// rl_unity flies slower by default
const defaultDroneSpeed = (m: string) => (m === "rl_unity" ? 3 : 15);

/** Mission settings shared by the Mission panel (Start, drone speed) and the Settings menu. */
export function useMissionSettings(config: Config | null, state: StateMsg | null, send: (m: ClientMsg) => void) {
  const running = state?.mission.status === "running";
  const [mode, setMode] = useState("");
  const [options, setOptions] = useState<Options>({ auto_dispatch: true, auto_replace: true, auto_stage: true, auto_switch: true, relook: false });
  const [expected, setExpected] = useState(15);
  const [lkpTimeH, setLkpTimeH] = useState(6);
  const [mix, setMix] = useState<SurvivorMix>(MIX_DEFAULT);
  const [spawn, setSpawn] = useState(15);
  const [droneSpeed, setDroneSpeed] = useState(15);
  // Trained agents for rl_unity; models dropped into the models folder show up when the list is opened.
  const [agents, setAgents] = useState(config?.agents ?? []);
  const refreshAgents = () => {
    fetch("api/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((c: Config | null) => c?.agents && setAgents(c.agents))
      .catch(() => undefined);
  };

  // First mode of the server once the config arrives.
  useEffect(() => {
    if (!config || mode) return;
    const m = config.modes[0] ?? "";
    setMode(m);
    setDroneSpeed(defaultDroneSpeed(m));
    setAgents(config.agents ?? []);
  }, [config, mode]);

  // Follow the server while a mission runs (another client may change things).
  const srvMode = state?.mission.mode;
  const srvOpts = state?.options;
  useEffect(() => {
    if (running && srvMode) setMode(srvMode);
  }, [running, srvMode]);
  useEffect(() => {
    if (srvOpts) setOptions(srvOpts);
  }, [srvOpts?.auto_dispatch, srvOpts?.auto_replace, srvOpts?.auto_stage, srvOpts?.auto_switch, srvOpts?.relook]); // eslint-disable-line react-hooks/exhaustive-deps

  const changeMode = (v: string) => {
    setMode(v);
    if (!running) {
      setDroneSpeed(defaultDroneSpeed(v));
      send({ type: "set_drone_speed", speed: defaultDroneSpeed(v) });
    }
    if (v === "rl_unity") refreshAgents();
    if (running) send({ type: "set_mode", mode: v });
  };

  const toggle = (k: keyof Options) => {
    const v = !options[k];
    setOptions({ ...options, [k]: v });
    if (running) send({ type: "set_options", [k]: v });
  };

  return {
    mode, changeMode, options, toggle, expected, setExpected, lkpTimeH, setLkpTimeH,
    mix, setMix, spawn, setSpawn, droneSpeed, setDroneSpeed, agents, refreshAgents,
  };
}

export type MissionSettings = ReturnType<typeof useMissionSettings>;

interface MenuProps {
  config: Config | null;
  state: StateMsg | null;
  ms: MissionSettings;
  send: (m: ClientMsg) => void;
  readOnly: boolean;
  showTruth: boolean;
  onShowTruth: (v: boolean) => void;
}

/** Status bar Settings menu: mission settings, then simulation settings. */
export function SettingsMenu({ config, state, ms, send, readOnly, showTruth, onShowTruth }: MenuProps) {
  const running = state?.mission.status === "running";
  const scenario = config?.scenarios?.[0] ?? "wilderness";
  const layouts = !!config?.survivor_layouts;
  const mixTotal = Object.values(ms.mix).reduce((a, b) => a + b, 0);
  const [speed, setSpeed] = useState(1);
  return (
    <div className="settings-menu">
      <h3>Mission settings</h3>
      <div className="settings-group">
        {OPTION_LABELS.map(([k, label]) => (
          <label key={k}>
            <input type="checkbox" checked={ms.options[k]} disabled={readOnly} onChange={() => ms.toggle(k)} /> {label}
          </label>
        ))}
        {ms.mode !== "rl_unity" && (
          <label title="A search drone that detects a survivor flies to a point beside it and looks again before confirming; untick to confirm on the first detection">
            <input type="checkbox" checked={ms.options.relook} disabled={readOnly} onChange={() => ms.toggle("relook")} /> Re-check detections
          </label>
        )}
      </div>
      <div className="settings-fields">
        <label className="field">
          Mode
          <select value={ms.mode} disabled={readOnly} onChange={(e) => ms.changeMode(e.target.value)} title={MODE_HINTS[ms.mode]}>
            {(config?.modes ?? []).map((x) => (
              <option key={x} value={x}>{x}</option>
            ))}
          </select>
        </label>
        {ms.mode === "rl_unity" && (
          <label className="field" title="The trained agent that flies D1 on Unity's observation; drop new models into the models folder. It can change during a mission">
            Model
            <select value={state?.agent ?? ""} disabled={readOnly} onFocus={ms.refreshAgents} onMouseDown={ms.refreshAgents}
              onChange={(e) => send({ type: "set_agent", agent: e.target.value })}>
              {ms.agents.length === 0 && <option value="">No agents in models/</option>}
              {ms.agents.map((a) => (
                <option key={a.id} value={a.id} title={a.id}>{a.label}</option>
              ))}
            </select>
          </label>
        )}
        <label className="field">
          Expected survivors
          <input
            type="number" min={0} step={1} value={ms.expected} disabled={running || readOnly}
            onChange={(e) => ms.setExpected(Math.max(0, Number(e.target.value)))}
          />
        </label>
        {scenario !== "flood" && (
          <label className="field">
            Hours since seen
            <input
              type="number" min={0.5} step={0.5} value={ms.lkpTimeH} disabled={running || readOnly}
              onChange={(e) => ms.setLkpTimeH(Math.max(0.5, Number(e.target.value)))}
            />
          </label>
        )}
      </div>
      {layouts && (
        <details className="mix" title="How likely each new survivor lands on each terrain kind; New survivors uses it and so does the search prior. Normalized to 100%">
          <summary>
            Survivor terrain mix <span className="mix-sum">{MIX_LABELS.map(([k]) => ms.mix[k]).join(" / ")}</span>
          </summary>
          <div className="mix-head">
            <span className={mixTotal === 100 ? "" : "warn"}>total {mixTotal}%</span>
            <button disabled={readOnly} onClick={() => ms.setMix(MIX_DEFAULT)} title="Back to 50 / 25 / 10 / 10 / 5">Default</button>
          </div>
          <div className="mix-fields">
            {MIX_LABELS.map(([k, label, tip]) => (
              <label key={k} title={tip}>
                {label}
                <input
                  type="number" min={0} max={100} step={1} value={ms.mix[k]} disabled={readOnly}
                  onChange={(e) => ms.setMix((m) => ({ ...m, [k]: Math.min(100, Math.max(0, Math.round(Number(e.target.value)) || 0)) }))}
                />
              </label>
            ))}
          </div>
        </details>
      )}

      <h3>Simulation</h3>
      <label title={state?.truth ? "Mark where the survivors really are" : "The server sends no ground truth (start it with --show-truth)"}>
        <input type="checkbox" checked={showTruth} disabled={!state?.truth}
          onChange={(e) => onShowTruth(e.target.checked)} />
        Show survivor truth on the map
      </label>
      {state?.graphics && (
        <label className="field" title="Render quality, anti-aliasing and video frame rate of the 3D views; Auto picks from this machine">
          Graphics
          <select value={state.graphics.preset} disabled={readOnly}
            onChange={(e) => send({ type: "set_graphics", preset: e.target.value })}>
            {state.graphics.presets.map((p) => (
              <option key={p} value={p}>{p === "auto" ? `Auto (${state.graphics!.auto})` : p[0].toUpperCase() + p.slice(1)}</option>
            ))}
          </select>
          <span className="hint">
            {Math.round(state.graphics.scale * 100)}% resolution, {state.graphics.msaa > 1 ? `${state.graphics.msaa}x anti-aliasing` : "no anti-aliasing"}, {state.graphics.fps} fps, {state.graphics.bitrate}bit/s
          </span>
        </label>
      )}
      {config?.backend === "kinematic" && (
        <label className="field">
          Sim speed {(state?.speed ?? speed).toFixed(1)}x
          <input
            type="range" min={0.5} max={20} step={0.5} value={speed} disabled={readOnly}
            onChange={(e) => {
              const v = Number(e.target.value);
              setSpeed(v);
              send({ type: "set_speed", speed: v });
            }}
          />
        </label>
      )}
      {layouts && (
        <div className="row">
          <label className="field" title="How many survivors New survivors places in the town (1-100)">
            Survivors in map
            <input
              type="number" min={1} max={100} step={1} value={ms.spawn} disabled={readOnly}
              onChange={(e) => ms.setSpawn(Math.min(100, Math.max(1, Math.round(Number(e.target.value)) || 1)))}
            />
          </label>
          <button disabled={readOnly || mixTotal <= 0} onClick={() => send({ type: "new_survivors", count: ms.spawn, mix: ms.mix })}
            title="Reset, then place 'Survivors in map' survivors at new random positions by the survivor terrain mix; Reset keeps them">
            New survivors
          </button>
        </div>
      )}
    </div>
  );
}
