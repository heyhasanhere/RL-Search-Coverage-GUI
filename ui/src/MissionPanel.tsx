import { useMemo } from "react";
import area from "@turf/area";
import bbox from "@turf/bbox";
import distance from "@turf/distance";
import type { MissionSettings } from "./MissionSettings";
import type { ClientMsg, Config, LonLat, Region, Scenario, StateMsg } from "./types";

interface Props {
  config: Config;
  state: StateMsg | null;
  polygon: LonLat[] | null;
  lkp: LonLat | null;
  regions: Region[];
  ms: MissionSettings;
  send: (m: ClientMsg) => void;
}

// Water level of a flood mission, m above the normal river; floods the demo town.
const WATER_LEVEL_M = 10.5;

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

export default function MissionPanel({ config, state, polygon, lkp, regions, ms, send }: Props) {
  const running = state?.mission.status === "running";
  const scenario = (config.scenarios?.[0] ?? "wilderness") as Scenario;
  const isFlood = scenario === "flood";
  const nDrones = Math.max(config.drones.length, 1);
  // rl_unity: the agent flies D1 alone and the other drone stands by for rescue
  const nSearch = ms.mode === "rl_unity" ? 1 : Math.min(2, nDrones);
  const sendDroneSpeed = () => send({ type: "set_drone_speed", speed: ms.droneSpeed });
  const hasMission = !!state && state.mission.status !== "idle";
  const stopped = hasMission && (state.mission.status === "finished" || !!state.mission.stopping);

  const m = useMemo(() => (polygon && polygon.length >= 4 ? measure(polygon) : null), [polygon]);
  const maxSide = config.max_grid * config.cell_m;
  let problem = "";
  if (!m) problem = "Draw a search area";
  else if (m.km2 > config.max_area_km2) problem = `Area exceeds ${config.max_area_km2.toFixed(2)} km2`;
  else if (m.side > maxSide) problem = `Bounding box side ${(m.side / 1000).toFixed(2)} km exceeds ${(maxSide / 1000).toFixed(2)} km`;
  else if (!isFlood && !lkp) problem = "Wilderness needs a last-seen point";

  const start = () => {
    if (problem || !polygon) return;
    sendDroneSpeed();
    send({
      type: "start_mission",
      polygon,
      ...(lkp ? { lkp } : {}),
      mode: ms.mode,
      n_search: nSearch,
      expected_survivors: ms.expected,
      options: ms.options,
      scenario,
      ...(!isFlood ? { lkp_time_h: ms.lkpTimeH } : {}),
      ...(isFlood ? { water_level_m: WATER_LEVEL_M } : {}),
      ...(regions.length ? { regions } : {}),
    });
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
      <label className="speed" title="Horizontal speed cap for all drones">
        Drone speed {ms.droneSpeed.toFixed(0)} m/s
        <input
          type="range" min={2} max={15} step={1} value={ms.droneSpeed}
          onChange={(e) => ms.setDroneSpeed(Number(e.target.value))}
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
      </div>
    </section>
  );
}
