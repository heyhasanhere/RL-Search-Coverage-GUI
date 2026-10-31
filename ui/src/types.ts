// Mirrors docs/PROTOCOL.md. Coordinates are [lon, lat].
export type LonLat = [number, number];

export interface DroneInfo {
  id: number;
  name: string;
  role: "search" | "rescue";
}

export interface Config {
  home: LonLat;
  cell_m: number;
  max_grid: number;
  max_area_km2: number;
  modes: string[];
  /** rl_unity: the trained agents the sim bridge can fly D1 with. */
  agents?: { id: string; label: string }[];
  drones: DroneInfo[];
  backend: "kinematic" | "flooduav";
  detector: "sim";
  /** v2: true when the server has a sim bridge, so the video panels can stream. */
  bridge?: boolean;
  scenarios?: string[];
  /** FloodUAV: the server can place the survivors at new random positions (`new_survivors`). */
  survivor_layouts?: boolean;
  /** FloodUAV: the scene boundary, preset as the search area. */
  area?: LonLat[];
  /** FloodUAV: top-down render of the scene, drawn under everything else (corners TL, TR, BR, BL). */
  map_image?: { url: string; coordinates: LonLat[] };
  /** Public deployment: commands need the operator password (`POST /api/login`). */
  login_required?: boolean;
  /** This browser holds the operator login cookie, or no login is required. */
  operator?: boolean;
}

export type DroneState =
  | "IDLE" | "STANDBY" | "SEARCH" | "RESCUE_TRANSIT"
  | "ON_SURVIVOR" | "RETURNING" | "ABORT" | "LANDED" | "FAILED";

export type AltitudeLevel = "low" | "high" | "transit";

export interface Drone extends DroneInfo {
  state: DroneState;
  pos: LonLat;
  alt: number;
  heading: number;
  battery_s: number;
  battery_frac: number;
  segment: LonLat[];
  target: LonLat | null;
  standby_point: LonLat | null;
  task: string;
  /** v2 optional: the search altitude level the agent chose (plan 9.1). */
  level?: AltitudeLevel | null;
  /** v2 optional: height above ground or canopy, in meters. */
  alt_agl?: number | null;
  /** v2 optional: battery sortie number, 1 on the first flight. */
  sortie?: number | null;
  /** v2 optional: seconds left of a battery swap; null when not swapping. */
  swap_remaining?: number | null;
  /** Camera ground footprint half sizes in meters: [along-track, cross-track]. */
  footprint_m?: [number, number] | null;
}

export type SurvivorStatus = "candidate" | "confirmed" | "rejected" | "assigned" | "reached";

export interface Survivor {
  id: number;
  pos: LonLat;
  status: SurvivorStatus;
  score: number;
  views: number;
  has_snapshot: boolean;
  assigned: number | null;
  t_detected: number;
  t_confirmed: number | null;
  t_reached: number | null;
}

export interface Options {
  auto_dispatch: boolean;
  auto_replace: boolean;
  auto_stage: boolean;
  auto_switch: boolean;
  /** Fly a re-look from a displaced viewpoint before confirming a candidate; off: confirm on the first detection. */
  relook: boolean;
}

export interface Mission {
  status: "idle" | "running" | "finished";
  /** Stop was pressed (or no drone is searching): drones are flying home. */
  stopping?: boolean;
  mode: string;
  polygon: LonLat[];
  lkp: LonLat | null;
  expected_total: number;
  remaining_fraction: number;
  expected_found: number;
  covered_fraction: number;
}

export interface StateMsg {
  type: "state";
  t: number;
  speed: number;
  mission: Mission;
  drones: Drone[];
  survivors: Survivor[];
  options: Options;
  truth: LonLat[] | null;
  /** Operator drone speed cap in m/s; null: the vehicle default. */
  drone_speed?: number | null;
  /** rl_unity flight limits: "training" (the agent's own, speed cap off) or "bridge" (the operator's cap). */
  agent_limits?: "training" | "bridge";
  /** rl_unity: id of the trained agent that flies. */
  agent?: string | null;
  /** FloodUAV graphics preset and what it sets; null: the backend has none. */
  graphics?: Graphics | null;
}

export interface Graphics {
  /** "auto", "low", "medium" or "high". */
  preset: string;
  /** The preset auto stands for on this machine. */
  auto: string;
  /** Share of the view size Unity renders, then scales up. */
  scale: number;
  /** Anti-aliasing samples (1: off). */
  msaa: number;
  fps: number;
  /** H.264 bit rate per video pane, as ffmpeg takes it ("6M"). */
  bitrate: string;
  presets: string[];
}

export interface BeliefMsg {
  type: "belief";
  corners: [LonLat, LonLat, LonLat, LonLat];
  png: string;
  coverage_png: string;
}

export interface EventMsg {
  type: "event";
  t: number;
  level: "info" | "warn" | "error";
  text: string;
}

/** One prior component as sent by the server: a named base64 RGBA PNG on the grid. */
export interface PriorLayer {
  /** Stable key used by the overlay checkboxes, e.g. "blend" or "walker". */
  key: string;
  /** Short label for the checkbox, e.g. "Walker sim". */
  label: string;
  /** Base64 RGBA PNG, same size as the blend, aligned to `corners`. */
  png: string;
}

/**
 * `prior` (server to client, once per mission, PROTOCOL section 5.3).
 * Every PNG covers the same grid rectangle given by `corners`.
 */
export interface PriorMsg {
  type: "prior";
  /** Top-left, top-right, bottom-right, bottom-left (MapLibre image source order). */
  corners: [LonLat, LonLat, LonLat, LonLat];
  /** Grid width in cells; images are this many pixels wide. */
  nx: number;
  /** Grid height in cells; images are this many pixels tall. */
  ny: number;
  cell_m: number;
  /** Scenario the prior was built for; shown in the status bar. */
  scenario?: string;
  /** Blend of all components; drawn when no single layer is toggled on. */
  blend: PriorLayer;
  /** One entry per prior component (plan 6.2, 6.4). */
  layers: PriorLayer[];
}

export type ServerMsg = StateMsg | BeliefMsg | EventMsg | PriorMsg;

// --------------------------------------------------------------------- replay
// The split replay (plan section 11, step 7). Mirrors sar/mission/replay.py.

/** One row of `GET /api/replays`: a mission log the operator can replay. */
export interface ReplayInfo {
  id: string;
  path: string;
  scenario: string | null;
  mode: string | null;
  seed: number | null;
  /** Wall-clock time the mission started, as a Unix timestamp, when known. */
  started: number | null;
  /** Mission duration in seconds, when the log has track rows. */
  duration: number | null;
  grid: number | null;
  expected: number | null;
  n_frames: number;
  n_finds: number;
  /** Recorded 3D videos in a packed replay (runs/replays). */
  n_videos?: number;
  /** The name of a packed replay (runs/replays/<name>), null for a plain mission log. */
  recording?: string | null;
}

export interface ReplayDrone extends Partial<Omit<Drone, "id" | "name" | "state" | "pos" | "alt" | "heading">> {
  id: number;
  name: string | null;
  state: string | null;
  pos: LonLat;
  alt: number | null;
  heading: number | null;
}

/** One sampled time step: the drone positions and the found-expected value. */
export interface ReplayFrame {
  /** Seconds since the first frame. */
  t: number;
  /** Fraction of the expected survivors found so far. */
  found: number | null;
  /** Expected survivors found so far (absolute). */
  found_abs?: number | null;
  covered?: number | null;
  remaining?: number | null;
  drones: ReplayDrone[];
}

export interface ReplayBelief {
  t: number;
  corners: [LonLat, LonLat, LonLat, LonLat];
  png: string;
  coverage_png: string;
}

export interface ReplayFind {
  t: number;
  sid: number | null;
  pos: LonLat | null;
  status: "candidate" | "confirmed" | "reached";
  drone: number | null;
  true: boolean | null;
}

export interface ReplayMeta {
  id: string;
  path: string;
  scenario: string | null;
  mode: string | null;
  seed: number | null;
  started: number | null;
  duration: number | null;
  grid: number | null;
  expected: number | null;
  n_frames: number;
  n_finds: number;
  polygon: LonLat[] | null;
  lkp?: LonLat | null;
}

/** The body of `GET /api/replay/<id>`. */
export interface Replay {
  meta: ReplayMeta;
  frames: ReplayFrame[];
  belief: ReplayBelief[];
  finds: ReplayFind[];
  events: { t: number; level: string; text: string }[];
  /** The found-expected curve as [[t, value], ...]. */
  curve: [number, number][];
  /** Recorded views of a packed replay, keyed by view (follow, overview, cam0, ...). */
  videos?: Record<string, ReplayVideo>;
  /** Drone state changes with their task text. */
  states?: { t: number; drone: number; state: DroneState; text: string }[];
  /** Each survivor with the replay time of each status change. */
  survivors?: ReplaySurvivor[];
  /** The prior layers of the mission (newer logs). */
  prior?: Omit<PriorMsg, "type"> | null;
}

export interface ReplaySurvivor {
  id: number;
  pos: LonLat | null;
  score: number;
  drone: number | null;
  true: boolean | null;
  t_detected: number;
  t_confirmed: number | null;
  t_rejected: number | null;
  t_assigned: number | null;
  assigned: number | null;
  t_reached: number | null;
}

/** One recorded view: points map the replay time to the video time, both in seconds. */
export interface ReplayVideo {
  url: string;
  /** True when the video is stretched evenly over the mission (no recorded clock). */
  approx: boolean;
  duration: number | null;
  points: [number, number][];
}

export type Scenario = "wilderness" | "flood";

/** Operator-drawn priority region: a polygon ring plus the weight the search uses. */
export interface Region {
  polygon: LonLat[];
  weight: number;
}

/** Survivor terrain mix: how likely a new survivor lands on each terrain kind (sar/prior/flood_bank.py SHARES). */
export type SurvivorMix = Record<"roofs" | "high_ground" | "bridges" | "water" | "uniform", number>;

export type ClientMsg =
  | {
      type: "start_mission";
      polygon: LonLat[];
      lkp?: LonLat;
      mode: string;
      n_search: number;
      expected_survivors: number;
      options: Options;
      // v2 scenario inputs (PROTOCOL section 5.1); the server ignores unknown fields.
      scenario?: Scenario;
      /** Hours since the last sighting, wilderness only. Sets the walker time budget. */
      lkp_time_h?: number;
      /** Water level in meters, flood only. Drives the water surface and the flood prior. */
      water_level_m?: number;
      /** Operator priority regions, as search teams draw them (plan 6.2). */
      regions?: Region[];
    }
  | { type: "stop_mission" }
  /** Start the last mission again: drones back on their pads, batteries full. */
  | { type: "continue_mission" }
  | { type: "restart_mission" }
  /** Back to before the first start: no mission, drones on their pads, batteries full. Nothing starts. */
  | { type: "reset_mission" }
  /** FloodUAV: Reset, then `count` survivors (1-100, default the build's 15) at new random positions; later Resets keep them.
   * `mix`: survivor terrain mix, weights by kind (normalized); the flood prior uses the same mix. */
  | { type: "new_survivors"; count?: number; mix?: SurvivorMix }
  /** Full battery for a landed drone. */
  | { type: "recharge"; drone: number }
  /** Horizontal speed cap for all drones, m/s. */
  | { type: "set_drone_speed"; speed: number }
  | { type: "set_graphics"; preset: string }
  | { type: "set_mode"; mode: string }
  /** rl_unity: the agent flies with its training flight limits, or with the operator's speed cap. */
  | { type: "set_agent_limits"; limits: "training" | "bridge" }
  /** rl_unity: which trained agent flies (Config.agents id). */
  | { type: "set_agent"; agent: string }
  | { type: "recall"; drone: number }
  | { type: "standby"; drone: number; point?: LonLat }
  | { type: "launch"; drone: number; task: "search" | "rescue"; survivor?: number }
  | { type: "dispatch"; drone: number; survivor: number }
  | { type: "confirm"; survivor: number }
  | { type: "reject"; survivor: number }
  | { type: "set_options"; auto_dispatch?: boolean; auto_replace?: boolean; auto_stage?: boolean; auto_switch?: boolean; relook?: boolean }
  | { type: "set_speed"; speed: number }
  /** v2: points the follow camera and the camera panel. */
  | { type: "select"; drone?: number; survivor?: number }
  /** Zoom and angle of the 3D follow camera (distance m, yaw and pitch deg) or a drone camera (fov, pan, tilt deg). */
  | { type: "camera"; view: "follow"; drone: number; distance: number; yaw: number; pitch: number }
  | { type: "camera"; view: "drone"; drone: number; fov: number; pan: number; tilt: number };

export const SURVIVOR_COLORS: Record<SurvivorStatus, string> = {
  candidate: "#facc15",
  confirmed: "#ef4444",
  assigned: "#f97316",
  reached: "#22c55e",
  rejected: "#9ca3af",
};

/** One color per drone (by id), for its map circle and list tag; the state shows as the ring. */
export const DRONE_PALETTE = ["#3b82f6", "#f97316", "#a855f7", "#14b8a6", "#ec4899", "#eab308"];
export const droneColor = (id: number) => DRONE_PALETTE[id % DRONE_PALETTE.length];

export const DRONE_COLORS: Record<DroneState, string> = {
  IDLE: "#9ca3af",
  STANDBY: "#22d3ee",
  SEARCH: "#3b82f6",
  RESCUE_TRANSIT: "#f97316",
  ON_SURVIVOR: "#e879f9",
  RETURNING: "#a78bfa",
  ABORT: "#ef4444",
  LANDED: "#6b7280",
  FAILED: "#7f1d1d",
};

/** Altitude levels (plan 9.1). Shown next to the drone name on the map and in the list. */
export const LEVEL_COLORS: Record<AltitudeLevel, string> = {
  low: "#38bdf8",
  high: "#c084fc",
  transit: "#fb923c",
};

export const LEVEL_SHORT: Record<AltitudeLevel, string> = {
  low: "L",
  high: "H",
  transit: "T",
};

export const SCENARIO_LABELS: Record<string, string> = {
  wilderness: "Wilderness lost person",
  flood: "Flood town",
};

export function fmtTime(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "-";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}
