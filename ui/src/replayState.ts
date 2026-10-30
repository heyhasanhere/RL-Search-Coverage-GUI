// Rebuilds the live messages (state, belief, events) of a recorded mission at any
// replay time, so the replay screen can reuse the live operator panels.
import type { BeliefMsg, Drone, DroneState, EventMsg, LonLat, PriorMsg, Replay, StateMsg, Survivor } from "./types";

/** Seconds of drone track drawn behind each drone (in the segment layer). */
const TRAIL_S = 90;

/** Index of the last item with `t <= t` in a list sorted by time, or -1. */
function lastAt<T extends { t: number }>(items: T[], t: number): number {
  let lo = 0;
  let hi = items.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (items[mid].t <= t) {
      best = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return best;
}

function survivorsAt(replay: Replay, t: number): Survivor[] {
  const out: Survivor[] = [];
  for (const s of replay.survivors ?? []) {
    if (s.pos == null || s.t_detected > t) continue;
    const seen = (x: number | null) => (x != null && x <= t ? x : null);
    let status: Survivor["status"] = "candidate";
    if (seen(s.t_rejected) != null) status = "rejected";
    if (seen(s.t_confirmed) != null) status = "confirmed";
    if (seen(s.t_assigned) != null) status = "assigned";
    if (seen(s.t_reached) != null) status = "reached";
    out.push({
      id: s.id, pos: s.pos, status, score: s.score, views: 1, has_snapshot: false,
      assigned: seen(s.t_assigned) != null || seen(s.t_reached) != null ? s.assigned : null,
      t_detected: s.t_detected, t_confirmed: seen(s.t_confirmed), t_reached: seen(s.t_reached),
    });
  }
  return out;
}

/** The `state` message the live server would have sent at replay time `t`. */
export function replayStateAt(replay: Replay, t: number): StateMsg {
  const frames = replay.frames;
  const i = lastAt(frames, t);
  const frame = i >= 0 ? frames[i] : frames[0];
  const states = replay.states ?? [];
  const expected = replay.meta.expected ?? 0;
  const found = frame?.found_abs ?? (frame?.found != null ? frame.found * expected : 0);

  // Each drone's track over the last TRAIL_S seconds, drawn as its segment.
  const trails = new Map<number, LonLat[]>();
  for (let k = Math.max(i, 0); k >= 0 && frames[k].t >= t - TRAIL_S; k--) {
    for (const d of frames[k].drones) {
      if (!d.pos) continue;
      const arr = trails.get(d.id) ?? [];
      arr.unshift(d.pos);
      trails.set(d.id, arr);
    }
  }

  const drones: Drone[] = (frame?.drones ?? []).map((d) => {
    // Newer logs carry the task text in every track row.
    let task = d.task ?? "";
    for (let k = d.task != null ? -1 : lastAt(states, t); k >= 0; k--) {
      if (states[k].drone === d.id) {
        task = states[k].text;
        break;
      }
    }
    const name = d.name ?? `D${d.id}`;
    return {
      ...d,
      id: d.id, name, role: d.role ?? (name.startsWith("R") ? "rescue" : "search"),
      state: (d.state ?? "IDLE") as DroneState, pos: d.pos, alt: d.alt ?? 0, heading: d.heading ?? 0,
      // Older logs have no battery: NaN tells the drone list to hide it.
      battery_s: d.battery_s ?? NaN, battery_frac: d.battery_frac ?? NaN,
      // Older logs have no planned segment: draw the recent track instead.
      segment: d.segment ?? trails.get(d.id) ?? [], target: d.target ?? null, standby_point: d.standby_point ?? null,
      task, alt_agl: d.alt_agl ?? d.alt,
    };
  });

  const duration = replay.meta.duration ?? 0;
  return {
    type: "state", t, speed: 1,
    mission: {
      status: t >= duration ? "finished" : "running", mode: replay.meta.mode ?? "-",
      polygon: replay.meta.polygon ?? [], lkp: replay.meta.lkp ?? null, expected_total: expected,
      remaining_fraction: frame?.remaining ?? (expected ? Math.max(0, 1 - found / expected) : 1),
      expected_found: found, covered_fraction: frame?.covered ?? NaN,
    },
    drones, survivors: survivorsAt(replay, t),
    options: { auto_dispatch: false, auto_replace: false, auto_stage: false, auto_switch: false }, truth: null,
  };
}

/** The prior message of the mission, when the log has one. */
export function replayPrior(replay: Replay): PriorMsg | null {
  return replay.prior ? { type: "prior", ...replay.prior } : null;
}

/** The last belief snapshot at or before `t`. */
export function replayBeliefAt(replay: Replay, t: number): BeliefMsg | null {
  const i = lastAt(replay.belief, t);
  const b = replay.belief[Math.max(i, 0)];
  return b ? { type: "belief", corners: b.corners, png: b.png, coverage_png: b.coverage_png } : null;
}

/** The events up to `t`. */
export function replayEventsAt(replay: Replay, t: number): EventMsg[] {
  const out: EventMsg[] = [];
  for (const e of replay.events) {
    if (e.t > t) break;
    out.push({ type: "event", t: e.t, level: e.level as EventMsg["level"], text: e.text });
  }
  return out;
}
