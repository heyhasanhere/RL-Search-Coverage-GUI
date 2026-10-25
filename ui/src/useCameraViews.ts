import { useCallback, useRef, useState } from "react";
import type { ClientMsg } from "./types";

/** Follow camera: distance from the drone (m) and its direction, yaw from behind and pitch above the drone. */
export interface FollowView { distance: number; yaw: number; pitch: number }
/** Drone camera: vertical field of view and the gimbal, tilt from straight down and pan sideways (degrees). */
export interface DroneView { fov: number; pan: number; tilt: number }

// The defaults match the bridge's start: follow offset 0,30,-30 and the 60 degree down camera.
export const FOLLOW_DEFAULT: FollowView = { distance: 42, yaw: 0, pitch: 45 };
export const DRONE_DEFAULT: DroneView = { fov: 60, pan: 0, tilt: 0 };

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const wrap = (v: number) => ((((v + 180) % 360) + 360) % 360) - 180;
const SEND_MS = 100;

/** Zoom and angle of the 3D follow camera and of each drone camera, sent to the sim as they change. */
export function useCameraViews(send: (m: ClientMsg) => void) {
  const [follow, setFollow] = useState<FollowView>(FOLLOW_DEFAULT);
  const [cams, setCams] = useState<Record<number, DroneView>>({});
  // At most one message per view every SEND_MS while the operator drags or scrolls.
  const pending = useRef(new Map<string, ClientMsg>());
  const timer = useRef<number | null>(null);
  const queue = useCallback(
    (key: string, msg: ClientMsg) => {
      pending.current.set(key, msg);
      if (timer.current != null) return;
      timer.current = window.setTimeout(() => {
        timer.current = null;
        for (const m of pending.current.values()) send(m);
        pending.current.clear();
      }, SEND_MS);
    },
    [send],
  );

  const setFollowView = useCallback(
    (f: (v: FollowView) => FollowView) => {
      setFollow((prev) => {
        const v = f(prev);
        queue("follow", { type: "camera", view: "follow", ...v });
        return v;
      });
    },
    [queue],
  );
  const setDroneView = useCallback(
    (drone: number, f: (v: DroneView) => DroneView) => {
      setCams((prev) => {
        const v = f(prev[drone] ?? DRONE_DEFAULT);
        queue(`cam${drone}`, { type: "camera", view: "drone", drone, ...v });
        return { ...prev, [drone]: v };
      });
    },
    [queue],
  );

  /** Wheel: steps > 0 zoom out. Drag: pixels moved. */
  const followZoom = useCallback(
    (steps: number) => setFollowView((v) => ({ ...v, distance: clamp(v.distance * 1.12 ** steps, 8, 250) })),
    [setFollowView],
  );
  const followDrag = useCallback(
    (dx: number, dy: number) =>
      setFollowView((v) => ({ ...v, yaw: wrap(v.yaw - dx * 0.4), pitch: clamp(v.pitch + dy * 0.3, 2, 89) })),
    [setFollowView],
  );
  const followReset = useCallback(() => setFollowView(() => FOLLOW_DEFAULT), [setFollowView]);
  const camZoom = useCallback(
    (drone: number, steps: number) => setDroneView(drone, (v) => ({ ...v, fov: clamp(v.fov * 1.1 ** steps, 8, 100) })),
    [setDroneView],
  );
  const camDrag = useCallback(
    (drone: number, dx: number, dy: number) =>
      setDroneView(drone, (v) => {
        const k = v.fov / 60;          // finer steps when zoomed in
        return { ...v, pan: clamp(v.pan + dx * 0.2 * k, -90, 90), tilt: clamp(v.tilt - dy * 0.2 * k, 0, 85) };
      }),
    [setDroneView],
  );
  const camReset = useCallback((drone: number) => setDroneView(drone, () => DRONE_DEFAULT), [setDroneView]);

  return { follow, cams, followZoom, followDrag, followReset, camZoom, camDrag, camReset };
}
