import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMission } from "./useMission";
import { useReplay } from "./useReplay";
import MapView, { type BaseLayer, type DrawTool } from "./MapView";
import MissionPanel from "./MissionPanel";
import DroneList from "./DroneList";
import SurvivorList from "./SurvivorList";
import EventLog from "./EventLog";
import VideoPanel from "./VideoPanel";
import PriorPanel from "./PriorPanel";
import RecordedVideo from "./RecordedVideo";
import ReplayControls from "./ReplayControls";
import { replayBeliefAt, replayEventsAt, replayPrior, replayStateAt } from "./replayState";
import type { LonLat, Region } from "./types";
import { fmtTime } from "./types";

const pct = (x: number | undefined) => (x == null || Number.isNaN(x) ? "-" : `${(x * 100).toFixed(0)}%`);
const DEFAULT_REGION_WEIGHT = 0.3;

export default function App() {
  const live = useMission();
  const { config, connected } = live;
  const replay = useReplay();
  const [view, setView] = useState<"live" | "replay">("live");
  const isReplay = view === "replay";
  const [replayId, setReplayId] = useState("");
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [polygon, setPolygon] = useState<LonLat[] | null>(null);
  const [lkp, setLkp] = useState<LonLat | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [regionWeight, setRegionWeight] = useState(DEFAULT_REGION_WEIGHT);
  const [pickingFor, setPickingFor] = useState<number | null>(null);
  const [drawTool, setDrawTool] = useState<DrawTool>("select");
  const [base, setBase] = useState<BaseLayer>("esri");
  const [showBelief, setShowBelief] = useState(true);
  const [showCoverage, setShowCoverage] = useState(false);
  const [showTruth, setShowTruth] = useState(false);
  const [showPrior, setShowPrior] = useState<Set<string>>(new Set());
  const [selectedDrone, setSelectedDrone] = useState<number | null>(null);
  const [selectedSurvivor, setSelectedSurvivor] = useState<number | null>(null);
  // Map panel background: the scene map or the live top-down overview (scene backends only).
  const [mapMode, setMapMode] = useState<"map" | "overview">("map");

  // ---------------------------------------------------------------- replay
  // One recorded mission shown in the live layout: the panels get the state,
  // belief and events rebuilt from the mission log at the slider time.
  useEffect(() => {
    if (replayId || !replay.list.length) return;
    const withVideo = replay.list.filter((r) => r.n_videos);
    setReplayId((withVideo[withVideo.length - 1] ?? replay.list[replay.list.length - 1]).id);
  }, [replay.list, replayId]);
  const { load } = replay;
  useEffect(() => {
    if (isReplay && replayId) void load(replayId);
  }, [isReplay, replayId, load]);
  const rec = replayId ? replay.replays[replayId] ?? null : null;
  const duration = rec?.meta.duration ?? 0;

  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || !isReplay) return;
    let last = performance.now();
    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setT((prev) => {
        const next = prev + dt * speed;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => {
      if (raf.current != null) cancelAnimationFrame(raf.current);
    };
  }, [playing, isReplay, speed, duration]);

  const shown = useMemo(() => {
    if (!isReplay) return { state: live.state, belief: live.belief, prior: live.prior, events: live.events };
    if (!rec) return { state: null, belief: null, prior: null, events: [] };
    return { state: replayStateAt(rec, t), belief: replayBeliefAt(rec, t), prior: replayPrior(rec), events: replayEventsAt(rec, t) };
  }, [isReplay, live.state, live.belief, live.prior, live.events, rec, t]);
  const { state, belief, prior, events } = shown;
  // Replay is read-only: nothing goes to the live server.
  const send = useMemo(() => (isReplay ? () => undefined : live.send), [isReplay, live.send]);
  const pickReplay = useCallback((id: string) => {
    setReplayId(id);
    setT(0);
    setPlaying(false);
  }, []);

  const onAreaChange = useCallback((p: LonLat[] | null, l: LonLat | null, r: Region[]) => {
    setPolygon(p);
    setLkp(l);
    setRegions(r);
  }, []);

  const onPick = useCallback(
    (p: LonLat) => {
      if (pickingFor == null) return;
      send({ type: "standby", drone: pickingFor, point: p });
      setPickingFor(null);
    },
    [pickingFor, send],
  );

  // Selecting a drone in any panel points the video panels at it (plan 11.1).
  const selectDrone = useCallback(
    (d: number) => {
      setSelectedDrone(d);
      setSelectedSurvivor(null);
      send({ type: "select", drone: d });
    },
    [send],
  );
  const selectSurvivor = useCallback(
    (sid: number) => {
      setSelectedSurvivor(sid);
      send({ type: "select", survivor: sid });
    },
    [send],
  );

  const togglePrior = useCallback((key: string) => {
    setShowPrior((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // Default the camera panel to the first drone once state arrives.
  if (selectedDrone == null && state?.drones.length) {
    const first = state.drones[0];
    if (first) setSelectedDrone(first.id);
  }

  const mission = state?.mission;
  const running = mission?.status === "running";
  const hasBridge = config?.bridge ?? false;
  const camDrone = selectedDrone ?? state?.drones[0]?.id ?? 0;
  const camName = state?.drones.find((d) => d.id === camDrone)?.name ?? `D${camDrone}`;
  const openCandidates = state?.survivors.filter((s) => s.status === "candidate").length ?? 0;
  const scenario = prior?.scenario ?? (config?.scenarios?.[0] ?? "wilderness");

  return (
    <div className="app">
      <header className="statusbar">
        <div className="brand">
          SAR Operator
          {isReplay ? (
            <span className="conn on">replay</span>
          ) : (
            <span className={`conn ${connected ? "on" : "off"}`}>{connected ? "connected" : "reconnecting"}</span>
          )}
        </div>
        <span>Scenario <b>{scenario}</b></span>
        <span>Mission <b>{mission?.status ?? "-"}</b></span>
        <span>Mode <b>{mission?.mode ?? "-"}</b></span>
        <span>Time <b>{fmtTime(state?.t)}</b></span>
        <span>
          Found <b>{mission ? mission.expected_found.toFixed(2) : "-"}</b>
          {mission && <> / {mission.expected_total.toFixed(1)}</>}
        </span>
        <span>Candidates <b>{openCandidates}</b></span>
        <span>Covered <b>{pct(mission?.covered_fraction)}</b></span>
        <span>Remaining <b>{pct(mission?.remaining_fraction)}</b></span>
        <div className="toggles">
          <button className={view === "live" ? "active" : ""} onClick={() => setView("live")} title="Live mission">
            Live
          </button>
          <button className={view === "replay" ? "active" : ""} onClick={() => setView("replay")} title="Replay a recorded mission">
            Replay
          </button>
          {!config?.map_image && (
            <select value={base} onChange={(e) => setBase(e.target.value as BaseLayer)}>
              <option value="esri">Satellite</option>
              <option value="osm">OSM</option>
            </select>
          )}
          <label><input type="checkbox" checked={showBelief} onChange={(e) => setShowBelief(e.target.checked)} /> Belief</label>
          <label><input type="checkbox" checked={showCoverage} onChange={(e) => setShowCoverage(e.target.checked)} /> Coverage</label>
          {state?.truth && (
            <label><input type="checkbox" checked={showTruth} onChange={(e) => setShowTruth(e.target.checked)} /> Truth</label>
          )}
        </div>
      </header>

      <div className="screen">
        <div className="cell cell-3d">
          {isReplay ? (
            <RecordedVideo
              video={rec?.videos?.follow}
              t={t} playing={playing} speed={speed}
              title="3D follow (recorded)"
              placeholder={rec ? "The follow view was not recorded for this mission" : "Pick a recorded mission"}
            />
          ) : (
          <VideoPanel
            src="video/follow.mjpg"
            title={`3D follow - ${camName}`}
            streamKey={`follow-${camDrone}`}
            placeholder={hasBridge ? "Waiting for the sim bridge stream" : "3D view needs a sim bridge (--bridge)"}
          />
          )}
        </div>

        <div className="rightcol">
          <div className="cell cell-cam">
            {isReplay ? (
              <RecordedVideo
                video={rec?.videos?.[`cam${camDrone}`]}
                t={t} playing={playing} speed={speed}
                title={`Drone camera - ${camName} (recorded)`}
                placeholder="The drone camera was not recorded for this mission"
              />
            ) : (
            <VideoPanel
              src={`video/camera/${camDrone}.mjpg`}
              title={`Drone camera - ${camName}`}
              streamKey={camDrone}
              placeholder={hasBridge ? "Waiting for the drone camera" : "Drone camera needs a sim bridge (--bridge)"}
            />
            )}
            {state && state.drones.length > 1 && (
              <div className="view-toggle cam-tabs">
                {state.drones.map((d) => (
                  <button key={d.id} className={d.id === camDrone ? "active" : ""} onClick={() => selectDrone(d.id)}>
                    {d.name}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="cell cell-map">
            {config ? (
              <MapView
                config={config}
                state={state}
                belief={belief}
                prior={prior}
                base={base}
                showBelief={showBelief && mapMode === "map"}
                showCoverage={showCoverage && mapMode === "map"}
                overview={mapMode === "overview"}
                showTruth={showTruth}
                showPrior={showPrior}
                drawLocked={running || isReplay}
                fitTo={isReplay ? rec?.meta.polygon : null}
                picking={pickingFor != null}
                selectedDrone={selectedDrone}
                regionWeight={regionWeight}
                drawKind={drawTool}
                onToolChange={setDrawTool}
                onPick={onPick}
                onAreaChange={onAreaChange}
                onSelectDrone={selectDrone}
                onSelectSurvivor={selectSurvivor}
              />
            ) : (
              <div className="panel muted">Loading /api/config ...</div>
            )}
            {config?.map_image && !isReplay && (
              <div className="view-toggle map-toggle">
                <button className={mapMode === "map" ? "active" : ""} onClick={() => setMapMode("map")}>Map</button>
                <button className={mapMode === "overview" ? "active" : ""} onClick={() => setMapMode("overview")}
                  title="Live top-down view of the scene">Overview</button>
              </div>
            )}
            <PriorPanel prior={prior} show={showPrior} onToggle={togglePrior} />
          </div>
        </div>

        <div className="bottomrow">
          <div className="cell cell-drones">
            {config ? (
              <>
                {isReplay ? (
                  <ReplayControls
                    list={replay.list} id={replayId} onPick={pickReplay} t={t} duration={duration}
                    playing={playing} speed={speed} onPlay={setPlaying} onSpeed={setSpeed}
                    onSeek={(v) => { setPlaying(false); setT(v); }} error={replay.error}
                  />
                ) : (
                  <MissionPanel config={config} state={state} polygon={polygon} lkp={lkp} regions={regions} send={send} />
                )}
                <DroneList
                  drones={state?.drones ?? []}
                  survivors={state?.survivors ?? []}
                  pickingFor={pickingFor}
                  selected={selectedDrone}
                  onSelect={selectDrone}
                  onPickStandby={setPickingFor}
                  send={send}
                  readOnly={isReplay}
                />
              </>
            ) : (
              <div className="panel muted">Loading /api/config ...</div>
            )}
            {!isReplay && (
            <div className="region-weight">
              Region weight
              <input
                type="range" min={0} max={1} step={0.05} value={regionWeight}
                onChange={(e) => setRegionWeight(Number(e.target.value))}
                disabled={running}
              />
              <b>{regionWeight.toFixed(2)}</b>
            </div>
            )}
          </div>
          <div className="cell cell-survivors">
            <SurvivorList
              survivors={state?.survivors ?? []}
              drones={state?.drones ?? []}
              selected={selectedSurvivor}
              onSelect={selectSurvivor}
              send={send}
              readOnly={isReplay}
            />
            <EventLog events={events} />
          </div>
        </div>
      </div>
    </div>
  );
}