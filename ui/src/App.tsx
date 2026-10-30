import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMission } from "./useMission";
import { useReplay } from "./useReplay";
import MapView, { type BaseLayer, type DrawTool } from "./MapView";
import MissionPanel from "./MissionPanel";
import DroneList from "./DroneList";
import SurvivorList from "./SurvivorList";
import EventLog from "./EventLog";
import VideoPanel from "./VideoPanel";
import OverlaysPanel from "./OverlaysPanel";
import { SettingsMenu, useMissionSettings } from "./MissionSettings";
import RecordedVideo from "./RecordedVideo";
import ReplayControls from "./ReplayControls";
import OperatorLogin from "./OperatorLogin";
import { FOLLOW_DEFAULT, useCameraViews } from "./useCameraViews";
import { replayBeliefAt, replayEventsAt, replayPrior, replayStateAt } from "./replayState";
import type { LonLat, Region } from "./types";
import { fmtTime } from "./types";

const DEFAULT_REGION_WEIGHT = 0.3;
const TRUTH_KEY = "sar.showTruth";

type Pane = { kind: "follow" | "camera"; drone: number | null };

function loadFlag(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === "1";
  } catch {
    return false;
  }
}

function saveFlag(key: string, v: boolean) {
  try {
    window.localStorage.setItem(key, v ? "1" : "0");
  } catch {
    // storage blocked: the setting lasts until the page reloads
  }
}

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
  const [showTruth, setShowTruthState] = useState(() => loadFlag(TRUTH_KEY));
  const setShowTruth = useCallback((v: boolean) => {
    setShowTruthState(v);
    saveFlag(TRUTH_KEY, v);
  }, []);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Each of the two left panes shows one drone's follow view or camera.
  // Both start on the 3D follow view. A pane drone of null shows the pane's own drone:
  // the first drone on top, the second below.
  const [panes, setPanes] = useState<Pane[]>([{ kind: "follow", drone: null }, { kind: "follow", drone: null }]);
  const setPane = useCallback((i: number, p: Partial<Pane>) => {
    setPanes((prev) => prev.map((q, j) => (j === i ? { ...q, ...p } : q)));
  }, []);
  const [showPrior, setShowPrior] = useState<Set<string>>(new Set());
  const [selectedDrone, setSelectedDrone] = useState<number | null>(null);
  const [selectedSurvivor, setSelectedSurvivor] = useState<number | null>(null);

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
  const cv = useCameraViews(send);
  const ms = useMissionSettings(config, live.state, live.send);
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
  const survivors = state?.survivors ?? [];
  const openCandidates = survivors.filter((s) => s.status === "candidate").length;
  const found = survivors.filter((s) => s.status === "confirmed" || s.status === "assigned" || s.status === "reached").length;
  const rescued = survivors.filter((s) => s.status === "reached").length;

  const droneName = (k: number) => state?.drones.find((d) => d.id === k)?.name ?? `D${k}`;
  const followPane = (k: number) => {
    const name = droneName(k);
    const view = cv.follows[k] ?? FOLLOW_DEFAULT;
    return isReplay ? (
      <RecordedVideo
        video={rec?.videos?.[k === 0 ? "follow" : `follow${k}`]}
        t={t} playing={playing} speed={speed}
        title={`3D follow - ${name} (recorded)`}
        placeholder={rec ? "The follow view was not recorded for this drone" : "Pick a recorded mission"}
      />
    ) : (
      <VideoPanel
        src={`video/follow/${k}.mjpg`}
        h264Src={`video/h264/follow/${k}`}
        title={`3D follow - ${name}`}
        streamKey={`follow-${k}`}
        placeholder={hasBridge ? "Waiting for the sim bridge stream" : "3D view needs a sim bridge (--bridge)"}
        onZoom={hasBridge ? (s) => cv.followZoom(k, s) : undefined}
        onDrag={hasBridge ? (dx, dy) => cv.followDrag(k, dx, dy) : undefined}
        onReset={hasBridge ? () => cv.followReset(k) : undefined}
        hint={`${view.distance.toFixed(0)} m, ${view.pitch.toFixed(0)}° up`}
      />
    );
  };
  const cameraPane = (k: number) => {
    const name = droneName(k);
    const view = cv.cams[k];
    return isReplay ? (
      <RecordedVideo
        video={rec?.videos?.[`cam${k}`]}
        t={t} playing={playing} speed={speed}
        title={`Drone camera - ${name} (recorded)`}
        placeholder="The drone camera was not recorded for this mission"
      />
    ) : (
      <VideoPanel
        src={`video/camera/${k}.mjpg`}
        h264Src={`video/h264/camera/${k}`}
        title={`Drone camera - ${name}`}
        streamKey={k}
        placeholder={hasBridge ? "Waiting for the drone camera" : "Drone camera needs a sim bridge (--bridge)"}
        onZoom={hasBridge ? (s) => cv.camZoom(k, s) : undefined}
        onDrag={hasBridge ? (dx, dy) => cv.camDrag(k, dx, dy) : undefined}
        onReset={hasBridge ? () => cv.camReset(k) : undefined}
        hint={view ? `fov ${view.fov.toFixed(0)}°, tilt ${view.tilt.toFixed(0)}°` : "fov 60°, straight down"}
      />
    );
  };
  const paneCell = (i: number, area: string) => {
    const pane = panes[i];
    const drone = pane.drone ?? state?.drones[i]?.id ?? camDrone;
    return (
      <div className={`cell ${area}`}>
        {pane.kind === "follow" ? followPane(drone) : cameraPane(drone)}
        <div className="view-toggle">
          <button className={pane.kind === "follow" ? "active" : ""} onClick={() => setPane(i, { kind: "follow" })}
            title="3D follow view of one drone">3D follow</button>
          <button className={pane.kind === "camera" ? "active" : ""} onClick={() => setPane(i, { kind: "camera" })}
            title="Camera of one drone">Drone camera</button>
        </div>
        {state && state.drones.length > 1 && (
          <div className="view-toggle cam-tabs">
            {state.drones.map((d) => (
              <button key={d.id} className={d.id === drone ? "active" : ""} onClick={() => setPane(i, { drone: d.id })}>
                {d.name}
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

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
        <span>Time <b>{fmtTime(state?.t)}</b></span>
        <span>Survivors found <b>{state ? found : "-"}</b></span>
        <span>Rescued <b>{state ? rescued : "-"}</b></span>
        <span>Candidates <b>{state ? openCandidates : "-"}</b></span>
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
          <div className="settings">
            <button className={settingsOpen ? "active" : ""} onClick={() => setSettingsOpen((v) => !v)}>Settings</button>
            {settingsOpen && (
              <SettingsMenu config={config} state={state} ms={ms} send={send} readOnly={isReplay}
                showTruth={showTruth} onShowTruth={setShowTruth} />
            )}
          </div>
          {!isReplay && config?.login_required && <OperatorLogin live={live} />}
        </div>
      </header>

      <div className="screen">
        {paneCell(0, "cell-3d")}
        {paneCell(1, "cell-cam")}
        <div className="cell cell-map">
          {config ? (
            <MapView
              config={config}
              state={state}
              belief={belief}
              prior={prior}
              base={base}
              showBelief={showBelief}
              showCoverage={showCoverage}
              showTruth={showTruth}
              showPrior={showPrior}
              drawLocked={running || isReplay}
              needArea={!polygon}
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
          <OverlaysPanel prior={prior} show={showPrior} onToggle={togglePrior}
            showBelief={showBelief} onBelief={setShowBelief} showCoverage={showCoverage} onCoverage={setShowCoverage} />
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
                  <MissionPanel config={config} state={state} polygon={polygon} lkp={lkp} regions={regions} ms={ms} send={send} />
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