// Tiny mock of the Python orchestrator (docs/PROTOCOL.md) for UI development.
// Usage: node mock_server.mjs   (listens on :8000; serves ./dist at / when built)
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";

const PORT = Number(process.env.PORT ?? 8000);
const HOME = [149.58, -33.42];
const G = 48; // mock belief grid size
const M_PER_DEG_LAT = 111320;
const mPerDegLon = M_PER_DEG_LAT * Math.cos((HOME[1] * Math.PI) / 180);

const config = {
  home: HOME, cell_m: 30.0, max_grid: 96, max_area_km2: 8.29,
  modes: ["rl", "rl_search", "mcts", "planner", "greedy", "lawnmower"],
  drones: [
    { id: 0, name: "S1", role: "search" },
    { id: 1, name: "S2", role: "search" },
    { id: 2, name: "R1", role: "rescue" },
  ],
  backend: "kinematic", detector: "sim",
  bridge: true, scenarios: ["wilderness", "flood"],
};

// ---------- PNG encoding ----------
function png(w, h, rgba) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) rgba.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4); crc.writeUInt32BE(zlib.crc32(td));
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0)),
  ]);
}

function snapshotPng(id) {
  const w = 64, buf = Buffer.alloc(w * w * 4);
  for (let i = 0; i < w * w; i++) {
    const x = i % w, y = Math.floor(i / w);
    const d = Math.hypot(x - 32, y - 32);
    buf.set(d < 10 ? [220, 40 + id * 40, 40, 255] : [60 + (x ^ y) % 30, 90, 50, 255], i * 4);
  }
  return png(w, w, buf);
}

// ---------- geometry ----------
function inside([x, y], ring) {
  let c = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
const distM = (a, b) => Math.hypot((a[0] - b[0]) * mPerDegLon, (a[1] - b[1]) * M_PER_DEG_LAT);
const headingDeg = (a, b) =>
  (Math.atan2((b[0] - a[0]) * mPerDegLon, (b[1] - a[1]) * M_PER_DEG_LAT) * 180) / Math.PI;

// ---------- simulation ----------
let t = 0, speed = 1;
let options = { auto_dispatch: true, auto_replace: false, auto_stage: false };
let mission = blankMission();
let drones = [], survivors = [], truth = [], grid = null, bounds = null, mask = null;
let view = 0;   // drone the follow camera tracks

function blankMission() {
  return { status: "idle", mode: "rl", polygon: [], lkp: null, expected_total: 0, remaining_fraction: 1, expected_found: 0, covered_fraction: 0 };
}
function resetDrones() {
  drones = config.drones.map((d) => ({
    ...d, state: "IDLE", pos: [...HOME], alt: 0, heading: 0, battery_s: 1500, battery_frac: 1,
    segment: [], target: null, standby_point: null, task: "on ground", path: [],
    level: null, alt_agl: 0, sortie: 1, swap_remaining: null,
  }));
}
resetDrones();

function lawnmower(i, n) {
  const [w, s, e, nn] = bounds;
  const pts = [], lanes = 8;
  for (let k = i; k < lanes; k += n) {
    const x = w + ((k + 0.5) / lanes) * (e - w);
    pts.push(...(k % 2 ? [[x, nn], [x, s]] : [[x, s], [x, nn]]));
  }
  return pts;
}

const clients = new Set();
function broadcast(obj) {
  const s = JSON.stringify(obj);
  for (const c of clients) if (c.readyState === 1) c.send(s);
}
const event = (level, text) => broadcast({ type: "event", t, level, text });

function startMission(msg) {
  const ring = msg.polygon;
  const lons = ring.map((p) => p[0]), lats = ring.map((p) => p[1]);
  bounds = [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
  const [w, s, e, n] = bounds;
  mask = new Uint8Array(G * G);
  grid = new Float32Array(G * G);
  for (let r = 0; r < G; r++)
    for (let c = 0; c < G; c++) {
      const p = [w + ((c + 0.5) / G) * (e - w), n - ((r + 0.5) / G) * (n - s)];
      if (!inside(p, ring)) continue;
      mask[r * G + c] = 1;
      grid[r * G + c] = msg.lkp ? Math.exp(-((distM(p, msg.lkp) / 400) ** 2)) + 0.2 : 1;
    }
  const tot = grid.reduce((a, b) => a + b, 0);
  for (let i = 0; i < grid.length; i++) grid[i] *= msg.expected_survivors / tot;
  mission = { ...blankMission(), status: "running", mode: msg.mode, polygon: ring, lkp: msg.lkp ?? null, expected_total: msg.expected_survivors };
  mission.scenario = msg.scenario ?? "wilderness";
  options = { ...options, ...msg.options };
  t = 0; survivors = []; truth = [];
  while (truth.length < 3) {
    const p = [w + Math.random() * (e - w), s + Math.random() * (n - s)];
    if (inside(p, ring)) truth.push(p);
  }
  resetDrones();
  buildPriorComponents();
  const pmsg = priorMsg();
  if (pmsg) broadcast(pmsg);
  drones.slice(0, Math.min(msg.n_search, 2)).forEach((d, i) => launchSearch(d, i, Math.min(msg.n_search, 2)));
  event("info", `Mission started, mode ${msg.mode}, ${msg.n_search} search drones`);
}

function launchSearch(d, i = d.id, n = 2) {
  d.state = "SEARCH";
  d.level = i % 2 ? "high" : "low";
  d.alt = d.level === "high" ? 70 : 25;
  d.alt_agl = d.alt;
  d.path = lawnmower(i % n, n);
  d.task = `search (${mission.mode})`;
}
function goTo(d, state, point, task) {
  d.state = state; d.path = [point]; d.target = point; d.task = task; d.alt = 90;
  d.alt_agl = 90; d.level = "transit";
}

function step(dt) {
  t += dt;
  for (const d of drones) {
    if (["IDLE", "LANDED", "STANDBY", "ON_SURVIVOR"].includes(d.state)) { d.segment = []; continue; }
    d.battery_s = Math.max(0, d.battery_s - dt); d.battery_frac = d.battery_s / 1500;
    let move = 12 * dt;
    while (move > 0 && d.path.length) {
      const nxt = d.path[0], dist = distM(d.pos, nxt);
      if (dist > 0.1) d.heading = (headingDeg(d.pos, nxt) + 360) % 360;
      if (dist <= move) { d.pos = [...nxt]; d.path.shift(); move -= dist; }
      else { const f = move / dist; d.pos = [d.pos[0] + (nxt[0] - d.pos[0]) * f, d.pos[1] + (nxt[1] - d.pos[1]) * f]; move = 0; }
    }
    d.segment = d.path.length ? [d.pos, ...d.path.slice(0, 1)] : [];
    if (!d.path.length) arrive(d);
    if (d.state === "SEARCH") observe(d);
  }
  for (const s of survivors) if (s.status === "candidate" && t - s.t_detected > 15 && options.auto_dispatch) confirm(s);
  if (mission.status === "running") {
    const tot = grid.reduce((a, b) => a + b, 0);
    mission.remaining_fraction = tot / mission.expected_total;
    mission.expected_found = mission.expected_total - tot;
    mission.covered_fraction = coverage.reduce((a, b) => a + b, 0) / mask.reduce((a, b) => a + b, 0);
  }
}

let coverage = new Uint8Array(G * G);
function observe(d) {
  const [w, s, e, n] = bounds;
  const c = Math.floor(((d.pos[0] - w) / (e - w)) * G), r = Math.floor(((n - d.pos[1]) / (n - s)) * G);
  for (let dr = -2; dr <= 2; dr++)
    for (let dc = -2; dc <= 2; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || cc < 0 || rr >= G || cc >= G || !mask[rr * G + cc]) continue;
      grid[rr * G + cc] *= 0.85; coverage[rr * G + cc] = 1;
    }
  truth.forEach((p, i) => {
    if (distM(d.pos, p) < 60 && !survivors.some((s) => s.id === i)) {
      survivors.push({ id: i, pos: p, status: "candidate", score: 0.6 + Math.random() * 0.35, views: 1, has_snapshot: true, assigned: null, t_detected: t, t_confirmed: null, t_reached: null });
      event("info", `${d.name} candidate #${i}`);
    }
  });
}

function confirm(s) {
  s.status = "confirmed"; s.t_confirmed = t; event("info", `Survivor #${s.id} confirmed`);
  const r = drones.find((d) => d.role === "rescue" && ["IDLE", "STANDBY"].includes(d.state));
  if (options.auto_dispatch && r) dispatch(r, s);
}
function dispatch(d, s) {
  s.status = "assigned"; s.assigned = d.id;
  goTo(d, "RESCUE_TRANSIT", s.pos, `rescue #${s.id}`);
  event("info", `${d.name} dispatched to #${s.id}`);
}

function arrive(d) {
  if (d.state === "SEARCH") { d.state = "RETURNING"; d.path = [HOME]; d.task = "search done, returning"; }
  else if (d.state === "RESCUE_TRANSIT") {
    const s = survivors.find((x) => x.assigned === d.id && x.status === "assigned");
    if (s) { s.status = "reached"; s.t_reached = t; event("info", `${d.name} reached #${s.id}`); }
    d.state = "ON_SURVIVOR"; d.target = null; d.task = "hovering on survivor";
  } else if (d.state === "RETURNING") { d.state = "LANDED"; d.alt = 0; d.target = null; d.task = "landed"; }
  else if (d.state === "ABORT") { d.state = "LANDED"; d.alt = 0; }
  if (d.state === "STANDBY_TRANSIT") { d.state = "STANDBY"; d.target = null; d.task = "parked"; }
}

function handle(msg) {
  const d = drones.find((x) => x.id === msg.drone);
  const s = survivors.find((x) => x.id === msg.survivor);
  switch (msg.type) {
    case "start_mission": coverage = new Uint8Array(G * G); return startMission(msg);
    case "stop_mission":
      mission.status = "finished";
      drones.forEach((x) => x.state !== "IDLE" && x.state !== "LANDED" && goTo(x, "RETURNING", HOME, "returning"));
      return event("warn", "Mission stopped");
    case "set_mode": mission.mode = msg.mode; return event("info", `Mode set to ${msg.mode}`);
    case "set_speed": speed = msg.speed; return;
    case "set_options": options = { ...options, ...msg }; delete options.type; return;
    case "recall": if (!d) break; goTo(d, "RETURNING", HOME, "returning"); return event("info", `${d.name} recalled`);
    case "standby": {
      if (!d) break;
      const p = msg.point ?? HOME;
      d.standby_point = p; goTo(d, "STANDBY_TRANSIT", p, "to standby");
      return event("info", `${d.name} to standby`);
    }
    case "launch":
      if (!d) break;
      if (mission.status !== "running") return event("error", "No mission running");
      if (msg.task === "search") { launchSearch(d); return event("info", `${d.name} launched to search`); }
      { const tgt = s ?? survivors.find((x) => x.status === "confirmed");
        if (!tgt) return event("error", "No survivor to rescue");
        return dispatch(d, tgt); }
    case "dispatch": if (!d || !s) break; return dispatch(d, s);
    case "confirm": if (!s) break; return confirm(s);
    case "reject": if (!s) break; s.status = "rejected"; return event("info", `Survivor #${s.id} rejected`);
    case "select": {
      // The Python server points the follow camera at this drone; mirror that.
      if (msg.drone != null) { view = msg.drone; return; }
      if (msg.survivor != null) {
        const sv = survivors.find((x) => x.id === msg.survivor);
        if (sv) view = drones.reduce((best, d) => (distM(d.pos, sv.pos) < distM(best.pos, sv.pos) ? d : best), drones[0]).id;
        return;
      }
      return;
    }
  }
  event("error", `Invalid command: ${JSON.stringify(msg)}`);
}

function stateMsg() {
  return {
    type: "state", t, speed, mission,
    drones: drones.map(({ path, ...d }) => ({ ...d, state: d.state === "STANDBY_TRANSIT" ? "RETURNING" : d.state })),
    survivors, options, truth: truth.length ? truth : null,
  };
}

// ---------- prior layers (PROTOCOL 5.3) ----------
let priorComponents = [];
function buildPriorComponents() {
  const [w, s, e, n] = bounds;
  const mk = (fn) => {
    const buf = Buffer.alloc(G * G * 4);
    for (let i = 0; i < G * G; i++) {
      if (!mask[i]) continue;
      fn(i, buf, 255);
    }
    return png(G, G, buf).toString("base64");
  };
  const lkp = mission.lkp ?? [w, s];
  priorComponents = [
    { key: "koester", label: "Koester", png: mk((i, b, a) => {
        const r = Math.floor(i / G), c = i % G;
        const p = [w + ((c + 0.5) / G) * (e - w), n - ((r + 0.5) / G) * (n - s)];
        const d = Math.min(1, distM(p, lkp) / 2000);
        b.set([Math.round(40 + 200 * (1 - d)), Math.round(60 + 150 * d), 60, 200], i * 4);
      }) },
    { key: "walker", label: "Walker", png: mk((i, b, a) => {
        const r = Math.floor(i / G), c = i % G;
        const p = [w + ((c + 0.5) / G) * (e - w), n - ((r + 0.5) / G) * (n - s)];
        const d = distM(p, lkp) / 1500;
        const v = Math.exp(-d * d);
        b.set([Math.round(60 + 180 * v), Math.round(200 * v + 40), 80, Math.round(80 + 140 * v)], i * 4);
      }) },
  ];
}

function priorMsg() {
  if (!bounds || !priorComponents.length) return null;
  const [w, s, e, n] = bounds;
  return {
    type: "prior",
    corners: [[w, n], [e, n], [e, s], [w, s]],
    nx: G, ny: G, cell_m: 30.0,
    scenario: mission.scenario ?? "wilderness",
    blend: { key: "blend", label: "Blend", png: beliefMsg().png },
    layers: priorComponents,
  };
}

function beliefMsg() {
  const [w, s, e, n] = bounds;
  const max = Math.max(...grid, 1e-9);
  const b = Buffer.alloc(G * G * 4), cov = Buffer.alloc(G * G * 4);
  for (let i = 0; i < G * G; i++) {
    if (!mask[i]) continue;
    const v = grid[i] / max;
    b.set([255, Math.round(200 * (1 - v)), 0, Math.round(40 + 170 * v)], i * 4);
    if (coverage[i]) cov.set([80, 200, 255, 140], i * 4);
  }
  return {
    type: "belief",
    corners: [[w, n], [e, n], [e, s], [w, s]],
    png: png(G, G, b).toString("base64"),
    coverage_png: png(G, G, cov).toString("base64"),
  };
}

setInterval(() => {
  step(0.2 * speed);
  broadcast(stateMsg());
}, 200);
setInterval(() => { if (mission.status === "running") broadcast(beliefMsg()); }, 1000);

// ---------- HTTP + WS ----------
const dist = path.join(path.dirname(fileURLToPath(import.meta.url)), "dist");

// Synthetic video frames. The mock has no renderer, so it sends a moving test pattern
// as MJPEG (PNG frames inside a multipart body), enough to exercise the video panels.
function videoFrame(kind, id, phase) {
  const w = 320, h = 180;
  const buf = Buffer.alloc(w * h * 4);
  const base = kind === "follow" ? [46, 62, 104] : kind === "overview" ? [38, 82, 68] : [96, 84, 62];
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const checker = ((x >> 4) + (y >> 4) + phase) % 2 ? 18 : 0;
      // a bright box that drifts, standing in for the detector / the tracked drone
      const cx = (w * 0.5 + Math.cos(phase / 20) * (w * 0.3)) | 0;
      const cy = (h * 0.5 + Math.sin(phase / 15) * (h * 0.3)) | 0;
      const inBox = Math.abs(x - cx) < 22 && Math.abs(y - cy) < 30;
      buf.set(
        inBox
          ? [250, 210, 60, 255]
          : [base[0] + checker, base[1] + checker, base[2] + checker, 255],
        i,
      );
    }
  return png(w, h, buf);
}

function serveMjpeg(res, kind, id) {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace; boundary=sarframe",
    "Cache-Control": "no-store",
  });
  let phase = id * 7;
  const timer = setInterval(() => {
    if (res.writableEnded) { clearInterval(timer); return; }
    const frame = videoFrame(kind, id, phase++);
    res.write(`--sarframe\r\nContent-Type: image/png\r\nContent-Length: ${frame.length}\r\n\r\n`);
    res.write(frame);
    res.write("\r\n");
  }, 200);
  res.on("close", () => clearInterval(timer));
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname === "/api/config") {
    res.writeHead(200, { "Content-Type": "application/json" });
    return res.end(JSON.stringify(config));
  }
  if (url.pathname === "/video/follow.mjpg") return serveMjpeg(res, "follow", view);
  if (url.pathname === "/video/overview.mjpg") return serveMjpeg(res, "overview", 0);
  const cm = url.pathname.match(/^\/video\/camera\/(\d+)\.mjpg$/);
  if (cm) return serveMjpeg(res, "camera", Number(cm[1]));
  const m = url.pathname.match(/^\/api\/snapshot\/(\d+)\.jpg$/);
  if (m) {
    if (!survivors.some((s) => s.id === Number(m[1]))) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "Content-Type": "image/png" }); // mock sends PNG bytes; browsers sniff it
    return res.end(snapshotPng(Number(m[1])));
  }
  const file = path.join(dist, url.pathname === "/" ? "index.html" : path.normalize(url.pathname));
  if (file.startsWith(dist) && fs.existsSync(file) && fs.statSync(file).isFile()) {
    const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
    res.writeHead(200, { "Content-Type": types[path.extname(file)] ?? "application/octet-stream" });
    return fs.createReadStream(file).pipe(res);
  }
  res.writeHead(404); res.end();
});
const wss = new WebSocketServer({ server, path: "/ws" });
wss.on("connection", (ws) => {
  clients.add(ws);
  ws.on("close", () => clients.delete(ws));
  ws.on("message", (data) => {
    try { handle(JSON.parse(data)); } catch (err) { event("error", `Bad message: ${err.message}`); }
  });
  ws.send(JSON.stringify({ type: "event", t, level: "info", text: "Connected to mock orchestrator" }));
});
server.listen(PORT, () => console.log(`mock orchestrator on http://localhost:${PORT}`));
