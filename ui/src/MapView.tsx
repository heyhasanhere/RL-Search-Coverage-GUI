import { useEffect, useRef, useState } from "react";
import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, ImageSource, MapMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
// MapLibre 6 loads its worker relative to its own module URL, which Vite's bundling breaks.
// Bundle the worker as its own ES module chunk and point MapLibre at it.
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import {
  TerraDraw,
  TerraDrawPointMode,
  TerraDrawPolygonMode,
  TerraDrawSelectMode,
} from "terra-draw";
import { TerraDrawMapLibreGLAdapter } from "terra-draw-maplibre-gl-adapter";
import type { BeliefMsg, Config, Drone, LonLat, PriorMsg, Region, StateMsg, Survivor } from "./types";
import { DRONE_COLORS, LEVEL_COLORS, LEVEL_SHORT, SURVIVOR_COLORS, droneColor } from "./types";

maplibregl.setWorkerUrl(workerUrl);

export type BaseLayer = "esri" | "osm";
export type DrawTool = "select" | "polygon" | "point" | "region";

interface Props {
  config: Config;
  state: StateMsg | null;
  belief: BeliefMsg | null;
  prior: PriorMsg | null;
  base: BaseLayer;
  showBelief: boolean;
  showCoverage: boolean;
  showTruth: boolean;
  /** Map background: the static scene image (false) or the live top-down frame (true). */
  overview?: boolean;
  /** Keys of the prior layers to overlay; "blend" is the combined map. */
  showPrior: Set<string>;
  drawLocked: boolean;
  picking: boolean;
  selectedDrone: number | null;
  /** Weight applied to newly drawn operator regions (default 0.3, plan 6.3). */
  regionWeight: number;
  /** The tool the operator has selected; decides what a new polygon means. */
  drawKind: DrawTool;
  /** Called when a shape finishes so the toolbar can fall back to Edit. */
  onToolChange: (tool: DrawTool) => void;
  onPick: (p: LonLat) => void;
  onAreaChange: (polygon: LonLat[] | null, lkp: LonLat | null, regions: Region[]) => void;
  onSelectDrone: (drone: number) => void;
  onSelectSurvivor: (sid: number) => void;
  /** Replay: zoom to this polygon whenever it changes. */
  fitTo?: LonLat[] | null;
}

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

function fc(features: GeoJSON.Feature[]): GeoJSON.FeatureCollection {
  return { type: "FeatureCollection", features };
}

function line(coords: LonLat[], props: Record<string, unknown>): GeoJSON.Feature {
  return { type: "Feature", properties: props, geometry: { type: "LineString", coordinates: coords } };
}

function droneElement(): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "drone-marker";
  el.innerHTML = '<div class="drone-dot"><span class="drone-label"></span><i class="drone-head"></i></div><span class="drone-level"></span>';
  return el;
}

const M_PER_DEG = 111320;

/** Camera footprint on the ground: a rectangle around the drone, long side across the track. */
function footprint(d: Drone): GeoJSON.Feature | null {
  if (!d.footprint_m) return null;
  const [along, cross] = d.footprint_m;
  const h = (d.heading * Math.PI) / 180;
  const fe = Math.sin(h), fn = Math.cos(h);          // forward, east and north parts
  const kLon = 1 / (M_PER_DEG * Math.cos((d.pos[1] * Math.PI) / 180));
  const pt = (a: number, c: number): LonLat => [
    d.pos[0] + (a * fe + c * fn) * kLon,
    d.pos[1] + (a * fn - c * fe) / M_PER_DEG,
  ];
  const ring = [pt(along, -cross), pt(along, cross), pt(-along, cross), pt(-along, -cross), pt(along, -cross)];
  return { type: "Feature", properties: { color: droneColor(d.id) }, geometry: { type: "Polygon", coordinates: [ring] } };
}

function survivorElement(): HTMLDivElement {
  const el = document.createElement("div");
  el.className = "survivor-marker";
  return el;
}

/** Keeps one DOM marker per id, creating and removing as needed. */
function syncMarkers<T extends { id: number; pos: LonLat }>(
  map: maplibregl.Map,
  store: Map<number, maplibregl.Marker>,
  items: T[],
  create: () => HTMLDivElement,
  update: (el: HTMLElement, item: T) => void,
) {
  const seen = new Set<number>();
  for (const item of items) {
    seen.add(item.id);
    let m = store.get(item.id);
    if (!m) {
      m = new maplibregl.Marker({ element: create() }).setLngLat(item.pos).addTo(map);
      store.set(item.id, m);
    }
    m.setLngLat(item.pos);
    update(m.getElement(), item);
  }
  for (const [id, m] of store) {
    if (!seen.has(id)) {
      m.remove();
      store.delete(id);
    }
  }
}

export default function MapView(props: Props) {
  const { config, state, belief, prior, base, showBelief, showCoverage, showTruth, showPrior, drawLocked, picking } = props;
  const overview = props.overview ?? false;
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const drawRef = useRef<TerraDraw | null>(null);
  const [ready, setReady] = useState(false);
  const tool = props.drawKind;
  const droneMarkers = useRef(new Map<number, maplibregl.Marker>());
  const survivorMarkers = useRef(new Map<number, maplibregl.Marker>());
  const truthMarkers = useRef(new Map<number, maplibregl.Marker>());
  const cb = useRef(props);
  cb.current = props;

  // Map and Terra Draw setup (once).
  useEffect(() => {
    const map = new maplibregl.Map({
      container: container.current!,
      center: config.home,
      zoom: 14,
      attributionControl: { compact: true },
      style: {
        version: 8,
        sources: {
          esri: {
            type: "raster",
            tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            tileSize: 256,
            maxzoom: 19,
            attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community",
          },
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            maxzoom: 19,
            attribution: "&copy; OpenStreetMap contributors",
          },
        },
        // A Unity scene backend brings its own map image: no real-world tiles under it.
        layers: config.map_image
          ? [{ id: "bg", type: "background", paint: { "background-color": "#0b1220" } }]
          : [
              { id: "esri", type: "raster", source: "esri" },
              { id: "osm", type: "raster", source: "osm", layout: { visibility: "none" } },
            ],
      },
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    mapRef.current = map;
    if (import.meta.env.DEV) (window as unknown as { __map: unknown }).__map = map;

    map.on("load", () => {
      if (config.map_image) {
        const c = config.map_image.coordinates;
        map.addSource("scene", { type: "image", url: config.map_image.url, coordinates: [c[0], c[1], c[2], c[3]] });
        map.addLayer({ id: "scene", type: "raster", source: "scene" });
      }
      map.addSource("area", { type: "geojson", data: EMPTY });
      map.addSource("segments", { type: "geojson", data: EMPTY });
      map.addSource("targets", { type: "geojson", data: EMPTY });
      map.addSource("standby", { type: "geojson", data: EMPTY });
      map.addSource("fov", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "fov-fill", type: "fill", source: "fov", paint: { "fill-color": ["get", "color"], "fill-opacity": 0.15 } });
      map.addLayer({ id: "fov-line", type: "line", source: "fov", paint: { "line-color": ["get", "color"], "line-width": 1.5, "line-opacity": 0.9 } });
      map.addLayer({ id: "area-line", type: "line", source: "area", paint: { "line-color": "#f8fafc", "line-width": 2, "line-opacity": 0.8 } });
      map.addLayer({ id: "segments", type: "line", source: "segments", paint: { "line-color": ["get", "color"], "line-width": 3, "line-opacity": 0.9 } });
      map.addLayer({ id: "targets", type: "line", source: "targets", paint: { "line-color": ["get", "color"], "line-width": 2, "line-dasharray": [2, 2] } });
      map.addLayer({
        id: "standby", type: "circle", source: "standby",
        paint: { "circle-radius": 6, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": "#22d3ee", "circle-stroke-width": 2 },
      });

      const draw = new TerraDraw({
        adapter: new TerraDrawMapLibreGLAdapter({ map }),
        modes: [
          new TerraDrawPolygonMode({
            styles: { fillColor: "#38bdf8", fillOpacity: 0.12, outlineColor: "#38bdf8", outlineWidth: 2 },
          }),
          new TerraDrawPointMode({ styles: { pointColor: "#f43f5e", pointWidth: 7, pointOutlineColor: "#ffffff", pointOutlineWidth: 2 } }),
          new TerraDrawSelectMode({
            flags: {
              polygon: { feature: { draggable: true, coordinates: { midpoints: true, draggable: true, deletable: true } } },
              point: { feature: { draggable: true } },
            },
          }),
        ],
      });
      draw.start();
      draw.setMode("select");
      drawRef.current = draw;
      // TerraDraw has one polygon mode, so the tool chosen when drawing starts tags the
      // finished feature with a `kind` property; the report below splits on it.
      const currentKind = () => cb.current.drawKind;

      const report = () => {
        const snap = draw.getSnapshot();
        const polys = snap.filter((f) => f.geometry.type === "Polygon");
        const area = polys.find((f) => (f.properties as { kind?: string }).kind !== "region");
        const regions = polys.filter((f) => (f.properties as { kind?: string }).kind === "region");
        const pt = snap.find((f) => f.geometry.type === "Point");
        cb.current.onAreaChange(
          area ? ((area.geometry as GeoJSON.Polygon).coordinates[0] as LonLat[]) : null,
          pt ? ((pt.geometry as GeoJSON.Point).coordinates as LonLat) : null,
          regions.map((f) => ({
            polygon: (f.geometry as GeoJSON.Polygon).coordinates[0] as LonLat[],
            weight: cb.current.regionWeight,
          })),
        );
      };
      draw.on("finish", (id) => {
        const f = draw.getSnapshotFeature(id);
        if (!f) return;
        const kind = f.geometry.type === "Polygon" ? currentKind() : "lkp";
        draw.updateFeatureProperties(id, { kind });
        if (kind !== "region") {
          // One search area and one last-seen point: a new one replaces the old.
          const old = draw.getSnapshot().filter((o) => o.id !== id && o.geometry.type === f.geometry.type);
          if (old.length) draw.removeFeatures(old.map((o) => o.id!));
          cb.current.onToolChange("select");
        }
        report();
      });
      draw.on("change", (_ids, type) => {
        if (type === "update" || type === "delete") report();
      });
      if (config.area) {
        draw.addFeatures([{
          type: "Feature",
          geometry: { type: "Polygon", coordinates: [config.area] },
          properties: { mode: "polygon", kind: "area" },
        }]);
        report();
        const lons = config.area.map((p) => p[0]);
        const lats = config.area.map((p) => p[1]);
        map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: 30, duration: 0 });
      }
      setReady(true);
    });

    map.on("click", (e: MapMouseEvent) => {
      if (cb.current.picking) cb.current.onPick([e.lngLat.lng, e.lngLat.lat]);
    });

    return () => {
      drawRef.current?.stop();
      drawRef.current = null;
      map.remove();
      mapRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Draw tool / lock / pick mode.
  useEffect(() => {
    const draw = drawRef.current;
    if (!ready || !draw) return;
    // TerraDraw only knows "polygon" and "point"; "region" draws a polygon too.
    const mode = drawLocked || picking ? "static" : tool === "region" ? "polygon" : tool;
    draw.setMode(mode);
    mapRef.current!.getCanvas().style.cursor = picking ? "crosshair" : "";
    // While drawing or picking, markers must not swallow the click meant for Terra Draw.
    const interactive = mode === "select" ? "auto" : "none";
    for (const el of document.querySelectorAll(".drone-marker, .survivor-marker, .truth-marker")) {
      (el as HTMLElement).style.pointerEvents = interactive;
    }
  }, [ready, tool, drawLocked, picking]);

  // Base layer toggle.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || config.map_image) return;
    map.setLayoutProperty("esri", "visibility", base === "esri" ? "visible" : "none");
    map.setLayoutProperty("osm", "visibility", base === "osm" ? "visible" : "none");
  }, [ready, base]);

  // Overview: the scene image follows the live top-down frame, about once a second.
  useEffect(() => {
    const map = mapRef.current;
    const img = config.map_image;
    if (!ready || !map || !img) return;
    const src = map.getSource("scene") as ImageSource | undefined;
    if (!src) return;
    const coordinates = img.coordinates as [LonLat, LonLat, LonLat, LonLat];
    if (!overview || !img.live_url) {
      src.updateImage({ url: img.url, coordinates });
      return;
    }
    const live = img.live_url;
    const tick = () => src.updateImage({ url: `${live}?t=${Date.now()}`, coordinates });
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [ready, overview, config.map_image]);

  // Belief and coverage image sources.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !belief) return;
    const imgs: [string, string, number][] = [
      ["belief", belief.png, 0.65],
      ["coverage", belief.coverage_png, 0.45],
    ];
    for (const [id, png, opacity] of imgs) {
      if (!png) continue;
      const url = `data:image/png;base64,${png}`;
      const src = map.getSource(id) as ImageSource | undefined;
      if (src) src.updateImage({ url, coordinates: belief.corners });
      else {
        map.addSource(id, { type: "image", url, coordinates: belief.corners });
        map.addLayer(
          { id, type: "raster", source: id, paint: { "raster-opacity": opacity, "raster-fade-duration": 0, "raster-resampling": "nearest" } },
          "area-line",
        );
      }
    }
  }, [ready, belief]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    if (map.getLayer("belief")) map.setLayoutProperty("belief", "visibility", showBelief ? "visible" : "none");
    if (map.getLayer("coverage")) map.setLayoutProperty("coverage", "visibility", showCoverage ? "visible" : "none");
  }, [ready, belief, showBelief, showCoverage]);

  // Prior layers: one image source per component plus the blend (plan 6.2, PROTOCOL 5.3).
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !prior) return;
    const entries: [string, string, number][] = [
      ...prior.layers.map((l): [string, string, number] => [`prior-${l.key}`, l.png, 0.7]),
      ["prior-blend", prior.blend.png, 0.6],
    ];
    for (const [id, png, opacity] of entries) {
      if (!png) continue;
      const url = `data:image/png;base64,${png}`;
      const src = map.getSource(id) as ImageSource | undefined;
      if (src) src.updateImage({ url, coordinates: prior.corners });
      else {
        map.addSource(id, { type: "image", url, coordinates: prior.corners });
        map.addLayer(
          { id, type: "raster", source: id, paint: { "raster-opacity": opacity, "raster-fade-duration": 0 } },
          "area-line",
        );
      }
    }
  }, [ready, prior]);

  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !prior) return;
    const all = ["blend", ...prior.layers.map((l) => l.key)];
    for (const key of all) {
      const layer = map.getLayer(`prior-${key}`);
      if (layer) map.setLayoutProperty(`prior-${key}`, "visibility", showPrior.has(key) ? "visible" : "none");
    }
  }, [ready, prior, showPrior]);

  // Replay: the recorded mission may be far from the configured home.
  const fitKey = props.fitTo && props.fitTo.length > 2 ? JSON.stringify(props.fitTo) : "";
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map || !fitKey) return;
    const pts: LonLat[] = JSON.parse(fitKey);
    const lons = pts.map((p) => p[0]);
    const lats = pts.map((p) => p[1]);
    map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: 30, duration: 0 });
  }, [ready, fitKey]);

  // Live vector layers and markers.
  useEffect(() => {
    const map = mapRef.current;
    if (!ready || !map) return;
    const drones: Drone[] = state?.drones ?? [];
    const survivors: Survivor[] = state?.survivors ?? [];
    const poly = state?.mission.polygon ?? [];
    (map.getSource("area") as GeoJSONSource).setData(poly.length > 2 ? fc([line([...poly, poly[0]], {})]) : EMPTY);
    (map.getSource("segments") as GeoJSONSource).setData(
      fc(drones.filter((d) => d.segment.length > 1).map((d) => line(d.segment, { color: DRONE_COLORS[d.state] }))),
    );
    (map.getSource("targets") as GeoJSONSource).setData(
      fc(drones.filter((d) => d.target).map((d) => line([d.pos, d.target!], { color: DRONE_COLORS[d.state] }))),
    );
    (map.getSource("fov") as GeoJSONSource).setData(
      fc(drones.map(footprint).filter((f): f is GeoJSON.Feature => f != null)),
    );
    (map.getSource("standby") as GeoJSONSource).setData(
      fc(
        drones
          .filter((d) => d.standby_point)
          .map((d) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: d.standby_point! } })),
      ),
    );

    syncMarkers(map, droneMarkers.current, drones, droneElement, (el, d) => {
      const dot = el.querySelector<HTMLElement>(".drone-dot")!;
      dot.style.background = droneColor(d.id);
      dot.style.borderColor = DRONE_COLORS[d.state];
      el.querySelector<HTMLElement>(".drone-head")!.style.transform = `rotate(${d.heading - map.getBearing()}deg)`;
      const lvl = d.level ?? null;
      const lv = el.querySelector<HTMLElement>(".drone-level")!;
      lv.textContent = lvl ? LEVEL_SHORT[lvl] : "";
      lv.style.background = lvl ? LEVEL_COLORS[lvl] : "transparent";
      el.classList.toggle("selected", d.id === props.selectedDrone);
      el.querySelector(".drone-label")!.textContent = d.name;
      const agl = d.alt_agl ?? d.alt;
      el.title = `${d.name} ${d.state} level ${lvl ?? "-"} alt ${agl.toFixed(0)} m AGL`;
      el.onclick = (ev) => {
        ev.stopPropagation();
        props.onSelectDrone(d.id);
      };
    });
    syncMarkers(map, survivorMarkers.current, survivors, survivorElement, (el, s) => {
      el.style.background = SURVIVOR_COLORS[s.status];
      el.title = `#${s.id} ${s.status} score ${s.score.toFixed(2)}`;
      el.onclick = (ev) => {
        ev.stopPropagation();
        props.onSelectSurvivor(s.id);
      };
    });
    const truth = showTruth && state?.truth ? state.truth.map((pos, id) => ({ id, pos })) : [];
    syncMarkers(map, truthMarkers.current, truth, () => {
      const el = document.createElement("div");
      el.className = "truth-marker";
      el.title = "ground truth";
      return el;
    }, () => {});
  }, [ready, state, showTruth]);

  const tools: [DrawTool, string][] = [
    ["select", "Edit"],
    ["polygon", "Area"],
    ["point", "LKP"],
    ["region", "Region"],
  ];
  return (
    <div className="map-wrap">
      <div ref={container} className="map" />
      <div className="draw-bar">
        {tools.map(([t, label]) => (
          <button key={t} className={tool === t ? "active" : ""} disabled={drawLocked} onClick={() => props.onToolChange(t)}>
            {label}
          </button>
        ))}
        <button
          disabled={drawLocked}
          onClick={() => {
            drawRef.current?.clear();
            props.onAreaChange(null, null, []);
          }}
        >
          Clear
        </button>
      </div>
      {picking && <div className="pick-hint">Click the map to place the standby point</div>}
    </div>
  );
}
