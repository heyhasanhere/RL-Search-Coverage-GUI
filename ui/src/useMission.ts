import { useCallback, useEffect, useRef, useState } from "react";
import type { BeliefMsg, ClientMsg, Config, EventMsg, PriorMsg, ServerMsg, StateMsg } from "./types";

const MAX_EVENTS = 100;

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

/** Loads /api/config (retrying) and keeps an auto-reconnecting WebSocket to the orchestrator. */
export function useMission() {
  const [config, setConfig] = useState<Config | null>(null);
  const [state, setState] = useState<StateMsg | null>(null);
  const [belief, setBelief] = useState<BeliefMsg | null>(null);
  const [prior, setPrior] = useState<PriorMsg | null>(null);
  const [events, setEvents] = useState<EventMsg[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const addEvent = useCallback((e: EventMsg) => {
    setEvents((prev) => [...prev, e].slice(-MAX_EVENTS));
  }, []);

  useEffect(() => {
    let stop = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const r = await fetch("api/config");
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        if (!stop) setConfig(await r.json());
      } catch {
        if (!stop) timer = window.setTimeout(load, 2000);
      }
    };
    load();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let stop = false;
    let retry = 1000;
    let timer: number | undefined;
    const connect = () => {
      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        retry = 1000;
        setConnected(true);
      };
      ws.onmessage = (ev) => {
        let msg: ServerMsg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === "state") {
          setState(msg);
          // no mission (before the first start or after a reset): drop the last mission's map layers
          if (msg.mission.status === "idle") {
            setBelief(null);
            setPrior(null);
          }
        }
        else if (msg.type === "belief") setBelief(msg);
        else if (msg.type === "prior") setPrior(msg);
        else if (msg.type === "event") addEvent(msg);
      };
      ws.onclose = () => {
        setConnected(false);
        if (wsRef.current === ws) wsRef.current = null;
        if (!stop) {
          timer = window.setTimeout(connect, retry);
          retry = Math.min(retry * 2, 5000);
        }
      };
      ws.onerror = () => ws.close();
    };
    connect();
    return () => {
      stop = true;
      window.clearTimeout(timer);
      wsRef.current?.close();
    };
  }, [addEvent]);

  const send = useCallback(
    (msg: ClientMsg) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
      else addEvent({ type: "event", t: 0, level: "error", text: `Not connected: ${msg.type} not sent` });
    },
    [addEvent],
  );

  return { config, state, belief, prior, events, connected, send, addEvent };
}
