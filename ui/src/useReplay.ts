import { useCallback, useEffect, useState } from "react";
import type { Replay, ReplayInfo } from "./types";

/**
 * Loads the mission log list (`GET /api/replays`) and, on demand, one log
 * (`GET /api/replay/<id>`).
 */
export function useReplay() {
  const [list, setList] = useState<ReplayInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [replays, setReplays] = useState<Record<string, Replay | undefined>>({});

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("api/replays");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const js = await r.json();
      setList(js.replays ?? []);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Load one replay and keep it in the cache so switching panes is instant.
  const load = useCallback(async (id: string) => {
    setLoading(true);
    try {
      const r = await fetch(`api/replay/${encodeURIComponent(id)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const js: Replay = await r.json();
      setReplays((prev) => ({ ...prev, [id]: js }));
      setError(null);
      return js;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      setLoading(false);
    }
  }, []);

  return { list, replays, load, loading, error, refresh };
}
