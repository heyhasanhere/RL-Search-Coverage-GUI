import { useEffect, useState } from "react";
import type { PriorMsg } from "./types";

interface Props {
  prior: PriorMsg | null;
  show: Set<string>;
  onToggle: (key: string) => void;
}

/**
 * Prior layer toggles (plan 11.3). The server sends one `prior` message per mission with a
 * base64 PNG per component plus the blend; each checkbox adds or removes that overlay.
 */
export default function PriorPanel({ prior, show, onToggle }: Props) {
  // Open for a few seconds when a new prior arrives, then collapsed to its title until hovered.
  const [fresh, setFresh] = useState(false);
  useEffect(() => {
    if (!prior) return;
    setFresh(true);
    const timer = window.setTimeout(() => setFresh(false), 4000);
    return () => window.clearTimeout(timer);
  }, [prior]);
  if (!prior) return null;
  const keys = ["blend", ...prior.layers.map((l) => l.key)];
  const label = (k: string) => (k === "blend" ? "Blend" : prior.layers.find((l) => l.key === k)?.label ?? k);
  return (
    <div className={`prior-toggles autohide${fresh ? " open" : ""}`} onClick={(e) => e.stopPropagation()}>
      <span className="autohide-title">Prior</span>
      {keys.map((k) => (
        <label key={k} title={`Overlay the ${label(k)} prior layer`}>
          <input type="checkbox" checked={show.has(k)} onChange={() => onToggle(k)} /> {label(k)}
        </label>
      ))}
    </div>
  );
}