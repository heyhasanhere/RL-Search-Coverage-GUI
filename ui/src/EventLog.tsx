import type { EventMsg } from "./types";
import { fmtTime } from "./types";

export default function EventLog({ events }: { events: EventMsg[] }) {
  return (
    <section className="panel events">
      <h2>Events</h2>
      <ul className="log">
        {[...events].reverse().map((e, i) => (
          <li key={events.length - i} className={`lvl-${e.level}`}>
            <span className="muted">{fmtTime(e.t)}</span> {e.text}
          </li>
        ))}
      </ul>
    </section>
  );
}
