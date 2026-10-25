import { useState } from "react";
import type { useMission } from "./useMission";

/** Status bar control for a public deployment: viewers watch, the operator logs in to send commands. */
export default function OperatorLogin({ live }: { live: ReturnType<typeof useMission> }) {
  const [open, setOpen] = useState(false);
  const [password, setPassword] = useState("");
  const [failed, setFailed] = useState(false);

  if (live.config?.operator) {
    return (
      <span className="login">
        <span className="conn on">operator</span>
        <button onClick={() => live.logout()}>Log out</button>
      </span>
    );
  }
  if (!open) {
    return (
      <span className="login">
        <span className="conn off">viewer</span>
        <button onClick={() => setOpen(true)}>Log in</button>
      </span>
    );
  }
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const ok = await live.login(password);
    setFailed(!ok);
    if (ok) {
      setOpen(false);
      setPassword("");
    }
  };
  return (
    <form className="login" onSubmit={submit}>
      <input type="password" autoFocus placeholder="Operator password" value={password}
             onChange={(e) => setPassword(e.target.value)} />
      <button type="submit">Log in</button>
      <button type="button" onClick={() => setOpen(false)}>Cancel</button>
      {failed && <span className="conn off">wrong password</span>}
    </form>
  );
}
