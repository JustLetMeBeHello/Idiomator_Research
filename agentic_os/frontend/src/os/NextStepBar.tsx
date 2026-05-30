import { useEffect, useState } from "react";
import { api } from "../api";
import type { Task } from "../api";

export default function NextStepBar() {
  const [next, setNext] = useState<Task | null>(null);
  useEffect(() => {
    api.next().then((r) => setNext(r && (r as any).error ? null : r));
  }, []);
  return (
    <div style={{ position: "sticky", top: 0, zIndex: 50, background: "#101820",
      borderBottom: "1px solid #2a3a4a", padding: "8px 16px", color: "#cfe" }}>
      {next ? (
        <span>▶ <strong>Next:</strong> {next.title} — <em style={{ opacity: 0.7 }}>{next.why}</em>{" "}
          <code style={{ fontSize: 11 }}>{next.next_action}</code></span>
      ) : (
        <span>✓ No pending next step.</span>
      )}
    </div>
  );
}
