import { useEffect, useState } from "react";
import { api } from "../../api";
import type { Experiment } from "../../api";

const COLOR: Record<string, string> = {
  done: "#5c5", not_run: "#fa5", skip: "#888", not_planned: "#888", unknown: "#888",
};

export default function ExperimentQueue() {
  const [exps, setExps] = useState<Experiment[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.experiments().then((r) => (r as any).error ? setErr((r as any).detail) : setExps(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!exps) return <div>Loading…</div>;
  return (
    <ul style={{ listStyle: "none", padding: 0, fontSize: 13 }}>
      {exps.map((e) => (
        <li key={e.id} style={{ padding: "4px 0" }}>
          <span style={{ color: COLOR[e.status] ?? "#888" }}>●</span>{" "}
          <strong>{e.id}</strong> {e.what} <em style={{ opacity: 0.6 }}>({e.status})</em>
        </li>
      ))}
    </ul>
  );
}
