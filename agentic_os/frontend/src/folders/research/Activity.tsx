import { useEffect, useState } from "react";
import { api } from "../../api";
import type { Commit } from "../../api";

export default function Activity() {
  const [cs, setCs] = useState<Commit[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.activity().then((r) => (r as any).error ? setErr((r as any).detail) : setCs(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!cs) return <div>Loading…</div>;
  return (
    <ul style={{ fontSize: 12, listStyle: "none", padding: 0 }}>
      {cs.map((c) => (
        <li key={c.hash}><code>{c.hash}</code> {c.subject} <span style={{ opacity: 0.5 }}>{c.when}</span></li>
      ))}
    </ul>
  );
}
