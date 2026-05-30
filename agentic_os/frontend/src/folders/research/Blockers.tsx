import { useEffect, useState } from "react";
import { api } from "../../api";
import type { Blocker } from "../../api";

export default function Blockers() {
  const [bl, setBl] = useState<Blocker[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.blockers().then((r) => (r as any).error ? setErr((r as any).detail) : setBl(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!bl) return <div>Loading…</div>;
  return (
    <ul style={{ fontSize: 13 }}>
      {bl.map((b, i) => (
        <li key={i} style={{ color: b.hard_blocker ? "#f77" : "#eee" }}>
          {b.hard_blocker && <strong>[HARD] </strong>}{b.title}
        </li>
      ))}
    </ul>
  );
}
