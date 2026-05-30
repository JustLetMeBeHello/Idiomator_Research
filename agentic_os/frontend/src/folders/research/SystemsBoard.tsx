import { useEffect, useState } from "react";
import { api } from "../../api";
import type { SystemRow } from "../../api";

const f2 = (x: number | null) => (x == null ? "—" : x.toFixed(2));

export default function SystemsBoard() {
  const [rows, setRows] = useState<SystemRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.systems().then((r) => {
      if ((r as any).error) setErr((r as any).detail);
      else setRows(r);
    });
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!rows) return <div>Loading…</div>;
  return (
    <table style={{ width: "100%", fontSize: 13 }}>
      <thead><tr><th>Sys</th><th>Joint F1</th><th>Stability</th><th>Cls F1</th><th></th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <td>{r.label}</td>
            <td>{f2(r.joint_f1)}{r.mismatch && <span title={`live ${f2(r.joint_f1_live)}`}> ⚠</span>}</td>
            <td>{f2(r.stability)}</td>
            <td>{f2(r.cls_f1)}</td>
            <td style={{ opacity: 0.6 }}>{r.note}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
