import { useEffect, useState } from "react";
import { api } from "../../api";
import type { AblationRow } from "../../api";

const heat = (v: number) => `rgba(90,170,255,${Math.max(0, Math.min(1, v))})`;

export default function AblationMatrix() {
  const [rows, setRows] = useState<AblationRow[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api.ablation().then((r) => (r as any).error ? setErr((r as any).detail) : setRows(r));
  }, []);
  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  if (!rows) return <div>Loading…</div>;
  return (
    <table style={{ width: "100%", fontSize: 12 }}>
      <thead><tr><th>Combo</th><th>Span F1</th><th>Indo F1</th><th>Stability</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.combo}>
            <td>{r.combo}</td>
            <td style={{ background: heat(r.span_f1) }}>{r.span_f1.toFixed(2)}</td>
            <td style={{ background: heat(r.indo_f1) }}>{r.indo_f1.toFixed(2)}</td>
            <td>{r.stability.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
