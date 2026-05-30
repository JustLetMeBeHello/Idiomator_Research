import { useEffect, useState } from "react";
import { api } from "../../api";
import type { Task } from "../../api";

export default function TasksPanel() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const load = () =>
    api.tasks().then((r) => (r as any).error ? setErr((r as any).detail) : setTasks(r));
  useEffect(() => { load(); }, []);

  const done = async (id: string) => {
    const res = await api.markDone(id);
    if (res.error) { setErr(res.detail); return; }
    await load();
  };

  if (err) return <div style={{ color: "#f77" }}>Error: {err}</div>;
  return (
    <ol style={{ fontSize: 13, paddingLeft: 18 }}>
      {tasks.map((t) => (
        <li key={t.id} style={{ marginBottom: 6 }}>
          <strong>{t.title}</strong>
          <div style={{ opacity: 0.6 }}>{t.why}</div>
          <code style={{ fontSize: 11 }}>{t.next_action}</code>{" "}
          {t.id.startsWith("exp-") && (
            <button aria-label={`done ${t.id}`} onClick={() => done(t.id)}>✓ done</button>
          )}
        </li>
      ))}
    </ol>
  );
}
