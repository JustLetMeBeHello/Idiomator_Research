const BASE = "http://localhost:8011";

export type SystemRow = {
  label: string; joint_f1: number | null; joint_f1_live: number | null;
  stability: number | null; cls_f1: number | null; mismatch: boolean; note: string | null;
};
export type AblationRow = { combo: string; span_f1: number; indo_f1: number; stability: number };
export type Experiment = { id: string; script: string; what: string; status: string; notes: string };
export type Blocker = { title: string; hard_blocker: boolean };
export type Commit = { hash: string; subject: string; when: string };
export type Folder = { id: string; name: string; icon: string; status: "live" | "stub" };
export type Task = {
  id: string; title: string; source_file: string; why: string;
  next_action: string; status: string; blocked_by: string[];
};

async function get<T>(path: string): Promise<T> {
  const r = await fetch(BASE + path);
  return r.json() as Promise<T>;
}

export const api = {
  folders: () => get<Folder[]>("/api/folders"),
  systems: () => get<SystemRow[]>("/api/research/systems"),
  ablation: () => get<AblationRow[]>("/api/research/ablation"),
  experiments: () => get<Experiment[]>("/api/research/experiments"),
  blockers: () => get<Blocker[]>("/api/research/blockers"),
  activity: () => get<Commit[]>("/api/activity"),
  tasks: () => get<Task[]>("/api/tasks"),
  next: () => get<Task | null>("/api/next"),
  markDone: (id: string) =>
    fetch(`${BASE}/api/tasks/${id}/done`, { method: "POST" }).then((r) => r.json()),
};
