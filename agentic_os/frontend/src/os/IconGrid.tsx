import type { Folder } from "../api";

export default function IconGrid(props: {
  folders: Folder[]; onOpen: (id: string) => void;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 24, padding: 32 }}>
      {props.folders.map((f) => (
        <button key={f.id} onClick={() => props.onOpen(f.id)}
          style={{ width: 96, height: 96, display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center", gap: 6,
            background: "transparent", border: "none", color: "#eee", cursor: "pointer" }}>
          <span style={{ fontSize: 40 }}>{f.icon}</span>
          <span>{f.name}</span>
          {f.status === "stub" && <small style={{ opacity: 0.5 }}>soon</small>}
        </button>
      ))}
    </div>
  );
}
