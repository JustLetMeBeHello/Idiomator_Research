export default function Taskbar(props: {
  open: string[]; onFocus: (id: string) => void;
}) {
  return (
    <div style={{ position: "fixed", bottom: 0, left: 0, right: 0, height: 40,
      background: "#0d0d12", borderTop: "1px solid #333", display: "flex",
      alignItems: "center", gap: 8, padding: "0 12px" }}>
      <strong style={{ color: "#7af" }}>◈ Agentic OS</strong>
      {props.open.map((id) => (
        <button key={id} onClick={() => props.onFocus(id)}
          style={{ background: "#22222c", color: "#eee", border: "1px solid #333",
            borderRadius: 4, padding: "2px 10px" }}>{id}</button>
      ))}
    </div>
  );
}
