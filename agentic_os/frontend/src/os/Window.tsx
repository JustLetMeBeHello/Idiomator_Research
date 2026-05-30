import { useState } from "react";
import type { ReactNode } from "react";

export default function Window(props: {
  title: string; onClose: () => void; children: ReactNode;
}) {
  const [pos, setPos] = useState({ x: 80, y: 80 });
  const onDrag = (e: React.MouseEvent) => {
    const sx = e.clientX, sy = e.clientY, ox = pos.x, oy = pos.y;
    const move = (m: MouseEvent) =>
      setPos({ x: ox + m.clientX - sx, y: oy + m.clientY - sy });
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };
  return (
    <div role="dialog" aria-label={props.title}
      style={{
        position: "absolute", left: pos.x, top: pos.y, width: 520, minHeight: 240,
        background: "#15151c", border: "1px solid #333", borderRadius: 8,
        boxShadow: "0 12px 40px rgba(0,0,0,.5)", color: "#eee", resize: "both",
        overflow: "auto",
      }}>
      <div onMouseDown={onDrag}
        style={{ display: "flex", justifyContent: "space-between", padding: "6px 10px",
          background: "#22222c", cursor: "move", borderBottom: "1px solid #333" }}>
        <strong>{props.title}</strong>
        <button onClick={props.onClose} aria-label="close">✕</button>
      </div>
      <div style={{ padding: 12 }}>{props.children}</div>
    </div>
  );
}
