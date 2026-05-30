import { useEffect, useState } from "react";
import { api } from "../api";
import type { Folder } from "../api";
import IconGrid from "./IconGrid";
import Window from "./Window";
import Taskbar from "./Taskbar";
import NextStepBar from "./NextStepBar";
import ResearchRoom from "../folders/research/ResearchRoom";
import StubRoom from "../folders/StubRoom";

export default function Desktop() {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [open, setOpen] = useState<string[]>([]);

  useEffect(() => { api.folders().then(setFolders); }, []);

  const openFolder = (id: string) =>
    setOpen((o) => (o.includes(id) ? o : [...o, id]));
  const close = (id: string) => setOpen((o) => o.filter((x) => x !== id));

  const roomFor = (id: string) => {
    const f = folders.find((x) => x.id === id);
    if (id === "research") return <ResearchRoom />;
    return <StubRoom name={f?.name ?? id} />;
  };

  return (
    <div style={{ minHeight: "100vh", background:
      "radial-gradient(circle at 30% 20%, #1a1a2e, #0a0a0f)" }}>
      <NextStepBar />
      <IconGrid folders={folders} onOpen={openFolder} />
      {open.map((id) => (
        <Window key={id} title={folders.find((f) => f.id === id)?.name ?? id}
          onClose={() => close(id)}>
          {roomFor(id)}
        </Window>
      ))}
      <Taskbar open={open} onFocus={openFolder} />
    </div>
  );
}
