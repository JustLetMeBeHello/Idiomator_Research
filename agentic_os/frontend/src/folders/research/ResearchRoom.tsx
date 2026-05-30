import type { ReactNode } from "react";
import SystemsBoard from "./SystemsBoard";
import AblationMatrix from "./AblationMatrix";
import ExperimentQueue from "./ExperimentQueue";
import Blockers from "./Blockers";
import Activity from "./Activity";
import TasksPanel from "./TasksPanel";

const Card = (p: { title: string; children: ReactNode }) => (
  <section style={{ border: "1px solid #333", borderRadius: 6, padding: 10, marginBottom: 10 }}>
    <h4 style={{ margin: "0 0 8px" }}>{p.title}</h4>
    {p.children}
  </section>
);

export default function ResearchRoom() {
  return (
    <div data-testid="research-room">
      <Card title="Systems"><SystemsBoard /></Card>
      <Card title="Ablation Matrix"><AblationMatrix /></Card>
      <Card title="Experiment Queue"><ExperimentQueue /></Card>
      <Card title="Blockers"><Blockers /></Card>
      <Card title="Tasks"><TasksPanel /></Card>
      <Card title="Activity"><Activity /></Card>
    </div>
  );
}
