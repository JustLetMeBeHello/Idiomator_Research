import { render, screen } from "@testing-library/react";
import { vi } from "vitest";
import SystemsBoard from "./SystemsBoard";
import { api } from "../../api";

vi.spyOn(api, "systems").mockResolvedValue([
  { label: "D", joint_f1: 0.7515, joint_f1_live: 0.7515, stability: 0.7061, cls_f1: 0.7823, mismatch: false, note: null },
  { label: "G", joint_f1: 0.4750, joint_f1_live: 0.4750, stability: 0.4658, cls_f1: null, mismatch: false, note: "span-only, not comparable" },
]);

test("renders systems with 2-decimal metrics and G note", async () => {
  render(<SystemsBoard />);
  expect(await screen.findByText("0.75")).toBeInTheDocument();
  expect(screen.getByText(/span-only, not comparable/)).toBeInTheDocument();
});
