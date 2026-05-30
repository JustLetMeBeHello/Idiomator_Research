import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import Desktop from "./Desktop";
import { api } from "../api";

vi.spyOn(api, "folders").mockResolvedValue([
  { id: "research", name: "Research", icon: "🔬", status: "live" },
  { id: "papers", name: "Papers", icon: "📄", status: "stub" },
]);

test("renders folder icons and opens a window on click", async () => {
  render(<Desktop />);
  const icon = await screen.findByText("Research");
  fireEvent.click(icon);
  await waitFor(() => expect(screen.getByRole("dialog")).toBeInTheDocument());
});
