import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import TasksPanel from "./TasksPanel";
import { api } from "../../api";

const T = (id: string, title: string, status = "not_run") => ({
  id, title, source_file: "x", why: "y", next_action: "do z", status, blocked_by: [],
});

test("lists tasks and marks one done, then refetches", async () => {
  const spy = vi.spyOn(api, "tasks")
    .mockResolvedValueOnce([T("exp-02", "Exp 02"), T("exp-04", "Exp 04")])
    .mockResolvedValueOnce([T("exp-04", "Exp 04")]);
  vi.spyOn(api, "markDone").mockResolvedValue({ ok: true, next: T("exp-04", "Exp 04") });

  render(<TasksPanel />);
  const doneBtn = await screen.findAllByRole("button", { name: /done/i });
  fireEvent.click(doneBtn[0]);
  await waitFor(() => expect(api.markDone).toHaveBeenCalledWith("exp-02"));
  await waitFor(() => expect(spy).toHaveBeenCalledTimes(2));
});
