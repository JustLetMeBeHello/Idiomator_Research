import "@testing-library/jest-dom";
import { vi, beforeEach } from "vitest";

beforeEach(() => {
  // Default: any component that fetches without an explicit mock gets an empty array.
  // Use mockImplementation (factory) so each call gets a fresh Response body —
  // a single shared Response instance throws "Body has already been read" after the first .json().
  // Tests that need specific data use vi.spyOn(api, ...) which takes precedence.
  vi.spyOn(globalThis, "fetch").mockImplementation(() =>
    Promise.resolve(
      new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } })
    )
  );
});
