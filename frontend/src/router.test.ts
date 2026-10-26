import { describe, expect, it } from "vitest";
import { pageHash, parseHash } from "./router";

describe("router", () => {
  it("parses routes", () => {
    expect(parseHash("")).toEqual({ name: "library" });
    expect(parseHash("#/")).toEqual({ name: "library" });
    expect(parseHash("#/pages/12")).toEqual({ name: "page", id: 12 });
    expect(parseHash("#/pages/12/")).toEqual({ name: "page", id: 12 });
    expect(parseHash("#/pages/x")).toEqual({ name: "library" });
    expect(parseHash(pageHash(5))).toEqual({ name: "page", id: 5 });
    expect(parseHash("#/training")).toEqual({ name: "training" });
    expect(parseHash("#/training/")).toEqual({ name: "training" });
  });
});
