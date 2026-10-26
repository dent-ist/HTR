import { describe, expect, it } from "vitest";
import { pageHash, parseHash, textHash } from "./router";

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
    expect(parseHash("#/review")).toEqual({ name: "review" });
  });

  it("parses the M6 routes", () => {
    expect(parseHash("#/abbreviations")).toEqual({ name: "abbreviations" });
    expect(parseHash("#/export/")).toEqual({ name: "export" });
    expect(parseHash("#/pages/3/text")).toEqual({ name: "text", id: 3 });
    expect(parseHash(textHash(9))).toEqual({ name: "text", id: 9 });
  });

  it("a page link can name a line", () => {
    expect(pageHash(7, 42)).toBe("#/pages/7?line=42");
    expect(parseHash("#/pages/7?line=42")).toEqual({ name: "page", id: 7, line: 42 });
    expect(parseHash("#/pages/7?line=x")).toEqual({ name: "library" });
    expect(pageHash(7)).toBe("#/pages/7");
  });
});
