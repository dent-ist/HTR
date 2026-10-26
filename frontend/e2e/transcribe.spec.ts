import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, test } from "@playwright/test";

const API = process.env.HTR_E2E_API!;
const SHOTS = process.env.HTR_E2E_SHOTS!;

interface ApiLine {
  index: number;
  text: string | null;
  is_ground_truth: boolean;
  text_source: string | null;
}

test("upload, segment (real Kraken), transcribe 3 lines by keyboard, reload, verify via API", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });

  // Synthetic page from the repo script (NOT real data).
  const imgDir = mkdtempSync(join(tmpdir(), "htr-e2e-img-"));
  execFileSync(
    "uv",
    ["run", "--project", resolve("../backend"), "python", resolve("../scripts/make_synthetic_pages.py"), imgDir, "-n", "1", "--seed", "0"],
    { stdio: "inherit" },
  );
  const file = join(imgDir, readdirSync(imgDir).find((f) => f.endsWith(".png"))!);

  const consoleErrors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));

  // ---- library: create a document, upload, duplicate upload ----
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "HTR library" })).toBeVisible();
  await page.getByLabel("New document name").fill("E2E notebook");
  await page.getByRole("button", { name: "Create document" }).click();
  await expect(page.getByRole("heading", { name: /E2E notebook/ })).toBeVisible();

  const input = page.locator('input[type="file"]');
  await input.setInputFiles(file);
  await expect(page.getByText(/Imported as page #\d+/)).toBeVisible();
  await input.setInputFiles(file);
  await expect(page.getByText(/Already imported \(identical file\)/)).toBeVisible();

  // ---- segment with real Kraken, watch the job finish ----
  await page.getByRole("button", { name: /^Segment synthetic_01/ }).click();
  await expect(page.getByText(/Queued for segmentation|Segmenting/)).toBeVisible();
  await expect(page.locator(".status.segmented")).toBeVisible({ timeout: 240_000 });
  await page.screenshot({ path: join(SHOTS, "library.png"), fullPage: true });

  // ---- page editor: transcribe 3 lines with the keyboard only ----
  await page.getByRole("link", { name: "Transcribe" }).click();
  const box = page.getByLabel(/^Transcription of line 1$/);
  await expect(box).toBeFocused();
  await expect(page.getByTestId("line-crop")).toBeVisible();
  const texts = ["first e2e line", "second e2e line", "third e2e line"];
  for (let i = 0; i < texts.length; i++) {
    await expect(page.getByLabel(`Transcription of line ${i + 1}`)).toBeFocused();
    await page.keyboard.type(texts[i]!);
    if (i === 1) await page.screenshot({ path: join(SHOTS, "editor-typing.png") });
    await page.keyboard.press("Enter");
  }
  await expect(page.getByLabel("Transcription of line 4")).toBeFocused();
  await expect(page.getByTestId("save-state")).toHaveText("All changes saved");
  await page.keyboard.press("Control+/");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "editor-help.png") });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await expect(page.getByTestId("save-state")).toHaveText("All changes saved");
  await page.screenshot({ path: join(SHOTS, "editor.png") });

  // ---- reload and verify persisted state in the UI and through the API ----
  await page.reload();
  await expect(page.getByTestId("progress")).toContainText("3/");
  const pages = (await (await request.get(`${API}/pages`)).json()) as { id: number }[];
  expect(pages).toHaveLength(1);
  const detail = (await (await request.get(`${API}/pages/${pages[0]!.id}`)).json()) as { lines: ApiLine[]; status: string };
  expect(detail.lines.length).toBeGreaterThanOrEqual(4);
  texts.forEach((t, i) => {
    expect(detail.lines[i]).toMatchObject({ text: t, is_ground_truth: true, text_source: "manual" });
  });
  for (const l of detail.lines.slice(3)) expect(l.is_ground_truth).toBe(false);
  expect(detail.status).toBe("partially_transcribed");
  await expect(page.getByLabel("Transcription of line 4")).toBeFocused(); // resumes at the first untranscribed line

  expect(consoleErrors).toEqual([]);
});
