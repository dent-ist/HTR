import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  SHOTS,
  apiPageByName,
  createDocument,
  docSection,
  makeSyntheticPage,
  pageRow,
  uploadAndSegment,
  waitSaved,
  watchConsole,
} from "./helpers";

test("upload, segment (real Kraken), transcribe 3 lines by keyboard, reload, verify via API", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const file = makeSyntheticPage(0, "synthetic_01.png");
  const consoleErrors = watchConsole(page);

  // ---- library: create a document, upload, duplicate upload ----
  await createDocument(page, "E2E notebook");
  await uploadAndSegment(page, "E2E notebook", file, "synthetic_01.png");
  await docSection(page, "E2E notebook").locator('input[type="file"]').setInputFiles(file);
  await expect(page.getByText(/Already imported \(identical file\)/)).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "library.png"), fullPage: true });

  // ---- page editor: transcribe 3 lines with the keyboard only ----
  await pageRow(page, "synthetic_01.png").getByRole("link", { name: "Transcribe" }).click();
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
  await waitSaved(page);
  await page.keyboard.press("Control+/");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "editor-help.png") });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toBeHidden();
  await waitSaved(page);
  await page.screenshot({ path: join(SHOTS, "editor.png") });

  // ---- reload and verify persisted state in the UI and through the API ----
  await page.reload();
  await expect(page.getByTestId("progress")).toContainText("3/");
  const detail = await apiPageByName(request, "synthetic_01.png");
  expect(detail.lines.length).toBeGreaterThanOrEqual(4);
  texts.forEach((t, i) => {
    expect(detail.lines[i]).toMatchObject({ text: t, is_ground_truth: true, text_source: "manual" });
  });
  for (const l of detail.lines.slice(3)) expect(l.is_ground_truth).toBe(false);
  expect(detail.status).toBe("partially_transcribed");
  await expect(page.getByLabel("Transcription of line 4")).toBeFocused(); // resumes at the first untranscribed line

  expect(consoleErrors).toEqual([]);
});
