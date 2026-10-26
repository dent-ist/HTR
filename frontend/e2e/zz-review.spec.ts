import { existsSync, mkdirSync, symlinkSync } from "node:fs";
import { join, resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { API, SHOTS, apiPageByName, createDocument, makeSyntheticPage, pageRow, uploadAndSegment, waitSaved, watchConsole } from "./helpers";

// Named zz- so it runs last: it installs the base model into the shared e2e data dir, which training.spec.ts must not see.
// Needs the pretrained model (`make base-model`): predictions come from it. It is linked into the e2e data dir.
const BASE = resolve("../data/models/base");

interface Stats {
  rounds: { reviewed: number; accepted: number; corrected: number; ignored: number }[];
}

async function totals(request: import("@playwright/test").APIRequestContext) {
  const s = (await (await request.get(`${API}/review/stats`)).json()) as Stats;
  const sum = (k: "reviewed" | "accepted" | "corrected" | "ignored") => s.rounds.reduce((a, r) => a + r[k], 0);
  return { reviewed: sum("reviewed"), accepted: sum("accepted"), corrected: sum("corrected"), ignored: sum("ignored") };
}

test("accept a prediction in the editor, then review three lines on the review page; verified through the API", async ({ page, request }) => {
  test.skip(!existsSync(BASE), "run `make base-model` first");
  mkdirSync(SHOTS, { recursive: true });
  const consoleErrors = watchConsole(page);
  const dataDir = process.env.HTR_E2E_DATA!;
  mkdirSync(join(dataDir, "models"), { recursive: true });
  if (!existsSync(join(dataDir, "models", "base"))) symlinkSync(BASE, join(dataDir, "models", "base"));

  const file = makeSyntheticPage(5, "review_01.png");
  await createDocument(page, "Review notebook");
  await uploadAndSegment(page, "Review notebook", file, "review_01.png");

  // predict every line with the base model (a real job on the real model)
  const started = await request.post(`${API}/jobs/predict`, { data: {} });
  expect(started.status()).toBe(202);
  const jobId = ((await started.json()) as { id: number }).id;
  await expect
    .poll(async () => ((await (await request.get(`${API}/jobs/${jobId}`)).json()) as { status: string }).status, { timeout: 240_000 })
    .toBe("succeeded");

  const before = await apiPageByName(request, "review_01.png");
  const first = before.lines[0]!;
  expect(first.text_source).toBe("predicted");
  expect(first.text).toBeTruthy();
  const t0 = await totals(request);

  // ---- editor: the prediction is prefilled and labelled; Enter accepts it
  await pageRow(page, "review_01.png").getByRole("link", { name: "Transcribe" }).click();
  const box = page.getByLabel("Transcription of line 1");
  await expect(box).toHaveValue(first.text!);
  await expect(page.getByTestId("predicted-badge")).toBeVisible();
  await expect(page.getByText(/Predicted text, not yet checked/)).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Lines" }).getByText(/\d+% (high|medium|low)/).first()).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "editor-prediction.png") });
  await page.keyboard.press("Enter");
  await waitSaved(page);
  const after = await apiPageByName(request, "review_01.png");
  expect(after.lines[0]).toMatchObject({ text: first.text, is_ground_truth: true, text_source: "manual" });
  const t1 = await totals(request);
  expect(t1.accepted).toBe(t0.accepted + 1);

  // ---- review page: accept one, correct one, ignore one (keyboard only)
  await page.getByRole("button", { name: "Review" }).click();
  await expect(page).toHaveURL(/#\/review$/);
  const input = page.getByLabel("Text of the line (prefilled with the prediction)");
  await expect(input).toBeFocused();
  await expect(page.getByTestId("review-crop")).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "review.png") });
  await page.keyboard.press("Enter"); // accept
  await expect(page.getByTestId("review-progress")).toContainText("1 reviewed");
  await expect(input).toBeFocused();
  await page.keyboard.press("End");
  await page.keyboard.type("!"); // a correction
  await expect(page.getByTestId("review-hint")).toContainText("save correction");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("review-progress")).toContainText("2 reviewed");
  await page.keyboard.press("Control+i"); // ignore
  await expect(page.getByTestId("review-progress")).toContainText("3 reviewed (1 accepted, 1 corrected, 1 ignored)");

  const t2 = await totals(request);
  expect(t2).toEqual({ reviewed: t1.reviewed + 3, accepted: t1.accepted + 1, corrected: t1.corrected + 1, ignored: t1.ignored + 1 });
  expect(consoleErrors).toEqual([]);
});
