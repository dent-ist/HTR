import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  API,
  SHOTS,
  apiPageByName,
  baselineMid,
  canvasView,
  clickWorld,
  createDocument,
  makeSyntheticPage,
  pageRow,
  toClient,
  uploadAndSegment,
  waitSaved,
  watchConsole,
} from "./helpers";

// Browser behaviours that unit tests cannot cover: wheel zoom, drag pan, click-to-select, Ctrl+I, Esc, and a
// failing save (PATCH intercepted) that must keep the typed text and be retryable.
test("canvas zoom/pan/select, Ctrl+I, Esc revert, and save failure with retry", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const name = "browser_page.png";
  // The deliberately failing PATCH makes the browser log one "500" resource error; nothing else is tolerated.
  const consoleErrors = watchConsole(page, [/status of 500/]);
  await createDocument(page, "Browser notebook");
  await uploadAndSegment(page, "Browser notebook", makeSyntheticPage(2, name), name);
  await pageRow(page, name).getByRole("link", { name: "Transcribe" }).click();
  await expect(page.getByLabel(/^Transcription of line 1$/)).toBeFocused();
  const canvas = page.getByTestId("page-canvas");
  const lines = async () => (await apiPageByName(request, name)).lines;
  const pageInfo = await apiPageByName(request, name);

  // ---- wheel zoom at the cursor: scale grows and the world point under the cursor stays put ----
  const fit = await canvasView(page);
  expect(fit.scale).toBeCloseTo(fit.width / pageInfo.width, 2);
  const cx = fit.left + fit.width * 0.4;
  const cy = fit.top + fit.height * 0.5;
  const worldUnder = (v: typeof fit): [number, number] => [(cx - v.left - v.tx) / v.scale, (cy - v.top - v.ty) / v.scale];
  const w0 = worldUnder(fit);
  await page.mouse.move(cx, cy);
  await page.mouse.wheel(0, -300);
  await expect.poll(async () => (await canvasView(page)).scale).toBeGreaterThan(fit.scale * 1.3);
  const zoomed = await canvasView(page);
  const w1 = worldUnder(zoomed);
  expect(Math.abs(w1[0] - w0[0])).toBeLessThan(1.5);
  expect(Math.abs(w1[1] - w0[1])).toBeLessThan(1.5);
  await page.screenshot({ path: join(SHOTS, "editor-zoomed.png") });

  // ---- drag pans by the dragged distance and does not select anything ----
  const before = await canvasView(page);
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx + 60, cy + 30, { steps: 5 });
  await page.mouse.up();
  await expect.poll(async () => (await canvasView(page)).tx).toBeCloseTo(before.tx + 60, 0);
  expect((await canvasView(page)).ty).toBeCloseTo(before.ty + 30, 0);
  await expect(page.getByLabel(/^Transcription of line 1$/)).toBeFocused(); // selection unchanged

  // ---- Alt+0 fits the page width again ----
  await page.keyboard.press("Alt+0");
  await expect.poll(async () => (await canvasView(page)).scale).toBeCloseTo(fit.scale, 3);

  // ---- click-to-select: click the third visible line on the canvas ----
  const all = await lines();
  const target = all[2]!;
  const mid = baselineMid(target, 0.5);
  const spot = await toClient(page, mid);
  const v = await canvasView(page);
  expect(spot.y).toBeLessThan(v.top + v.height - 10); // visible without panning
  await clickWorld(page, mid);
  await expect(page.getByLabel("Transcription of line 3")).toBeFocused();
  await expect(page.getByText("Line 3 of")).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "editor-selected.png") });

  // ---- Esc reverts unsaved typing ----
  const input = page.getByLabel("Transcription of line 3");
  await page.keyboard.type("scratch text");
  await expect(input).toHaveValue("scratch text");
  await expect(page.getByTestId("save-state")).toHaveText("Unsaved changes");
  await page.keyboard.press("Escape");
  await expect(input).toHaveValue("");
  await waitSaved(page);

  // ---- Ctrl+I ignores, and again un-ignores ----
  await page.keyboard.press("Control+i");
  await expect.poll(async () => (await lines())[2]!.ignored).toBe(true);
  await expect(input).toHaveAttribute("readonly", "");
  await expect(page.locator(".dock .badge.ign")).toBeVisible();
  await page.keyboard.press("Control+i");
  await expect.poll(async () => (await lines())[2]!.ignored).toBe(false);
  await expect(input).not.toHaveAttribute("readonly", "");

  // ---- save failure: typed text is kept, error is shown, Retry works ----
  await page.route(`${API}/lines/*`, (route) =>
    route.request().method() === "PATCH" ? route.fulfill({ status: 500, json: { detail: "simulated outage" } }) : route.continue(),
  );
  await page.keyboard.type("keep this text");
  await page.keyboard.press("Enter"); // optimistic save, moves to line 4
  const alert = page.getByTestId("save-error");
  await expect(alert).toContainText("simulated outage");
  await expect(alert).toContainText("Your text is kept");
  await expect(page.getByTestId("save-state")).toHaveText("Save failed");
  await page.screenshot({ path: join(SHOTS, "editor-save-failed.png") });
  expect((await lines())[2]!.text).toBeFalsy(); // server never got it
  await alert.getByRole("button", { name: "Go to line" }).click();
  await expect(page.getByLabel("Transcription of line 3")).toHaveValue("keep this text");

  await page.unroute(`${API}/lines/*`);
  await alert.getByRole("button", { name: "Retry save" }).click();
  await expect(page.getByTestId("save-error")).toBeHidden();
  await waitSaved(page);
  expect((await lines())[2]!).toMatchObject({ text: "keep this text", is_ground_truth: true, text_source: "manual" });

  // reload: still there
  await page.reload();
  await expect(page.getByTestId("progress")).toContainText("1/");
  await expect(canvas).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
