import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import {
  SHOTS,
  apiPageByName,
  baselineMid,
  clickWorld,
  createDocument,
  makeSyntheticPage,
  pageRow,
  toClient,
  uploadAndSegment,
  watchConsole,
} from "./helpers";

test("repair: split, merge, delete, draw baseline, ignore region, undo; reload and verify via API", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const name = "repair_page.png";
  const consoleErrors = watchConsole(page);
  await createDocument(page, "Repair notebook");
  await uploadAndSegment(page, "Repair notebook", makeSyntheticPage(1, name), name);
  await pageRow(page, name).getByRole("link", { name: "Transcribe" }).click();
  await expect(page.getByLabel(/^Transcription of line 1$/)).toBeFocused();

  const banner = page.getByTestId("mode-banner");
  const lines = async () => (await apiPageByName(request, name)).lines;
  const count = (n: number) => expect.poll(async () => (await lines()).length, { timeout: 15_000 }).toBe(n);

  const start = await lines();
  const n0 = start.length;
  expect(n0).toBeGreaterThanOrEqual(6);
  const line0 = start[0]!;

  // ---- split line 1 at 40% of its baseline: preview first, Enter confirms ----
  await page.keyboard.press("Alt+S");
  await expect(banner).toHaveAttribute("data-mode", "split");
  await clickWorld(page, baselineMid(line0, 0.4));
  await page.screenshot({ path: join(SHOTS, "mode-split.png") });
  await page.keyboard.press("Enter");
  await count(n0 + 1);
  await expect(banner).toHaveAttribute("data-mode", "select");
  const afterSplit = await lines();
  expect(afterSplit[0]!.id).toBe(line0.id);
  expect(afterSplit[0]!.baseline.at(-1)![0]).toBe(afterSplit[1]!.baseline[0]![0]); // cut at the same x
  expect(afterSplit.map((l) => l.index)).toEqual(afterSplit.map((_, i) => i));
  const idsAfterSplit = afterSplit.map((l) => l.id);

  // ---- merge the halves again: pick the right half with a click, Enter confirms ----
  await page.keyboard.press("Alt+M");
  await expect(banner).toHaveAttribute("data-mode", "merge");
  await clickWorld(page, baselineMid(afterSplit[1]!, 0.5));
  await page.screenshot({ path: join(SHOTS, "mode-merge.png") });
  await page.keyboard.press("Enter");
  await count(n0);
  expect((await lines()).some((l) => l.id === afterSplit[1]!.id)).toBe(false);

  // ---- Ctrl+Z undoes the merge (same ids come back), twice more: delete + undo ----
  await page.keyboard.press("Control+z");
  await count(n0 + 1);
  expect((await lines()).map((l) => l.id)).toEqual(idsAfterSplit);
  await page.keyboard.press("Alt+Delete");
  await count(n0);
  await page.keyboard.press("Control+z");
  await count(n0 + 1);
  expect((await lines()).map((l) => l.id)).toEqual(idsAfterSplit);

  // ---- draw a baseline for a "missed" line in the free band above the first line ----
  const all = await lines();
  const pageInfo = await apiPageByName(request, name);
  const minY = Math.min(...all.flatMap((l) => l.polygon.map((p) => p[1])));
  expect(minY, "free space above the first line").toBeGreaterThan(70);
  const y = Math.round(minY * 0.55 + 15);
  const drawn: [number, number][] = [
    [150, y],
    [300, y + 3],
    [450, y],
  ];
  await page.keyboard.press("Alt+B");
  await expect(banner).toHaveAttribute("data-mode", "baseline");
  for (const p of drawn) await clickWorld(page, p);
  await page.screenshot({ path: join(SHOTS, "mode-draw.png") });
  await page.keyboard.press("Enter");
  await count(n0 + 2);
  const withNew = await lines();
  const added = withNew.find((l) => !idsAfterSplit.includes(l.id))!;
  expect(added.index).toBe(0); // placed in top-to-bottom reading order
  added.baseline.forEach((p, i) => {
    expect(Math.abs(p[0] - drawn[i]![0])).toBeLessThanOrEqual(1);
    expect(Math.abs(p[1] - drawn[i]![1])).toBeLessThanOrEqual(1);
  });
  expect(added.polygon.length).toBeGreaterThanOrEqual(4);
  expect(Math.min(...added.polygon.map((p) => p[1]))).toBeGreaterThanOrEqual(0);

  // ---- ignore that band by dragging a rectangle; undo; do it again ----
  const dragRegion = async () => {
    await page.keyboard.press("Alt+R");
    await expect(banner).toHaveAttribute("data-mode", "region");
    const a = await toClient(page, [100, 3]); // after the mode banner has settled, the canvas may have moved
    const b = await toClient(page, [pageInfo.width * 0.7, minY - 3]);
    await page.mouse.move(a.x, a.y);
    await page.mouse.down();
    await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 4 });
    await page.mouse.move(b.x, b.y, { steps: 4 });
    if (!dragRegion.shot) {
      dragRegion.shot = true;
      await page.screenshot({ path: join(SHOTS, "mode-region.png") });
    }
    await page.mouse.up();
  };
  dragRegion.shot = false;
  const addedState = async () => (await lines()).find((l) => l.id === added.id)!.ignored;
  await dragRegion();
  await expect.poll(addedState).toBe(true);
  expect((await lines()).filter((l) => l.ignored).map((l) => l.id)).toEqual([added.id]);
  await page.keyboard.press("Control+z");
  await expect.poll(addedState).toBe(false);
  await dragRegion();
  await expect.poll(addedState).toBe(true);

  // ---- reload: everything persisted ----
  await page.reload();
  await expect(page.getByTestId("progress")).toContainText("0/");
  const final = await apiPageByName(request, name);
  expect(final.lines).toHaveLength(n0 + 2);
  expect(final.lines.map((l) => l.index)).toEqual(final.lines.map((_, i) => i));
  expect(final.lines.find((l) => l.id === added.id)).toMatchObject({ ignored: true, index: 0 });
  expect(final.lines.filter((l) => l.ignored)).toHaveLength(1);
  expect(final.lines.map((l) => l.id).sort()).toEqual([...idsAfterSplit, added.id].sort());
  expect(consoleErrors).toEqual([]);
});
