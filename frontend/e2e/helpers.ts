import { execFileSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { type APIRequestContext, type Page, expect } from "@playwright/test";

export const API = process.env.HTR_E2E_API!;
export const SHOTS = process.env.HTR_E2E_SHOTS!;

export interface ApiLine {
  id: number;
  index: number;
  text: string | null;
  text_source: string | null;
  is_ground_truth: boolean;
  ignored: boolean;
  baseline: [number, number][];
  polygon: [number, number][];
}
export interface ApiPage {
  id: number;
  width: number;
  height: number;
  status: string;
  original_filename: string;
  lines: ApiLine[];
}

/** A synthetic page (scripts/make_synthetic_pages.py, NOT real data) copied to a unique file name. */
export function makeSyntheticPage(seed: number, filename: string): string {
  const dir = mkdtempSync(join(process.env.HTR_E2E_ROOT ?? tmpdir(), "img-"));
  execFileSync(
    "uv",
    ["run", "--project", resolve("../backend"), "python", resolve("../scripts/make_synthetic_pages.py"), dir, "-n", "1", "--seed", String(seed)],
    { stdio: "inherit" },
  );
  const src = join(dir, readdirSync(dir).find((f) => f.endsWith(".png"))!);
  const dest = join(dir, filename);
  copyFileSync(src, dest);
  return dest;
}

/** Collects console errors / page errors; `ignore` filters expected ones. */
export function watchConsole(page: Page, ignore: RegExp[] = []): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !ignore.some((r) => r.test(m.text()))) errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  return errors;
}

export async function createDocument(page: Page, name: string): Promise<void> {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "HTR library" })).toBeVisible();
  await page.getByLabel("New document name").fill(name);
  await page.getByRole("button", { name: "Create document" }).click();
  await expect(page.getByRole("heading", { name: new RegExp(name) })).toBeVisible();
}

export function docSection(page: Page, name: string) {
  return page.getByRole("region", { name: new RegExp(name) });
}

export function pageRow(page: Page, filename: string) {
  return page.getByRole("row", { name: new RegExp(filename.replace(".", "\\.")) });
}

/** Upload one file into a document and run real Kraken segmentation on it. */
export async function uploadAndSegment(page: Page, docName: string, file: string, filename: string): Promise<void> {
  await docSection(page, docName).locator('input[type="file"]').setInputFiles(file);
  await expect(docSection(page, docName).getByText(/Imported as page #\d+/)).toBeVisible();
  await page.getByRole("button", { name: new RegExp(`^Segment ${filename.replace(".", "\\.")}`) }).click();
  await expect(pageRow(page, filename).locator(".status.segmented")).toBeVisible({ timeout: 240_000 });
}

export async function apiPageByName(request: APIRequestContext, filename: string): Promise<ApiPage> {
  const pages = (await (await request.get(`${API}/pages`)).json()) as ApiPage[];
  const found = pages.find((p) => p.original_filename === filename);
  expect(found, `page ${filename} in the API`).toBeTruthy();
  return (await (await request.get(`${API}/pages/${found!.id}`)).json()) as ApiPage;
}

export interface CanvasView {
  scale: number;
  tx: number;
  ty: number;
  left: number;
  top: number;
  width: number;
  height: number;
}

export async function canvasView(page: Page): Promise<CanvasView> {
  const c = page.getByTestId("page-canvas");
  const box = (await c.boundingBox())!;
  return {
    scale: Number(await c.getAttribute("data-scale")),
    tx: Number(await c.getAttribute("data-tx")),
    ty: Number(await c.getAttribute("data-ty")),
    left: box.x,
    top: box.y,
    width: box.width,
    height: box.height,
  };
}

/** Page (working-copy) coordinates to browser client coordinates for the current viewport. */
export async function toClient(page: Page, world: [number, number]): Promise<{ x: number; y: number }> {
  const v = await canvasView(page);
  return { x: v.left + world[0] * v.scale + v.tx, y: v.top + world[1] * v.scale + v.ty };
}

export async function clickWorld(page: Page, world: [number, number]): Promise<void> {
  const { x, y } = await toClient(page, world);
  await page.mouse.click(x, y);
}

export function baselineMid(line: ApiLine, fraction = 0.5): [number, number] {
  const b = line.baseline;
  const x0 = b[0]![0];
  const x1 = b[b.length - 1]![0];
  const x = x0 + (x1 - x0) * fraction;
  for (let i = 1; i < b.length; i++) {
    const [ax, ay] = b[i - 1]!;
    const [bx, by] = b[i]!;
    if (x >= Math.min(ax, bx) && x <= Math.max(ax, bx) && ax !== bx) return [x, ay + ((by - ay) * (x - ax)) / (bx - ax)];
  }
  return [x, b[0]![1]];
}

export async function waitSaved(page: Page): Promise<void> {
  await expect(page.getByTestId("save-state")).toHaveText("All changes saved");
}
