import { type ChildProcess, spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { API, SHOTS, apiPageByName, createDocument, makeSyntheticPage, uploadAndSegment, waitSaved, watchConsole } from "./helpers";

// M6 end to end on the real backend (real Kraken segmentation): abbreviations, expanded export, raw / expanded text
// view, project bundle export through the UI, the UI's refusal to import into a non-empty project, and a real import
// into a FRESH second backend (own empty data directory) whose contents are then compared with the original through
// the API. The second backend is started and stopped by this test.

const FRESH_PORT = 8101;
const FRESH_API = `http://127.0.0.1:${FRESH_PORT}`;

async function waitHealthy(url: string, ms = 120_000): Promise<void> {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    try {
      if ((await fetch(`${url}/health`)).ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`${url} did not become healthy`);
}

function killTree(p: ChildProcess): void {
  if (p.pid) {
    try {
      process.kill(-p.pid, "SIGTERM");
    } catch {
      /* already gone */
    }
  }
}

test("abbreviations, expanded export, text view, bundle export and import into a fresh project", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const consoleErrors = watchConsole(page, [/409 \(Conflict\)/]); // the refused import is an expected failed request
  const work = mkdtempSync(join(tmpdir(), "htr-e2e-m6-"));
  let fresh: ChildProcess | null = null;
  try {
    // ---- a segmented page with two transcribed lines that use abbreviations ----
    const file = makeSyntheticPage(11, "m6_page.png");
    await createDocument(page, "M6 notebook");
    await uploadAndSegment(page, "M6 notebook", file, "m6_page.png");
    await page.getByRole("row", { name: /m6_page\.png/ }).getByRole("link", { name: "Transcribe" }).click();
    await expect(page.getByLabel(/^Transcription of line 1$/)).toBeFocused();
    await page.keyboard.type("pt c/o SOB");
    await page.keyboard.press("Enter");
    await page.keyboard.type("Pt seen w/ pt");
    await page.keyboard.press("Enter");
    await waitSaved(page);

    // ---- abbreviations page: add entries, an ambiguous one, occurrence counts and the link into the editor ----
    await page.goto("/#/abbreviations");
    await expect(page.getByRole("heading", { name: "Abbreviations", level: 1 })).toBeVisible();
    for (const [w, x] of [
      ["pt", "patient"],
      ["c/o", "complains of"],
      ["SOB", "shortness of breath"],
      ["SOB", "son of a bitch"],
    ] as const) {
      await page.getByLabel("Written form").first().fill(w);
      await page.getByLabel("Expansion").first().fill(x);
      await page.getByRole("button", { name: "Add", exact: true }).click();
      await expect(page.getByRole("button", { name: `Delete ${w} = ${x}` })).toBeVisible();
    }
    await expect(page.getByTestId("ambiguous-total")).toHaveText("2 ambiguous");
    await expect(page.getByRole("button", { name: "3 occurrences of pt" })).toBeVisible();
    await page.getByRole("button", { name: "3 occurrences of pt" }).click();
    const occ = page.getByRole("list", { name: "Occurrences of pt" });
    await expect(occ.getByRole("link")).toHaveCount(3); // pt, Pt, pt
    await page.getByRole("button", { name: "Preview expansion" }).click();
    await expect(page.getByTestId("preview-out")).toHaveText("patient complains of SOB[?]");
    await page.screenshot({ path: join(SHOTS, "abbreviations.png"), fullPage: true });
    await occ.getByRole("link").nth(1).click(); // second occurrence is on line 2
    await expect(page.getByLabel("Transcription of line 2")).toHaveValue("Pt seen w/ pt");

    // ---- text view: raw and expanded side by side ----
    await page.getByRole("button", { name: "Text view" }).click();
    await expect(page.getByRole("heading", { name: "Text: m6_page.png" })).toBeVisible();
    const row1 = page.locator("tr", { hasText: "pt c/o SOB" });
    await expect(row1.locator("td.exp")).toHaveText("patient complains of SOB[?]");
    await expect(page.locator("tr", { hasText: "Pt seen w/ pt" }).locator("td.exp")).toHaveText("Patient seen w/ patient");
    await page.screenshot({ path: join(SHOTS, "text-view.png"), fullPage: true });

    // ---- export page: expanded text for the document ----
    await page.goto("/#/export");
    await page.getByLabel("What to export").selectOption({ label: "M6 notebook" });
    await page.getByRole("button", { name: "Preview text" }).click();
    await expect(page.getByLabel("Export preview")).toContainText("patient complains of SOB");
    const [txt] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Download text" }).click()]);
    expect(txt.suggestedFilename()).toBe("htr-M6-notebook-expanded.txt");
    const exported = readFileSync(await txt.path(), "utf8");
    expect(exported).toBe("=== M6 notebook / page 1 (m6_page.png) ===\npatient complains of SOB\nPatient seen w/ patient\n");
    await page.screenshot({ path: join(SHOTS, "export.png"), fullPage: true });

    // ---- bundle: export through the UI (browser saves the stream to a file) ----
    const [bundle] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "Export project bundle" }).click()]);
    expect(bundle.suggestedFilename()).toMatch(/^htr-project-\d{8}-\d{6}\.htrbundle$/);
    const bundlePath = join(work, bundle.suggestedFilename());
    await bundle.saveAs(bundlePath);
    expect(readFileSync(bundlePath).subarray(0, 2).toString()).toBe("PK");

    // ---- importing it into this (non-empty) project is refused with a clear message, and changes nothing ----
    const before = JSON.stringify((await (await request.get(`${API}/pages`)).json()) as unknown);
    await page.locator('input[type="file"]').setInputFiles(bundlePath);
    await page.getByRole("button", { name: "Import project bundle" }).click();
    await expect(page.getByTestId("import-error")).toContainText("Import refused: the project is not empty");
    await expect(page.getByTestId("import-error")).toContainText("never merges");
    await page.screenshot({ path: join(SHOTS, "export-import-refused.png"), fullPage: true });
    expect(JSON.stringify((await (await request.get(`${API}/pages`)).json()) as unknown)).toBe(before);

    // ---- a fresh backend (empty data directory) takes the bundle ----
    fresh = spawn("uv", ["run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(FRESH_PORT)], {
      cwd: resolve("../backend"),
      env: { ...process.env, HTR_DATA_DIR: join(work, "fresh-data"), HTR_DEVICE: "cpu" },
      stdio: "ignore",
      detached: true,
    });
    await waitHealthy(FRESH_API);
    expect(((await (await request.get(`${FRESH_API}/pages`)).json()) as unknown[]).length).toBe(0);
    const imp = await request.post(`${FRESH_API}/bundle/import`, {
      multipart: { file: { name: "project.htrbundle", mimeType: "application/zip", buffer: readFileSync(bundlePath) } },
      timeout: 120_000,
    });
    expect(imp.status(), await imp.text()).toBe(200);

    // everything is there and identical: pages, lines (text, flags, geometry, order), abbreviations, images, export
    const ids = ((await (await request.get(`${API}/pages`)).json()) as { id: number }[]).map((p) => p.id);
    const freshIds = ((await (await request.get(`${FRESH_API}/pages`)).json()) as { id: number }[]).map((p) => p.id);
    expect(freshIds).toEqual(ids);
    for (const id of ids) {
      const a = await (await request.get(`${API}/pages/${id}`)).json();
      const b = await (await request.get(`${FRESH_API}/pages/${id}`)).json();
      expect(b).toEqual(a);
      expect(Buffer.compare(await (await request.get(`${FRESH_API}/pages/${id}/original`)).body(), await (await request.get(`${API}/pages/${id}/original`)).body())).toBe(0);
    }
    expect(await (await request.get(`${FRESH_API}/abbreviations`)).json()).toEqual(await (await request.get(`${API}/abbreviations`)).json());
    expect(await (await request.get(`${FRESH_API}/export?expanded=true&ambiguous=mark`)).text()).toBe(
      await (await request.get(`${API}/export?expanded=true&ambiguous=mark`)).text(),
    );
    const detail = await apiPageByName(request, "m6_page.png");
    expect(detail.lines[0]).toMatchObject({ text: "pt c/o SOB", is_ground_truth: true });
    // a second import into the now non-empty fresh project is refused too
    const again = await request.post(`${FRESH_API}/bundle/import`, {
      multipart: { file: { name: "project.htrbundle", mimeType: "application/zip", buffer: readFileSync(bundlePath) } },
    });
    expect(again.status()).toBe(409);
    expect(consoleErrors).toEqual([]);
  } finally {
    if (fresh) killTree(fresh);
    rmSync(work, { recursive: true, force: true });
  }
});
