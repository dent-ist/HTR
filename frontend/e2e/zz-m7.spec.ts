import { expect, test } from "@playwright/test";
import { API, watchConsole } from "./helpers";

// M7 in a real browser: the health strip, the offline banner with Retry, the error toast, the suggestions panel being
// off by default, and the server refusing a forged Host header.
test("status strip, offline banner with retry, error toast, local-LLM panel off, forged Host refused", async ({ page, request }) => {
  // The deliberately failing requests make Chromium log resource errors; nothing else is tolerated.
  const consoleErrors = watchConsole(page, [/Failed to load resource/, /net::ERR_/, /status of 5\d\d/]);

  // ---- health strip ----
  await page.goto("/");
  const pill = page.getByTestId("status-indicator");
  await expect(pill).toHaveText(/Ready/);
  await pill.click();
  const panel = page.getByRole("region", { name: "System status" });
  await expect(panel).toContainText("Database");
  await expect(panel).toContainText("Job worker");
  await expect(panel).toContainText("Disk space");
  await pill.click();

  // ---- offline banner and Retry ----
  await page.route(`${API}/**`, (route) => route.abort());
  await page.reload();
  const banner = page.getByTestId("offline-banner");
  await expect(banner).toBeVisible();
  await expect(banner).toContainText("make dev");
  await expect(pill).toHaveText(/Server offline/);
  await page.unroute(`${API}/**`);
  await banner.getByRole("button", { name: "Retry" }).click();
  await expect(banner).toBeHidden();
  await expect(pill).toHaveText(/Ready/);

  // ---- an API error shows the message and the hint, once, as a toast ----
  await page.route(`${API}/documents`, (route) =>
    route.request().method() === "POST"
      ? route.fulfill({
          status: 507,
          json: { detail: "x", code: "disk_full", message: "The disk is full, so nothing more can be saved.", hint: "Free some space." },
        })
      : route.continue(),
  );
  await page.getByLabel("New document name").fill("Will not be created");
  await page.getByRole("button", { name: "Create document" }).click();
  await expect(page.getByTestId("toasts")).toContainText("The disk is full, so nothing more can be saved. Free some space.");
  await page.unroute(`${API}/documents`);

  // ---- the local-LLM suggestions panel is off by default and offers nothing to run ----
  await page.goto("/#/abbreviations");
  await expect(page.getByTestId("llm-off")).toContainText("HTR_LLM_ENABLED=1");
  await expect(page.getByRole("button", { name: "Suggest abbreviations" })).toHaveCount(0);

  // ---- a forged Host header is refused by the server (DNS-rebinding protection) ----
  const forged = await request.get(`${API}/documents`, { headers: { Host: "attacker.example:8100" } });
  expect(forged.status()).toBe(421);
  expect((await forged.json()).code).toBe("host_not_allowed");
  const normal = await request.get(`${API}/documents`);
  expect(normal.status()).toBe(200);

  expect(consoleErrors).toEqual([]);
});
