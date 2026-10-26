import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { SHOTS, watchConsole } from "./helpers";

// The e2e backend has a fresh data directory without a downloaded base model, so this checks the page loads against
// the real API, shows the ground-truth counts and the recommended minimum, and explains what is missing.
// A real training run is covered by `make test-train` (backend) and the synthetic-project proof in the README.
test("training page loads from the library, shows counts, warnings and the missing-model hint", async ({ page, request }) => {
  mkdirSync(SHOTS, { recursive: true });
  const consoleErrors = watchConsole(page);
  await page.goto("/");
  await page.getByRole("link", { name: "Training" }).click();
  await expect(page).toHaveURL(/#\/training$/);
  await expect(page.getByRole("heading", { name: "Training", level: 1 })).toBeVisible();

  await expect(page.getByTestId("gt-count")).toHaveText(/\d+ ground-truth lines?, about 100 recommended for the first round/);
  await expect(page.getByRole("list", { name: /Warnings/ })).toContainText("recommended");
  await expect(page.getByText(/No recognition model is installed/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Start training" })).toBeDisabled();
  await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  await expect(page.getByText("No models yet.")).toBeVisible();
  await page.screenshot({ path: join(SHOTS, "training.png"), fullPage: true });

  // the numbers on the page are the API's
  const summary = (await (await request.get(`${process.env.HTR_E2E_API}/training/summary`)).json()) as { ground_truth_lines: number };
  await expect(page.getByTestId("gt-count")).toContainText(`${summary.ground_truth_lines} ground-truth`);

  await page.getByRole("link", { name: "Back to library" }).click();
  await expect(page.getByRole("heading", { name: "HTR library" })).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
