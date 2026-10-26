import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "@playwright/test";

// Opt-in end-to-end test (`make test-e2e`): real backend (own temp data dir, real Kraken) + Vite dev server.
// The frontend must run on 5173 (CORS allowlist); the API uses 8100 so a normal `make dev` on 8000 is not touched.
const API_PORT = 8100;
const WEB_PORT = 5173;
// Config is evaluated again in workers; the env var makes them reuse the directory chosen by the main process.
// Everything the run writes to the temp directory (the app's data dir, generated page images) lives under one root,
// which the global teardown deletes, so a run leaves nothing behind (set HTR_E2E_KEEP=1 to keep it for debugging).
process.env.HTR_E2E_ROOT ??= mkdtempSync(join(tmpdir(), "htr-e2e-"));
process.env.HTR_E2E_DATA ??= join(process.env.HTR_E2E_ROOT, "data");
process.env.HTR_E2E_API ??= `http://127.0.0.1:${API_PORT}`;
process.env.HTR_E2E_SHOTS ??= join(process.cwd(), "test-results", "screenshots");

export default defineConfig({
  testDir: "./e2e",
  globalTeardown: "./e2e/global-teardown.ts",
  timeout: 300_000,
  expect: { timeout: 15_000 },
  workers: 1,
  fullyParallel: false,
  reporter: [["list"]],
  outputDir: "test-results/artifacts",
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    viewport: { width: 1400, height: 900 },
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: `uv run uvicorn app.main:app --host 127.0.0.1 --port ${API_PORT}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${API_PORT}/health`,
      env: { HTR_DATA_DIR: process.env.HTR_E2E_DATA, HTR_DEVICE: "cpu" },
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      url: `http://127.0.0.1:${WEB_PORT}/`,
      env: { VITE_API_BASE_URL: process.env.HTR_E2E_API },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
