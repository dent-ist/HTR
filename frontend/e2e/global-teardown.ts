import { rmSync } from "node:fs";

/** Delete everything the run put in the temp directory (see playwright.config.ts). HTR_E2E_KEEP=1 keeps it. */
export default function globalTeardown(): void {
  const root = process.env.HTR_E2E_ROOT;
  if (root && !process.env.HTR_E2E_KEEP) rmSync(root, { recursive: true, force: true });
}
