#!/usr/bin/env node
/**
 * GPU-path smoke test for the SWE game after the codegen rewire.
 *
 * Loads apps/swe-game/index.html in real Chrome with the CPU worker
 * engine, runs it for a few seconds at a low resolution, and checks
 * that the simulation advances and produces no console / page errors.
 *
 * Requires a static server at the repo root, e.g.:
 *   python3 -m http.server 8770   (run from the Zoomy repo root)
 *
 * Usage: node cpu_smoke.mjs [--url http://localhost:8770] [--nx 60] [--seconds 4]
 */
import { chromium } from "playwright-core";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);
const BASE = args.url || "http://localhost:8770";
const NX = Number(args.nx || 60);
const SECONDS = Number(args.seconds || 4);
const URL = `${BASE}/apps/swe-game/index.html?gpu=1&profile=1`;

const browser = await chromium.launch({
  executablePath: "/usr/bin/google-chrome",
  headless: true,
  args: ["--no-sandbox"],
});
const page = await browser.newPage();
const errors = [];

// Real JS errors via pageerror / console — but skip the generic
// "Failed to load resource" console line (network failures are caught
// precisely via CDP below, which also gives the URL).
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => {
  if (m.type() !== "error") return;
  const text = m.text();
  if (text.includes("Failed to load resource")) return;
  errors.push("console.error: " + text);
});

// Network failures across the page *and* its workers, via CDP — the
// page-level response handler does not see dedicated-worker requests.
const cdp = await page.context().newCDPSession(page);
await cdp.send("Network.enable");
cdp.on("Network.responseReceived", (e) => {
  if (e.response.status >= 400 && !e.response.url.includes("favicon.ico"))
    errors.push(`HTTP ${e.response.status}: ${e.response.url}`);
});

await page.goto(URL, { waitUntil: "load" });
await page.waitForFunction(() => typeof window.__game !== "undefined", null, {
  timeout: 10000,
});

// Drop to a low resolution (the generic CPU solver is the slow fallback
// engine) and let the worker re-initialise.
await page.evaluate((nx) => window.__game.setResolution(nx), NX);
await page.waitForFunction(
  () => document.getElementById("loading").style.display === "none",
  null,
  { timeout: 15000 }
);

const result = await page.evaluate(async (seconds) => {
  window.__prof.frames = 0;
  window.__prof.lastTime = 0;
  const t0 = window.__game.gameTime;
  window.__game.start();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  return {
    engine: window.__game.engine,
    advanced: window.__game.gameTime - t0,
    frames: window.__prof.frames,
  };
}, SECONDS);

await browser.close();

const ok =
  result.engine === "gpu" &&
  result.advanced > 0 &&
  result.frames > 0 &&
  errors.length === 0;

console.log(`engine=${result.engine}  simAdvanced=${result.advanced.toFixed(2)}s` +
  `  frames=${result.frames}`);
if (errors.length) {
  console.log("errors:");
  for (const e of errors) console.log("  " + e);
}
console.log(ok ? "GPU SMOKE: PASS" : "GPU SMOKE: FAIL");
process.exit(ok ? 0 : 1);
