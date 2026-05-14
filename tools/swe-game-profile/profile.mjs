#!/usr/bin/env node
/**
 * Drive the SWE game in a real Chrome (via Playwright) and profile both
 * engines.  Compares CPU (worker) vs GPU (WebGL2) on the same scenario.
 *
 * Usage: node profile.mjs [--url http://localhost:8765/] [--seconds 5]
 *
 * Reports per engine:
 *   - frames captured during the window
 *   - sim seconds advanced (the in-game clock)
 *   - simSec / wallSec ratio  (higher = faster playthrough)
 *   - average JS time per frame for the simulation step (advance)
 *   - average JS time per frame for rendering
 *   - any console errors that fired
 *
 * Requires playwright-core and a local Google Chrome.  No browser download.
 */

import { chromium } from "playwright-core";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);
const URL_BASE = args.url || "http://localhost:8765/";
const WINDOW_SEC = Number(args.seconds || 5);
const NX = Number(args.nx || 200);

const CHROME_PATH = "/usr/bin/google-chrome";

async function run(engine) {
  // GPU is the default engine now; ?gpu=0 forces the CPU worker.
  const search = engine === "gpu" ? "?gpu=1&profile=1" : "?gpu=0&profile=1";
  const url = URL_BASE.replace(/\/$/, "") + "/" + search;

  const browser = await chromium.launch({
    executablePath: CHROME_PATH,
    headless: true,
    args: [
      "--no-sandbox",
      "--enable-features=Vulkan,UseSkiaRenderer",
      "--ignore-gpu-blocklist",
      "--enable-gpu-rasterization",
      "--enable-webgl",
      "--use-angle=default",
    ],
  });
  const ctx = await browser.newContext({ viewport: { width: 900, height: 900 } });
  const page = await ctx.newPage();

  const errors = [];
  page.on("pageerror", (err) => errors.push("pageerror: " + err.message));
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push("console.error: " + msg.text());
  });

  await page.goto(url, { waitUntil: "load" });

  // Wait for __game to attach (engine init completes).
  await page.waitForFunction(() => typeof window.__game !== "undefined", null, { timeout: 10000 });

  // Wait a moment for the resolution change / first render to settle.
  await page.evaluate(async (nx) => {
    if (window.__game.engine === "gpu") {
      // The GPU sim is already at 200; switching is async.
    }
    if (nx !== 200) window.__game.setResolution(nx);
    await new Promise(r => setTimeout(r, 500));
  }, NX);

  // Reset prof counters, start sim, and run for WINDOW_SEC wall seconds.
  const result = await page.evaluate(async (seconds) => {
    window.__prof = { frames: 0, advanceMs: 0, renderMs: 0, lastTime: 0, nsLastFrame: 0 };
    const t0 = performance.now();
    const gt0 = window.__game.gameTime;
    window.__game.start();

    await new Promise((r) => setTimeout(r, seconds * 1000));

    const t1 = performance.now();
    const gt1 = window.__game.gameTime;
    return {
      engine: window.__game.engine,
      wallMs: t1 - t0,
      simSecAdvanced: gt1 - gt0,
      frames: window.__prof.frames,
      avgAdvanceMs: window.__prof.frames > 0 ? window.__prof.advanceMs / window.__prof.frames : null,
      avgRenderMs: window.__prof.frames > 0 ? window.__prof.renderMs / window.__prof.frames : null,
      nsLastFrame: window.__prof.nsLastFrame,  // CPU only: sim sub-steps in last frame
    };
  }, WINDOW_SEC);

  await browser.close();
  return { ...result, errors };
}

function fmt(n, unit = "", digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toFixed(digits) + unit;
}

(async () => {
  console.log(`\nProfiling SWE game at ${URL_BASE} (NX=${NX}, window=${WINDOW_SEC}s)\n`);
  for (const eng of ["cpu", "gpu"]) {
    process.stdout.write(`  ${eng.toUpperCase().padEnd(4)}: `);
    try {
      const r = await run(eng);
      const fps = (r.frames / (r.wallMs / 1000));
      const ratio = r.simSecAdvanced / (r.wallMs / 1000);
      console.log(
        `frames=${r.frames}  fps=${fmt(fps, "", 1).padEnd(5)}  ` +
        `simAdvanced=${fmt(r.simSecAdvanced, "s")}  ratio=${fmt(ratio, "×")}  ` +
        `advance=${fmt(r.avgAdvanceMs, "ms")}/frame  ` +
        `render=${fmt(r.avgRenderMs, "ms")}/frame` +
        (eng === "cpu" ? `  nsLastFrame=${r.nsLastFrame}` : "")
      );
      if (r.errors.length) {
        console.log("    errors:");
        for (const e of r.errors) console.log("      " + e);
      }
    } catch (e) {
      console.log("FAILED: " + e.message);
    }
  }
})();
