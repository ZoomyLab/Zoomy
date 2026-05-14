# SWE-game profile harness

Headless Chrome driver for the game; useful for fixing/profiling the
GPU engine without a UI.

## Setup

```
npm install --no-save playwright-core
```

Uses the system Chrome at `/usr/bin/google-chrome`; no big browser
download. Requires a local HTTP server serving the game on
`http://localhost:8765/` (e.g. `python3 -m http.server 8765` in
`web/playground/swe-game/`).

## Usage

```
node profile.mjs                 # default: 5 s window per engine, NX=200
node profile.mjs --seconds 10
node profile.mjs --nx 300
```

Output shows per engine: frames, fps, sim seconds advanced, sim/wall
ratio, average JS time per frame for advance / render, and any
console errors.

## How it works

The game exposes a `window.__game` testing facade when loaded with
`?profile=1`, plus a `window.__prof` counter. The driver navigates
to `?gpu=1&profile=1` or `?profile=1` (CPU), calls `__game.start()`,
waits N wall seconds, then reads `__prof`.

Note: headless Chrome heavily throttles Web Workers, so the CPU
numbers are not representative of real-device CPU performance. The
GPU numbers are accurate (system Chrome uses real GPU/ANGLE).
