# Zoomy — desktop app (Electron)

The **desktop** target of the one-GUI-everywhere plan: the backend-less Zoomy GUI
as a native app for Windows, macOS and Linux. A thin Electron shell loads the
GUI; its brain (`zoomy_cli` + the in-browser Pyodide worker) runs in the renderer
— no backend, no server.

## Run (dev)

```
cd apps/electron-app
npm install
npm start
```

## Package for Windows / macOS / Linux

```
npm run dist        # current OS
npm run dist:all    # -mwl (macOS + Windows + Linux; needs the right host/CI)
```

Output goes to `dist/` (`.exe`/NSIS, `.dmg`, `.AppImage`). Cross-compiling all
three from one host has the usual electron-builder caveats — the reliable path is
a CI matrix (GitHub Actions `runs-on: [macos, windows, ubuntu]`), which is the
next step for the release pipeline.

## Notes

- `ZOOMY_GUI_URL` env var overrides which GUI build to load (hosted by default;
  point at a local/offline build once roadmap #10 lands).
- The app injects COOP/COEP response headers so the renderer is cross-origin
  isolated → `SharedArrayBuffer` is available → the kernel's cooperative
  interrupt (Stop) works.
- Offline: bundle the `gui` static build into `files` and `loadFile`
  it instead of `loadURL` — a follow-up with #10.
