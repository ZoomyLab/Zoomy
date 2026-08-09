# Zoomy — VS Code extension

The Zoomy model-configuration GUI inside VS Code. This is the **local** target of
the one-GUI-everywhere plan: the same backend-less GUI that ships on the web,
wrapped in a VS Code Webview. The GUI's brain (`zoomy_cli` + the in-browser
Pyodide worker) runs entirely in the Webview — no VS Code backend, no server.

## Use

- Command palette → **Zoomy: Open model configuration**.
- The GUI opens in an editor tab. First kernel boot takes ~2–3 min (Pyodide +
  `zoomy-core`), then it's cached.

## Configuration

- `zoomy.guiUrl` — which GUI build to embed. Defaults to the hosted
  `https://zoomylab.github.io/Zoomy/gui/`. Point it at a local build
  (a `file://`/`http://localhost` path serving the `library/zoomy_gui` build output)
  for **offline** use.

## Build / install

```
cd apps/vscode-ext
npx @vscode/vsce package --allow-missing-repository -o zoomy-gui.vsix
code --install-extension zoomy-gui.vsix
```

## Notes / follow-ups

- This wraps the hosted GUI in a Webview (thin, always up to date). A fully
  self-contained offline `.vsix` would bundle the `gui` static build
  and the `gui/` assets and serve them from the extension via
  `asWebviewUri`/`localResourceRoots` — a follow-up once the offline build
  (roadmap #10) lands.
- The interrupt (Stop) feature needs cross-origin isolation (SharedArrayBuffer);
  in a Webview that requires the embedded page to be COOP/COEP-isolated (the
  hosted build already ships a service worker for this on the web).
