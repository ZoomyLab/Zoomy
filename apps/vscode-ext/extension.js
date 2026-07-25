// Zoomy VS Code extension: opens the model-configuration GUI in a Webview.
//
// The GUI's brain (zoomy_cli + the in-browser Pyodide worker) runs entirely in
// the Webview — no VS Code backend, no server. This is the "local" target of the
// one-GUI-everywhere plan: the same GUI that ships on the web, wrapped in VS Code.
//
// By default it embeds the hosted GUI (config `zoomy.guiUrl`); point that at a
// local build (e.g. a file:// or http://localhost path serving the theia-preview
// output) for offline use.
const vscode = require('vscode');

function activate(context) {
    context.subscriptions.push(
        vscode.commands.registerCommand('zoomy.openGui', () => {
            const panel = vscode.window.createWebviewPanel(
                'zoomyGui', 'Zoomy — model configuration', vscode.ViewColumn.Active,
                { enableScripts: true, retainContextWhenHidden: true }
            );
            const url = vscode.workspace.getConfiguration('zoomy').get('guiUrl')
                || 'https://zoomylab.github.io/Zoomy/theia-preview/';
            panel.webview.html = pageHtml(url);
        })
    );
}

function pageHtml(url) {
    // Embed the GUI in a full-bleed iframe. GitHub Pages does not send
    // X-Frame-Options, so framing is allowed; the iframe context still reaches
    // the Pyodide CDN + PyPI for the in-browser kernel.
    const safe = String(url).replace(/"/g, '&quot;');
    return `<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
  content="default-src 'none'; frame-src ${safe} https:; style-src 'unsafe-inline';">
<style>html,body,iframe{margin:0;padding:0;border:0;width:100%;height:100vh;overflow:hidden}</style>
</head><body>
<iframe src="${safe}" allow="cross-origin-isolated" referrerpolicy="no-referrer"></iframe>
</body></html>`;
}

function deactivate() { }
module.exports = { activate, deactivate };
