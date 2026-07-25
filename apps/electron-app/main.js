// Zoomy Electron app — the "desktop" target of the one-GUI-everywhere plan.
//
// A thin Electron shell that loads the backend-less Zoomy GUI. The GUI's brain
// (zoomy_cli + the in-browser Pyodide worker) runs in the renderer — no backend.
// electron-builder packages this for Windows, macOS and Linux.
const { app, BrowserWindow, session } = require('electron');

const GUI_URL = process.env.ZOOMY_GUI_URL || 'https://zoomylab.github.io/Zoomy/theia-preview/';

// Enable SharedArrayBuffer (used by the kernel's cooperative interrupt) by
// making the renderer cross-origin isolated: inject COOP/COEP on responses.
function enableCrossOriginIsolation() {
    session.defaultSession.webRequest.onHeadersReceived((details, cb) => {
        cb({
            responseHeaders: {
                ...details.responseHeaders,
                'Cross-Origin-Opener-Policy': ['same-origin'],
                'Cross-Origin-Embedder-Policy': ['require-corp'],
            },
        });
    });
}

function createWindow() {
    const win = new BrowserWindow({
        width: 1440, height: 920,
        title: 'Zoomy',
        webPreferences: { contextIsolation: true, nodeIntegration: false },
    });
    win.removeMenu();
    win.loadURL(GUI_URL);
}

app.whenReady().then(() => {
    enableCrossOriginIsolation();
    createWindow();
    app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) { createWindow(); } });
});

app.on('window-all-closed', () => { if (process.platform !== 'darwin') { app.quit(); } });
