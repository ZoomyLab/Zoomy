import { ContainerModule, injectable, inject } from '@theia/core/shared/inversify';
import { CommandContribution, CommandRegistry, MenuContribution, MenuModelRegistry, MenuPath, MAIN_MENU_BAR, SelectionService, URI } from '@theia/core';
import { NavigatorContextMenu } from '@theia/navigator/lib/browser/navigator-contribution';
import {
    FrontendApplicationContribution, OpenerService, open, CommonMenus,
    WidgetFactory, WidgetManager, ApplicationShell, AbstractViewContribution, bindViewContribution,
    QuickInputService, codicon
} from '@theia/core/lib/browser';
import { OutlineViewService } from '@theia/outline-view/lib/browser/outline-view-service';
import { OutlineSymbolInformationNode } from '@theia/outline-view/lib/browser/outline-view-widget';
import { StatusBar, StatusBarAlignment } from '@theia/core/lib/browser/status-bar';
import { FileService, FileServiceContribution } from '@theia/filesystem/lib/browser/file-service';
import { RemoteFileServiceContribution } from '@theia/filesystem/lib/browser/remote-file-service-contribution';
import { WorkspaceService } from '@theia/workspace/lib/browser/workspace-service';
import { FileChangeType } from '@theia/filesystem/lib/common/files';
import { MemoryFileSystemProvider } from './memory-fs-provider';
import { getZoomyCli, onBackendsChanged } from './zoomy-cli-loader';
import * as monaco from '@theia/monaco-editor-core';
import { NotebookService } from '@theia/notebook/lib/browser';
import { CellKind } from '@theia/notebook/lib/common';
import { NotebookTypeRegistry } from '@theia/notebook/lib/browser/notebook-type-registry';
import { NotebookKernelService } from '@theia/notebook/lib/browser/service/notebook-kernel-service';
import { NotebookExecutionStateService } from '@theia/notebook/lib/browser/service/notebook-execution-state-service';
import { CellOutputWebviewFactory } from '@theia/notebook/lib/browser/renderers/cell-output-webview';
import { IpynbSerializer } from './ipynb-serializer';
import { PyodideKernel } from './pyodide-kernel';
import { NOTEBOOK_JSON } from './notebook-content';
import { DomOutputWebview } from './dom-output-webview';
import { ZoomyStartWidget } from './start-page-widget';
import { ZoomyModelConfigWidget } from './model-config-widget';
import { ZoomyViewWidget } from './zoomy-view-widget';
import { ZoomyParamsWidget } from './zoomy-params-widget';
import { ZoomySimOutputWidget } from './zoomy-sim-output-widget';
import { getPyodideClient, PyodideClient } from './pyodide-runtime';
import { registerZoomyCompletions } from './completion-provider';

const VIEW_TYPE = 'zoomy-notebook';
const NB_URI = new URI('file:///pyodide.ipynb');
const EDITOR_URI = new URI('file:///zoomy_model.py');
// A "Zoomy" menu in the top menu bar (next to Help), for backend + surfaces.
const ZOOMY_MENU: MenuPath = [...MAIN_MENU_BAR, '9_zoomy'];

const CMD = {
    openModelConfig: 'zoomy.openModelConfig',
    openEditor: 'zoomy.openEditor',
    openNotebook: 'zoomy.openNotebook',
    newCase: 'zoomy.newCase',
    run: 'zoomy.run',
    exportPy: 'zoomy.exportPy',
    exportIpynb: 'zoomy.exportIpynb',
    importCase: 'zoomy.importCase',
    saveProject: 'zoomy.saveProject',
    loadProject: 'zoomy.loadProject',
    connectBackend: 'zoomy.connectBackend',
};

const SAMPLE_PY = `"""A Zoomy model, edited in a backend-less Theia editor."""
import numpy as np
from zoomy_core.model.models import SME, Newtonian, NavierSlip, StressFree
import zoomy_core.model.boundary_conditions as BC

model = SME(
    level=2,
    parameters={"nu": 0.1, "lambda_s": 0.5},
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=BC.BoundaryConditions([BC.Wall(tag="left"), BC.Wall(tag="right")]),
)
print(model)
`;

/** Places the Zoomy view in the left activity bar (native slot, with an icon). */
@injectable()
export class ZoomyViewContribution extends AbstractViewContribution<ZoomyViewWidget> {
    constructor() {
        super({
            widgetId: ZoomyViewWidget.ID,
            widgetName: 'Zoomy',
            defaultWidgetOptions: { area: 'left', rank: 100 },
            toggleCommandId: 'zoomy.toggleView',
        });
    }
    async initializeLayout(): Promise<void> { await this.openView({ activate: false, reveal: true }); }
}

/** Places the "Zoomy Parameters" view in the RIGHT activity bar (docked but
 *  collapsed by default). The config widget reveals it when a card's Parameters
 *  button is clicked and collapses it on tab switch. */
@injectable()
export class ZoomyParamsViewContribution extends AbstractViewContribution<ZoomyParamsWidget> {
    constructor() {
        super({
            widgetId: ZoomyParamsWidget.ID,
            widgetName: 'Parameters',
            defaultWidgetOptions: { area: 'right', rank: 200 },
            toggleCommandId: 'zoomy.toggleParamsView',
        });
    }
    // Dock it in the right area (so its icon shows) without expanding the panel.
    async initializeLayout(): Promise<void> { await this.openView({ activate: false, reveal: false }); }
}

/** Places the "Simulation" console in the BOTTOM panel (docked, collapsed). It is
 *  auto-revealed when a simulation runs. */
@injectable()
export class ZoomySimOutputViewContribution extends AbstractViewContribution<ZoomySimOutputWidget> {
    constructor() {
        super({
            widgetId: ZoomySimOutputWidget.ID,
            widgetName: 'Simulation',
            defaultWidgetOptions: { area: 'bottom', rank: 300 },
            toggleCommandId: 'zoomy.toggleSimOutput',
        });
    }
    async initializeLayout(): Promise<void> { await this.openView({ activate: false, reveal: false }); }
}

/**
 * Registers our reliable in-memory + IndexedDB provider for the `file` scheme,
 * synchronously, at FileService init. Theia's browser-only default registers
 * `file` via RemoteFileServiceContribution, which only calls
 * `service.registerProvider('file', …)` AFTER the OPFS provider's `ready`
 * resolves — and OPFS fails to initialize in a blob worker on some browsers, so
 * `ready` rejects and `file` is NEVER registered (every read/write throws
 * ENOPRO, breaking the workspace). We claim `file` synchronously here and
 * neutralize the Remote contribution (below) so nothing conflicts or boots OPFS.
 */
@injectable()
export class ZoomyFileServiceContribution implements FileServiceContribution {
    @inject(MemoryFileSystemProvider) protected readonly provider: MemoryFileSystemProvider;
    registerFileSystemProviders(service: FileService): void {
        try { service.registerProvider('file', this.provider as any); }
        catch (e) { console.error('[zoomy-fs] register file provider failed', e); }
    }
}

/** No-op replacement for RemoteFileServiceContribution: keeps OPFS from ever
 *  constructing (its @postConstruct init is what fails) and from double-
 *  registering the `file` scheme. */
@injectable()
export class NoopFileServiceContribution implements FileServiceContribution {
    registerFileSystemProviders(): void { /* intentionally empty */ }
}

@injectable()
class ZoomyContribution implements FrontendApplicationContribution, CommandContribution, MenuContribution {
    @inject(NotebookService) protected readonly notebookService: NotebookService;
    @inject(NotebookTypeRegistry) protected readonly typeRegistry: NotebookTypeRegistry;
    @inject(NotebookKernelService) protected readonly kernelService: NotebookKernelService;
    @inject(NotebookExecutionStateService) protected readonly execService: NotebookExecutionStateService;
    @inject(OpenerService) protected readonly openerService: OpenerService;
    @inject(FileService) protected readonly fileService: FileService;
    @inject(IpynbSerializer) protected readonly serializer: IpynbSerializer;
    @inject(WidgetManager) protected readonly widgetManager: WidgetManager;
    @inject(ApplicationShell) protected readonly shell: ApplicationShell;
    @inject(QuickInputService) protected readonly quickInput: QuickInputService;
    @inject(SelectionService) protected readonly selectionService: SelectionService;
    @inject(StatusBar) protected readonly statusBar: StatusBar;
    @inject(WorkspaceService) protected readonly workspaceService: WorkspaceService;
    protected kernel: PyodideKernel;
    protected client: PyodideClient;

    onStart(): void {
        this.client = getPyodideClient(m => console.log('[pyodide]', m));
        try { this.notebookService.registerNotebookSerializer(VIEW_TYPE, this.serializer); } catch (e) { console.error('ZOOMY serializer FAIL', e); }
        try { this.typeRegistry.registerNotebookType({ type: VIEW_TYPE, displayName: 'Zoomy Notebook', selector: [{ filenamePattern: '*.ipynb' }] }, 'Zoomy'); } catch (e) { console.error('ZOOMY type FAIL', e); }
        try { this.notebookService.markReady(); } catch { /* already ready */ }
        this.kernel = new PyodideKernel(this.notebookService, this.execService, this.client, m => console.log('[pyodide]', m));
        try { this.kernelService.registerKernel(this.kernel); } catch (e) { console.error('ZOOMY kernel FAIL', e); }
        try { registerZoomyCompletions(this.client, m => console.log('[pyodide]', m)); } catch (e) { console.error('ZOOMY completions FAIL', e); }

        // #10 offline + cross-origin isolation service worker.
        try { if ('serviceWorker' in navigator) { navigator.serviceWorker.register('sw.js').catch(() => {}); } } catch { /* ignore */ }

        this.setBackendStatus([]);
        // Reflect connect/disconnect (from anywhere) in the status bar.
        onBackendsChanged(() => { this.mc().then(w => this.setBackendStatus(w.connectedTags || [])).catch(() => { /* ignore */ }); });
        // Keep every case's case.py and case.ipynb in sync on save.
        this.startCaseSync();
        // Make the right-hand Outline jump between a case.py's sections.
        this.registerCaseOutline();
        // Open file:///zoomy as the workspace (so the Explorer shows the cases as
        // folders), then land on the model configuration. openWorkspace reloads
        // the window once with preserveWindow; guarded so it can never loop.
        this.ensureWorkspaceThenOpen();
        if (typeof location !== 'undefined' && /[?&]autorun/.test(location.search)) {
            setTimeout(() => this.openNotebook(true).catch(e => console.error('zoomy autorun', e)), 1500);
        }
    }

    /** Root of the single-source-of-truth project (cases live under cases/). */
    protected static readonly WORKSPACE_ROOT = 'file:///zoomy';
    protected async ensureWorkspaceThenOpen(): Promise<void> {
        const root = ZoomyContribution.WORKSPACE_ROOT;
        try {
            await this.workspaceService.ready;
            const isRoot = this.workspaceService.tryGetRoots().some(r => r.resource.toString() === root);
            const tried = (() => { try { return !!sessionStorage.getItem('zoomy-ws-open-tried'); } catch { return false; } })();
            if (!isRoot && !tried) {
                try { sessionStorage.setItem('zoomy-ws-open-tried', '1'); } catch { /* ignore */ }
                const uri = new URI(root);
                if (!(await this.fileService.exists(uri))) { await this.fileService.createFolder(uri); }
                const cases = uri.resolve('cases');
                if (!(await this.fileService.exists(cases))) { await this.fileService.createFolder(cases); }
                // preserveWindow (auto when nothing is open) reloads THIS window
                // with the workspace set; the reload re-enters onStart with the
                // workspace already open, so we fall through to openModelConfig.
                this.workspaceService.open(uri);
                return;
            }
        } catch (e) { console.warn('zoomy workspace open', e); }
        // Land directly on the model configuration, in the classical IDE layout.
        this.openModelConfig().catch(e => console.error('zoomy open config', e));
    }

    /** Register a Monaco document-symbol provider so the native Outline shows a
     *  case.py's sections (Model / Mesh / Settings / Run / Visualization) parsed
     *  from its `# %% … zoomy={…}` cell markers — click to jump to a section. */
    protected registerCaseOutline(): void {
        const m: any = monaco;
        if (!m?.languages?.registerDocumentSymbolProvider) { console.warn('zoomy outline: monaco languages unavailable'); return; }
        const cap = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
        const SECTIONS = ['meta', 'model', 'mesh', 'solver', 'settings', 'run', 'visualization'];
        m.languages.registerDocumentSymbolProvider('python', {
            displayName: 'Zoomy Case',
            provideDocumentSymbols: (model: any) => {
                const path = String(model?.uri?.path || '');
                if (!/\/cases\/[^/]+\/case\.py$/.test(path)) { return []; }
                const lines = String(model.getValue()).split('\n');
                const seen = new Map<string, number>();
                lines.forEach((line, i) => {
                    const m = /#\s*%%.*zoomy=(\{.*\})\s*$/.exec(line);
                    if (!m) { return; }
                    let meta: any; try { meta = JSON.parse(m[1]); } catch { return; }
                    const sec = meta.section || (SECTIONS.includes(meta.role) ? meta.role : null);
                    if (sec && sec !== 'meta' && !seen.has(sec)) { seen.set(sec, i); }
                });
                const kind = m.languages.SymbolKind.Module;
                return [...seen.entries()].map(([sec, i]) => {
                    const range = new m.Range(i + 1, 1, i + 1, Math.max(1, (lines[i] || '').length));
                    return { name: cap(sec), detail: '', kind, range, selectionRange: range, tags: [] };
                });
            },
        });
    }

    /** Extract the .py source (jupytext) from a case.ipynb's cells. */
    protected pyFromIpynb(ipynb: string): string {
        const nb = JSON.parse(ipynb);
        return (nb.cells || []).map((c: any) => (Array.isArray(c.source) ? c.source.join('') : (c.source || ''))).join('\n\n');
    }

    /** Keep every case's case.py and case.ipynb in sync: whenever one is saved,
     *  regenerate the other from it (parseCase → exportCase). Writes only when the
     *  content actually changes, and an echo-guard skips our own writes, so it can
     *  never loop. Only syncs a pair that already exists (won't auto-create). */
    protected async startCaseSync(): Promise<void> {
        let cli: any;
        try { cli = await getZoomyCli(); } catch (e) { console.warn('zoomy sync: cli unavailable', e); return; }
        const echo = new Map<string, string>(); // path → content we just wrote
        const rx = /^\/zoomy\/cases\/[^/]+\/case\.(py|ipynb)$/;
        this.fileService.onDidFilesChange(async event => {
            for (const change of event.changes) {
                if (change.type === FileChangeType.DELETED) { continue; }
                const uri = change.resource;
                const p = uri.path.toString();
                const m = rx.exec(p);
                if (!m) { continue; }
                let content: string;
                try { content = (await this.fileService.read(uri)).value; } catch { continue; }
                if (echo.get(p) === content) { echo.delete(p); continue; } // our own write
                const isPy = m[1] === 'py';
                const sibling = uri.parent.resolve(isPy ? 'case.ipynb' : 'case.py');
                if (!(await this.fileService.exists(sibling))) { continue; } // only sync an existing pair
                try {
                    const spec = cli.parseCase(isPy ? content : this.pyFromIpynb(content));
                    const next = cli.exportCase(spec, isPy ? 'ipynb' : 'py');
                    const cur = (await this.fileService.read(sibling)).value;
                    if (cur === next) { continue; } // already in sync — stop the chain
                    echo.set(sibling.path.toString(), next);
                    await this.fileService.write(sibling, next);
                } catch (e) { console.warn('zoomy case sync', p, e); }
            }
        });
    }

    protected async mc(): Promise<ZoomyModelConfigWidget> {
        const w = (await this.widgetManager.getOrCreateWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget;
        // Reflect connected backends in the status bar (the "connected backend"
        // indicator brought over from the old GUI, in a native, portable slot).
        if (!w.onBackendsChanged) { w.onBackendsChanged = tags => this.setBackendStatus(tags); this.setBackendStatus(w.connectedTags || []); }
        // Let the config widget reconcile the right-hand Parameters panel.
        if (!w.paramsPanel) { w.paramsPanel = { sync: () => { this.syncParams(); } }; }
        // Let the config widget reveal the bottom "Simulation" console on Run.
        if (!w.simPanel) { w.simPanel = { reveal: () => this.revealSimOutput() }; }
        return w;
    }
    /** Reveal + expand the bottom "Simulation" console. */
    protected async revealSimOutput(): Promise<void> {
        const sw = await this.widgetManager.getOrCreateWidget(ZoomySimOutputWidget.ID);
        if (!sw.isAttached) { await this.shell.addWidget(sw, { area: 'bottom' }); }
        await this.shell.activateWidget(sw.id);
        this.shell.expandPanel('bottom');
    }
    /** Reconcile the right-hand Parameters panel to the widget's desired state.
     *  Serialized + last-write-wins: activateWidget can take ~2s, so rapid
     *  open→tab-switch must not leave the panel expanded after a collapse. */
    protected paramsSyncing = false;
    protected paramsSyncPending = false;
    protected async syncParams(): Promise<void> {
        if (this.paramsSyncing) { this.paramsSyncPending = true; return; }
        this.paramsSyncing = true;
        try {
            const pw = await this.widgetManager.getOrCreateWidget(ZoomyParamsWidget.ID);
            do {
                this.paramsSyncPending = false;
                const w = (await this.widgetManager.getWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget | undefined;
                const shouldOpen = !!w?.hasActiveParams();
                if (shouldOpen) {
                    if (!pw.isAttached) { await this.shell.addWidget(pw, { area: 'right' }); }
                    await this.shell.activateWidget(pw.id);
                    // Re-check after the (slow) activate — a collapse may have raced in.
                    if (!this.paramsSyncPending && w?.hasActiveParams()) { this.shell.expandPanel('right'); }
                } else {
                    await this.shell.collapsePanel('right');
                }
            } while (this.paramsSyncPending);
        } catch (e) { console.warn('zoomy params sync', e); } finally { this.paramsSyncing = false; }
    }
    protected setBackendStatus(tags: string[]): void {
        const connected = tags.length > 0;
        this.statusBar.setElement('zoomy.backend', {
            text: connected ? '$(server) ' + tags.join(', ') : '$(plug) Connect backend',
            tooltip: connected
                ? 'Connected Zoomy backends: ' + tags.join(', ') + '. Click to connect another; manage/disconnect in the Zoomy panel.'
                : 'No backend — running in-browser (Pyodide). Click to connect a compute backend.',
            command: CMD.connectBackend, alignment: StatusBarAlignment.LEFT, priority: 6000,
        });
    }
    protected async openModelConfig(): Promise<void> {
        const w = await this.mc();
        if (!w.isAttached) { this.shell.addWidget(w, { area: 'main' }); }
        this.shell.activateWidget(w.id);
    }
    protected async newCase(): Promise<void> {
        const name = await this.quickInput.input({ prompt: 'New case name', placeHolder: 'dam_break_1d' });
        if (name && name.trim()) { const w = await this.mc(); await this.openModelConfig(); await w.newCase(name); }
    }
    protected async openEditor(): Promise<void> {
        await this.fileService.write(EDITOR_URI, SAMPLE_PY);
        await open(this.openerService, EDITOR_URI);
    }
    protected async openNotebook(run = false): Promise<void> {
        await this.fileService.write(NB_URI, NOTEBOOK_JSON);
        await open(this.openerService, NB_URI);
        try { this.kernelService.selectKernelForNotebook(this.kernel, { uri: NB_URI, viewType: VIEW_TYPE }); } catch (e) { console.warn('kernel select', e); }
        if (run) {
            await new Promise(r => setTimeout(r, 1200));
            const model = this.notebookService.getNotebookEditorModel(NB_URI);
            if (model) {
                const handles = model.cells.filter(c => c.cellKind === CellKind.Code).map(c => c.handle);
                await this.kernel.executeNotebookCellsRequest(NB_URI, handles);
            }
        }
    }
    protected async connectBackend(): Promise<void> {
        const w = await this.mc();
        const url = await this.quickInput.input({ prompt: 'Connect a Zoomy backend by URL', value: w.backendUrl, placeHolder: 'http://localhost:8080' });
        if (url) { w.backendUrl = url; await w.connectBackend(); }
    }
    /** Explorer context command: open the selected case file in the configurator. */
    protected selectedUri(): URI | undefined {
        const sel: any = this.selectionService.selection;
        const node = Array.isArray(sel) ? sel[0] : sel;
        const u = node?.uri || node?.fileStat?.resource;
        return u ? new URI(u.toString()) : undefined;
    }
    protected async openCaseInConfigurator(): Promise<void> {
        const uri = this.selectedUri();
        if (!uri) { return; }
        const path = uri.path.toString();
        if (!/\.(py|ipynb)$/.test(path)) { return; }
        try {
            const content = await this.fileService.read(uri);
            const w = await this.mc();
            w.openCaseText(content.value, path.endsWith('.ipynb'), uri.path.base);
            await this.openModelConfig();
        } catch (e) { console.error('openCaseInConfigurator', e); }
    }

    registerCommands(reg: CommandRegistry): void {
        reg.registerCommand({ id: CMD.openModelConfig, label: 'Zoomy: Open model configuration' }, { execute: () => this.openModelConfig() });
        reg.registerCommand({ id: CMD.openEditor, label: 'Zoomy: Open code editor' }, { execute: () => this.openEditor() });
        reg.registerCommand({ id: CMD.openNotebook, label: 'Zoomy: Open Pyodide notebook' }, { execute: () => this.openNotebook() });
        reg.registerCommand({ id: CMD.newCase, label: 'Zoomy: New case…' }, { execute: () => this.newCase() });
        reg.registerCommand({ id: 'zoomy.openNamedCase' }, { execute: async (name: string) => { await this.openModelConfig(); (await this.mc()).openCaseByName(name); } });
        reg.registerCommand({ id: 'zoomy.removeCase' }, { execute: async (name: string) => { if (name) { (await this.mc()).removeCase(name); } } });
        reg.registerCommand({ id: 'zoomy.duplicateCase' }, { execute: async (name: string) => { if (name) { await this.openModelConfig(); (await this.mc()).duplicateCase(name); } } });
        reg.registerCommand({ id: 'zoomy.coupleCases' }, { execute: async (names: string[]) => { if (Array.isArray(names) && names.length >= 2) { await this.openModelConfig(); (await this.mc()).coupleCases(names); } } });
        reg.registerCommand({ id: 'zoomy.decoupleCase' }, { execute: async (name: string) => { if (name) { await this.openModelConfig(); (await this.mc()).decoupleCase(name); } } });
        reg.registerCommand({ id: 'zoomy.openCoupling' }, { execute: async (name: string) => { if (name) { (await this.mc()).openCoupling(name); } } });
        reg.registerCommand({ id: 'zoomy.dissolveCoupling' }, { execute: async (name: string) => { if (name) { await this.openModelConfig(); (await this.mc()).dissolveCoupling(name); } } });
        reg.registerCommand({ id: 'zoomy.runCoupling' }, { execute: async (name: string) => { if (name) { await this.openModelConfig(); (await this.mc()).runCoupling(name); } } });
        reg.registerCommand({ id: 'zoomy.renameCase' }, { execute: async (name: string) => { if (!name) { return; } const leaf = name.split('/').pop(); const next = await this.quickInput.input({ prompt: 'Rename case', value: leaf, placeHolder: leaf }); if (next && next.trim() && next.trim() !== leaf) { await this.openModelConfig(); (await this.mc()).renameCase(name, next); } } });
        reg.registerCommand({ id: CMD.run, label: 'Zoomy: Run simulation' }, { execute: async () => { await this.openModelConfig(); (await this.mc()).runAssembly(); } });
        reg.registerCommand({ id: 'zoomy.openInNotebook', label: 'Zoomy: Open case in Notebook Mode' }, { execute: async () => { await this.openModelConfig(); (await this.mc()).openInNotebook(); } });
        reg.registerCommand({ id: CMD.exportPy, label: 'Zoomy: Export case (.py)' }, { execute: async () => (await this.mc()).exportCase('py') });
        reg.registerCommand({ id: CMD.exportIpynb, label: 'Zoomy: Export case (.ipynb)' }, { execute: async () => (await this.mc()).exportCase('ipynb') });
        reg.registerCommand({ id: CMD.importCase, label: 'Zoomy: Import case…' }, { execute: async () => (await this.mc()).importCase() });
        reg.registerCommand({ id: CMD.saveProject, label: 'Zoomy: Save project' }, { execute: async () => (await this.mc()).saveProject() });
        reg.registerCommand({ id: CMD.loadProject, label: 'Zoomy: Load project' }, { execute: async () => (await this.mc()).loadProject() });
        reg.registerCommand({ id: 'zoomy.generateGuiLink', label: 'Zoomy: Generate GUI link' }, { execute: async () => {
            const asset = await this.quickInput.input({ prompt: 'Asset URL for the GUI link (a project .zip URL, or zenodo:<id>)', placeHolder: 'https://…/project.zip   or   zenodo:1234567' });
            if (!asset || !asset.trim()) { return; }
            // GUI base = origin + pathname with any query/hash stripped, so the link
            // works on the deployed …/Zoomy/theia-preview/ path AND locally.
            let baseUrl = '';
            try { baseUrl = location.origin + location.pathname; } catch { /* non-browser */ }
            const link = baseUrl + '?project=' + encodeURIComponent(asset.trim());
            try { await navigator.clipboard.writeText(link); } catch { /* clipboard may be blocked */ }
            // Reuses the ?project= param the GUI already consumes (loadProjectFromUrl).
            await this.quickInput.input({ prompt: 'GUI link — copied to clipboard (Ctrl/Cmd-C to copy again)', value: link });
        } });
        reg.registerCommand({ id: CMD.connectBackend, label: 'Zoomy: Connect backend…' }, { execute: () => this.connectBackend() });
        reg.registerCommand({ id: 'zoomy.disconnectBackend', label: 'Zoomy: Disconnect backend' }, { execute: async (tag: string) => { if (tag) { (await this.mc()).disconnectBackend(tag); } } });
        reg.registerCommand({ id: 'zoomy.scanBackends', label: 'Zoomy: Scan for local backends' }, { execute: async () => { await this.openModelConfig(); (await this.mc()).scanBackends(); } });
        reg.registerCommand({ id: 'zoomy.openCaseFile', label: 'Zoomy: Open case.py in editor' }, { execute: async () => { await this.openModelConfig(); (await this.mc()).editCardFile(); } });
        reg.registerCommand({ id: 'zoomy.openCaseHere', label: 'Open in model configurator' }, {
            execute: () => this.openCaseInConfigurator(),
            isVisible: () => { const u = this.selectedUri(); return !!u && /\.(py|ipynb)$/.test(u.path.toString()); },
        });
    }
    registerMenus(menus: MenuModelRegistry): void {
        // A top-level "Zoomy" menu next to Help.
        menus.registerSubmenu(ZOOMY_MENU, 'Zoomy');
        menus.registerMenuAction([...ZOOMY_MENU, '1_config'], { commandId: CMD.newCase, label: 'New case…' });
        menus.registerMenuAction([...ZOOMY_MENU, '1_config'], { commandId: CMD.openModelConfig, label: 'Model configuration' });
        menus.registerMenuAction([...ZOOMY_MENU, '1_config'], { commandId: 'zoomy.openInNotebook', label: 'Open in Notebook Mode' });
        menus.registerMenuAction([...ZOOMY_MENU, '1_config'], { commandId: CMD.run, label: 'Run simulation' });
        menus.registerMenuAction([...ZOOMY_MENU, '2_case'], { commandId: CMD.exportPy, label: 'Export case (.py)' });
        menus.registerMenuAction([...ZOOMY_MENU, '2_case'], { commandId: CMD.exportIpynb, label: 'Export case (.ipynb)' });
        menus.registerMenuAction([...ZOOMY_MENU, '2_case'], { commandId: CMD.importCase, label: 'Import case…' });
        menus.registerMenuAction([...ZOOMY_MENU, '3_project'], { commandId: CMD.saveProject, label: 'Save project' });
        menus.registerMenuAction([...ZOOMY_MENU, '3_project'], { commandId: CMD.loadProject, label: 'Load project' });
        menus.registerMenuAction([...ZOOMY_MENU, '4_backend'], { commandId: CMD.connectBackend, label: 'Connect backend…' });
        menus.registerMenuAction(CommonMenus.FILE, { commandId: CMD.openNotebook, label: 'Zoomy: Open Pyodide notebook' });
        menus.registerMenuAction(CommonMenus.FILE, { commandId: CMD.openEditor, label: 'Zoomy: Open code editor' });
        // Right-click a .py/.ipynb in the Explorer → Open in model configurator.
        menus.registerMenuAction(NavigatorContextMenu.NAVIGATION, { commandId: 'zoomy.openCaseHere', label: 'Open in model configurator', order: 'z' });
    }
}

/** Outline for the .ipynb notebook. Theia's built-in NotebookOutlineContribution
 *  only fires on its own focus tracking (which doesn't trigger for this custom
 *  notebook — hence "cannot provide outline information"). This publishes an
 *  outline straight from the active notebook's cells (duck-typed via `.model.
 *  cells`), labelled by each markdown cell's heading, so the .ipynb gets the
 *  same Model / Mesh / … outline as the .py editor. */
@injectable()
class ZoomyNotebookOutlineContribution implements FrontendApplicationContribution {
    @inject(ApplicationShell) protected readonly shell: ApplicationShell;
    @inject(OutlineViewService) protected readonly outline: OutlineViewService;

    onStart(): void {
        this.shell.onDidChangeActiveWidget(() => this.refresh());
        this.shell.onDidChangeCurrentWidget(() => this.refresh());
    }
    protected refresh(): void {
        try {
            const w: any = this.shell.activeWidget || this.shell.currentWidget;
            const cells: any[] = w?.model?.cells;
            if (!Array.isArray(cells) || !cells.length) { return; }   // not a notebook → leave others' outline alone
            this.outline.publish(cells.map((c, i) => this.node(c, i)));
        } catch { /* ignore */ }
    }
    protected node(cell: any, i: number): OutlineSymbolInformationNode {
        const src = String(cell?.source || '');
        const md = cell?.cellKind === CellKind.Markup;
        let label: string;
        if (md) { const m = src.match(/^\s*#{1,6}\s+(.+?)\s*$/m); label = (m ? m[1] : (src.split('\n').find((l: string) => l.trim()) || 'Markdown')).slice(0, 60); }
        else { label = (src.split('\n').find((l: string) => l.trim()) || 'Code').slice(0, 60); }
        return {
            id: 'zoomy-nb-cell-' + i,
            name: label,
            iconClass: codicon(md ? 'markdown' : 'symbol-namespace'),
            children: [], parent: undefined, selected: false, expanded: false,
        } as OutlineSymbolInformationNode;
    }
}

console.log('ZOOMY module evaluated');
export default new ContainerModule((bind, _unbind, isBound, rebind) => {
    // Replace Theia's OPFS filesystem provider (fails to init in a blob worker on
    // some browsers → breaks the whole FileService/workspace) with a reliable
    // in-memory + IndexedDB provider registered synchronously for the `file`
    // scheme. This extension loads after @theia/filesystem, so the rebind wins.
    bind(MemoryFileSystemProvider).toSelf().inSingletonScope();
    bind(ZoomyFileServiceContribution).toSelf().inSingletonScope();
    bind(FileServiceContribution).toService(ZoomyFileServiceContribution);
    // Neutralize the OPFS/remote contribution so it neither double-registers
    // `file` nor constructs the OPFS provider (whose @postConstruct init fails).
    if (isBound(RemoteFileServiceContribution)) { rebind(RemoteFileServiceContribution).to(NoopFileServiceContribution as any).inSingletonScope(); }

    bind(IpynbSerializer).toSelf().inSingletonScope();
    // browser-only: the iframe output webview factory is unbound — supply a DOM one.
    bind(CellOutputWebviewFactory).toConstantValue((() => new DomOutputWebview()) as any);
    // The Zoomy activity-bar view (left panel).
    bind(ZoomyViewWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyViewWidget.ID, createWidget: () => ctx.container.get(ZoomyViewWidget) })).inSingletonScope();
    bindViewContribution(bind, ZoomyViewContribution);
    bind(FrontendApplicationContribution).toService(ZoomyViewContribution);
    // The right-hand "Zoomy Parameters" panel.
    bind(ZoomyParamsWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyParamsWidget.ID, createWidget: () => ctx.container.get(ZoomyParamsWidget) })).inSingletonScope();
    bindViewContribution(bind, ZoomyParamsViewContribution);
    bind(FrontendApplicationContribution).toService(ZoomyParamsViewContribution);
    // The bottom "Simulation" console.
    bind(ZoomySimOutputWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomySimOutputWidget.ID, createWidget: () => ctx.container.get(ZoomySimOutputWidget) })).inSingletonScope();
    bindViewContribution(bind, ZoomySimOutputViewContribution);
    bind(FrontendApplicationContribution).toService(ZoomySimOutputViewContribution);
    // Kept but no longer the landing surface.
    bind(ZoomyStartWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyStartWidget.ID, createWidget: () => ctx.container.get(ZoomyStartWidget) })).inSingletonScope();
    bind(ZoomyModelConfigWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyModelConfigWidget.ID, createWidget: () => ctx.container.get(ZoomyModelConfigWidget) })).inSingletonScope();
    bind(ZoomyContribution).toSelf().inSingletonScope();
    bind(FrontendApplicationContribution).toService(ZoomyContribution);
    bind(ZoomyNotebookOutlineContribution).toSelf().inSingletonScope();
    bind(FrontendApplicationContribution).toService(ZoomyNotebookOutlineContribution);
    bind(CommandContribution).toService(ZoomyContribution);
    bind(MenuContribution).toService(ZoomyContribution);
});
