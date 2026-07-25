import { ContainerModule, injectable, inject } from '@theia/core/shared/inversify';
import { CommandContribution, CommandRegistry, MenuContribution, MenuModelRegistry, MenuPath, MAIN_MENU_BAR, SelectionService, URI } from '@theia/core';
import { NavigatorContextMenu } from '@theia/navigator/lib/browser/navigator-contribution';
import {
    FrontendApplicationContribution, OpenerService, open, CommonMenus,
    WidgetFactory, WidgetManager, ApplicationShell, AbstractViewContribution, bindViewContribution,
    QuickInputService
} from '@theia/core/lib/browser';
import { StatusBar, StatusBarAlignment } from '@theia/core/lib/browser/status-bar';
import { FileService, FileServiceContribution } from '@theia/filesystem/lib/browser/file-service';
import { RemoteFileServiceContribution } from '@theia/filesystem/lib/browser/remote-file-service-contribution';
import { WorkspaceService } from '@theia/workspace/lib/browser/workspace-service';
import { MemoryFileSystemProvider } from './memory-fs-provider';
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

    protected async mc(): Promise<ZoomyModelConfigWidget> {
        const w = (await this.widgetManager.getOrCreateWidget(ZoomyModelConfigWidget.ID)) as ZoomyModelConfigWidget;
        // Reflect connected backends in the status bar (the "connected backend"
        // indicator brought over from the old GUI, in a native, portable slot).
        if (!w.onBackendsChanged) { w.onBackendsChanged = tags => this.setBackendStatus(tags); this.setBackendStatus(w.connectedTags || []); }
        return w;
    }
    protected setBackendStatus(tags: string[]): void {
        this.statusBar.setElement('zoomy.backend', {
            text: tags.length ? '$(server) ' + tags.join(', ') : '$(server) no backend',
            tooltip: tags.length ? 'Connected Zoomy backends: ' + tags.join(', ') : 'No backend connected — running in-browser (Pyodide). Click to connect.',
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
        reg.registerCommand({ id: CMD.run, label: 'Zoomy: Run simulation' }, { execute: async () => { await this.openModelConfig(); (await this.mc()).runAssembly(); } });
        reg.registerCommand({ id: CMD.exportPy, label: 'Zoomy: Export case (.py)' }, { execute: async () => (await this.mc()).exportCase('py') });
        reg.registerCommand({ id: CMD.exportIpynb, label: 'Zoomy: Export case (.ipynb)' }, { execute: async () => (await this.mc()).exportCase('ipynb') });
        reg.registerCommand({ id: CMD.importCase, label: 'Zoomy: Import case…' }, { execute: async () => (await this.mc()).importCase() });
        reg.registerCommand({ id: CMD.saveProject, label: 'Zoomy: Save project' }, { execute: async () => (await this.mc()).saveProject() });
        reg.registerCommand({ id: CMD.loadProject, label: 'Zoomy: Load project' }, { execute: async () => (await this.mc()).loadProject() });
        reg.registerCommand({ id: CMD.connectBackend, label: 'Zoomy: Connect backend…' }, { execute: () => this.connectBackend() });
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
    // Kept but no longer the landing surface.
    bind(ZoomyStartWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyStartWidget.ID, createWidget: () => ctx.container.get(ZoomyStartWidget) })).inSingletonScope();
    bind(ZoomyModelConfigWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({ id: ZoomyModelConfigWidget.ID, createWidget: () => ctx.container.get(ZoomyModelConfigWidget) })).inSingletonScope();
    bind(ZoomyContribution).toSelf().inSingletonScope();
    bind(FrontendApplicationContribution).toService(ZoomyContribution);
    bind(CommandContribution).toService(ZoomyContribution);
    bind(MenuContribution).toService(ZoomyContribution);
});
