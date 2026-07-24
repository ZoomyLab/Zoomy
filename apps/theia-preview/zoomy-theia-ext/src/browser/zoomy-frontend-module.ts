import { ContainerModule, injectable, inject } from '@theia/core/shared/inversify';
import { CommandContribution, CommandRegistry, MenuContribution, MenuModelRegistry, URI } from '@theia/core';
import {
    FrontendApplicationContribution, OpenerService, open, CommonMenus,
    WidgetFactory, WidgetManager, ApplicationShell
} from '@theia/core/lib/browser';
import { StatusBar, StatusBarAlignment } from '@theia/core/lib/browser/status-bar';
import { FileService } from '@theia/filesystem/lib/browser/file-service';
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
import { ZoomyStartWidget, OPEN_EDITOR, OPEN_NOTEBOOK } from './start-page-widget';

const VIEW_TYPE = 'zoomy-notebook';
const NB_URI = new URI('file:///pyodide.ipynb');
const EDITOR_URI = new URI('file:///zoomy_model.py');
const START = { id: 'zoomy.start', label: 'Zoomy: Go to start page' };
const CMD_EDITOR = { id: OPEN_EDITOR, label: 'Zoomy: Open code editor' };
const CMD_NOTEBOOK = { id: OPEN_NOTEBOOK, label: 'Zoomy: Open Pyodide notebook' };

const SAMPLE_PY = `"""A Zoomy model, edited in a backend-less Theia editor.

This is the same symbolic model the notebook runs — kept here so the code
editor route has something real to show. Open the notebook to execute it on
the in-browser Pyodide kernel.
"""
import numpy as np
from zoomy_core.model.models import SME, Newtonian, NavierSlip, StressFree
import zoomy_core.model.boundary_conditions as BC

model = SME(
    level=2,                                   # two moments beyond the depth average
    parameters={"nu": 0.1, "lambda_s": 0.5},   # bulk viscosity + bed slip
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=BC.BoundaryConditions([
        BC.Wall(tag="left"), BC.Wall(tag="right"),
    ]),
)
print(model)
`;

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
    @inject(StatusBar) protected readonly statusBar: StatusBar;
    protected kernel: PyodideKernel;

    onStart(): void {
        console.log('ZOOMY onStart running');
        try { this.notebookService.registerNotebookSerializer(VIEW_TYPE, this.serializer); console.log('ZOOMY serializer ok'); } catch (e) { console.error('ZOOMY serializer FAIL', e); }
        try { this.typeRegistry.registerNotebookType({ type: VIEW_TYPE, displayName: 'Zoomy Notebook', selector: [{ filenamePattern: '*.ipynb' }] }, 'Zoomy'); console.log('ZOOMY type ok'); } catch (e) { console.error('ZOOMY type FAIL', e); }
        try { this.notebookService.markReady(); } catch { /* already ready */ }
        this.kernel = new PyodideKernel(this.notebookService, this.execService, m => console.log('[pyodide]', m));
        try { this.kernelService.registerKernel(this.kernel); console.log('ZOOMY kernel registered'); } catch (e) { console.error('ZOOMY kernel FAIL', e); }

        this.statusBar.setElement('zoomy.back', {
            text: '$(home) Zoomy start', tooltip: 'Back to the Zoomy start page',
            command: START.id, alignment: StatusBarAlignment.LEFT, priority: 5000,
        });

        this.openStart().catch(e => console.error('zoomy start', e));
        if (typeof location !== 'undefined' && /[?&]autorun/.test(location.search)) {
            setTimeout(() => this.openNotebook(true).catch(e => console.error('zoomy autorun', e)), 1500);
        }
    }

    protected async openStart(): Promise<void> {
        const w = await this.widgetManager.getOrCreateWidget(ZoomyStartWidget.ID);
        if (!w.isAttached) { this.shell.addWidget(w, { area: 'main' }); }
        this.shell.activateWidget(w.id);
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

    registerCommands(reg: CommandRegistry): void {
        reg.registerCommand(START, { execute: () => this.openStart() });
        reg.registerCommand(CMD_EDITOR, { execute: () => this.openEditor() });
        reg.registerCommand(CMD_NOTEBOOK, { execute: () => this.openNotebook() });
    }
    registerMenus(menus: MenuModelRegistry): void {
        menus.registerMenuAction(CommonMenus.FILE, { commandId: START.id, label: START.label });
        menus.registerMenuAction(CommonMenus.FILE, { commandId: CMD_NOTEBOOK.id, label: CMD_NOTEBOOK.label });
        menus.registerMenuAction(CommonMenus.FILE, { commandId: CMD_EDITOR.id, label: CMD_EDITOR.label });
    }
}

console.log('ZOOMY module evaluated');
export default new ContainerModule(bind => {
    console.log('ZOOMY ContainerModule binding');
    bind(IpynbSerializer).toSelf().inSingletonScope();
    // browser-only: the iframe output webview factory is unbound — supply a DOM one.
    // Must return the instance synchronously (Theia binds factory() as a constant
    // and constructs the notebook editor widget synchronously).
    bind(CellOutputWebviewFactory).toConstantValue((() => new DomOutputWebview()) as any);
    bind(ZoomyStartWidget).toSelf();
    bind(WidgetFactory).toDynamicValue(ctx => ({
        id: ZoomyStartWidget.ID,
        createWidget: () => ctx.container.get(ZoomyStartWidget),
    })).inSingletonScope();
    bind(ZoomyContribution).toSelf().inSingletonScope();
    bind(FrontendApplicationContribution).toService(ZoomyContribution);
    bind(CommandContribution).toService(ZoomyContribution);
    bind(MenuContribution).toService(ZoomyContribution);
});
