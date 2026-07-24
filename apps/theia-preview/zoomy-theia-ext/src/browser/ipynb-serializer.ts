import { injectable } from '@theia/core/shared/inversify';
import { BinaryBuffer } from '@theia/core/lib/common/buffer';
import { NotebookSerializer } from '@theia/notebook/lib/browser';
import { CellKind, NotebookData, TransientOptions } from '@theia/notebook/lib/common';

/** Minimal Jupyter .ipynb <-> Theia NotebookData serializer (native, no plugin host). */
@injectable()
export class IpynbSerializer implements NotebookSerializer {
    readonly options: TransientOptions = { transientOutputs: false, transientCellMetadata: {}, transientDocumentMetadata: {} };

    async toNotebook(bytes: BinaryBuffer): Promise<NotebookData> {
        const nb = JSON.parse(bytes.toString() || '{"cells":[]}');
        const cells = (nb.cells || []).map((c: any) => {
            const src = Array.isArray(c.source) ? c.source.join('') : (c.source || '');
            const code = c.cell_type === 'code';
            return {
                cellKind: code ? CellKind.Code : CellKind.Markup,
                language: code ? 'python' : 'markdown',
                source: src,
                outputs: [],
                metadata: {},
            };
        });
        return { cells, metadata: nb.metadata || {} };
    }

    async fromNotebook(data: NotebookData): Promise<BinaryBuffer> {
        const cells = data.cells.map(c => ({
            cell_type: c.cellKind === CellKind.Code ? 'code' : 'markdown',
            source: c.source.split(/(?<=\n)/),
            metadata: {},
            ...(c.cellKind === CellKind.Code ? { outputs: [], execution_count: null } : {}),
        }));
        const nb = { cells, metadata: data.metadata || {}, nbformat: 4, nbformat_minor: 5 };
        return BinaryBuffer.fromString(JSON.stringify(nb, null, 1));
    }
}
