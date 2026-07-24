import * as monaco from '@theia/monaco-editor-core';
import { PyodideClient, CompletionItem } from './pyodide-runtime';

/* Registers a Python completion provider backed by the worker's jedi. Because
 * Monaco powers BOTH the file editor and every notebook cell, this single
 * registration gives autocomplete on all Python surfaces. jedi is warmed in the
 * background at boot (+ IDBFS parso cache), so completions are ~50 ms once ready;
 * the very first call may wait on the install. */

function kindFor(type: string): monaco.languages.CompletionItemKind {
    const K = monaco.languages.CompletionItemKind;
    switch (type) {
        case 'function': return K.Function;
        case 'class': return K.Class;
        case 'instance': return K.Variable;
        case 'module': return K.Module;
        case 'keyword': return K.Keyword;
        case 'param': return K.Field;
        case 'property': return K.Property;
        case 'statement': return K.Variable;
        default: return K.Text;
    }
}

export function registerZoomyCompletions(client: PyodideClient, log: (m: string) => void): monaco.IDisposable {
    // This minimal Theia app has no Python grammar extension, so `.py` files and
    // notebook cells can open as plaintext. Register the language id so they get
    // 'python', and attach the provider to both ids to be safe.
    try {
        if (!monaco.languages.getLanguages().some(l => l.id === 'python')) {
            monaco.languages.register({ id: 'python', extensions: ['.py'], aliases: ['Python', 'python'] });
        }
    } catch (e) { log('register python lang: ' + ((e as any)?.message || e)); }

    const provider: monaco.languages.CompletionItemProvider = {
        triggerCharacters: ['.', '(', ','],
        async provideCompletionItems(model, position, _context, token): Promise<monaco.languages.CompletionList> {
            try {
                const res = await client.complete(model.getValue(), position.lineNumber, position.column - 1);
                if (token.isCancellationRequested) { return { suggestions: [] }; }
                const word = model.getWordUntilPosition(position);
                const range = new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn);
                const suggestions = (res.completions || []).map((c: CompletionItem) => ({
                    label: c.name,
                    kind: kindFor(c.type),
                    insertText: c.name,
                    detail: c.signature || c.module || c.type,
                    documentation: c.docstring ? { value: c.docstring } : undefined,
                    range,
                }));
                return { suggestions };
            } catch (e) {
                log('completion error: ' + ((e as any)?.message || e));
                return { suggestions: [] };
            }
        },
    };
    const disposables = ['python', 'plaintext'].map(lang => monaco.languages.registerCompletionItemProvider(lang, provider));
    return { dispose(): void { disposables.forEach(d => d.dispose()); } };
}
