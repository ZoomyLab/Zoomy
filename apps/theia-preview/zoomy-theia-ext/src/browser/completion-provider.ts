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

/** Register a compact Python Monarch tokenizer so `.py` files and notebook code
 *  cells get syntax coloring — Theia's monaco-editor-core ships the language id
 *  but no Python grammar, so without this everything renders as plain text. */
let _pyGrammarDone = false;
function ensurePythonGrammar(log: (m: string) => void): void {
    if (_pyGrammarDone) { return; }
    try {
        monaco.languages.setLanguageConfiguration('python', {
            comments: { lineComment: '#' },
            brackets: [['{', '}'], ['[', ']'], ['(', ')']],
            autoClosingPairs: [{ open: '{', close: '}' }, { open: '[', close: ']' }, { open: '(', close: ')' }, { open: '"', close: '"' }, { open: "'", close: "'" }],
        });
        monaco.languages.setMonarchTokensProvider('python', {
            defaultToken: '',
            keywords: ['and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'],
            constants: ['True', 'False', 'None', 'self', 'cls'],
            tokenizer: {
                root: [
                    [/[a-zA-Z_]\w*/, { cases: { '@keywords': 'keyword', '@constants': 'constant.language', '@default': 'identifier' } }],
                    [/#.*$/, 'comment'],
                    [/@[a-zA-Z_]\w*/, 'annotation'],
                    [/"""/, { token: 'string', next: '@tstringd' }],
                    [/'''/, { token: 'string', next: '@tstrings' }],
                    [/"/, { token: 'string', next: '@stringd' }],
                    [/'/, { token: 'string', next: '@strings' }],
                    [/\d+\.?\d*([eE][+-]?\d+)?[jJ]?/, 'number'],
                    [/[+\-*/%=<>!&|^~]+/, 'operator'],
                ],
                tstringd: [[/[^"]+/, 'string'], [/"""/, { token: 'string', next: '@pop' }], [/"/, 'string']],
                tstrings: [[/[^']+/, 'string'], [/'''/, { token: 'string', next: '@pop' }], [/'/, 'string']],
                stringd: [[/[^"\\]+/, 'string'], [/\\./, 'string.escape'], [/"/, { token: 'string', next: '@pop' }]],
                strings: [[/[^'\\]+/, 'string'], [/\\./, 'string.escape'], [/'/, { token: 'string', next: '@pop' }]],
            },
        } as any);
        _pyGrammarDone = true;
    } catch (e) { log('python grammar: ' + ((e as any)?.message || e)); }
}

export function registerZoomyCompletions(client: PyodideClient, log: (m: string) => void): monaco.IDisposable {
    // This minimal Theia app has no Python grammar extension, so `.py` files and
    // notebook cells can open as plaintext. Register the language id so they get
    // 'python', and attach the provider to both ids to be safe.
    try {
        if (!monaco.languages.getLanguages().some(l => l.id === 'python')) {
            monaco.languages.register({ id: 'python', extensions: ['.py'], aliases: ['Python', 'python'] });
        }
        ensurePythonGrammar(log);
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
