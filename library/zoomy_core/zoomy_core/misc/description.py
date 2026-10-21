"""Rich description object for notebook-native rendering and document export.

`Description` holds markdown content and renders automatically in
Jupyter via `_repr_markdown_`.  Export to `.md` or `.tex` files
for direct use in documents.
"""

from __future__ import annotations

import re


class Description:
    """Renderable description that Jupyter displays as markdown."""

    def __init__(self, markdown: str):
        self._md = markdown

    # ── Jupyter rendering ─────────────────────────────────────────────

    def _repr_markdown_(self):
        return self._md

    def __repr__(self):
        text = self._md
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\$\$\n?(.+?)\n?\$\$', r'[equation]', text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$', r'\1', text)
        return text

    def __str__(self):
        return self._md

    def __add__(self, other):
        if isinstance(other, Description):
            return Description(self._md + "\n" + other._md)
        return Description(self._md + "\n" + str(other))

    # ── Export formats ────────────────────────────────────────────────

    def to_markdown(self) -> str:
        """Return raw markdown string."""
        return self._md

    def to_latex(self) -> str:
        """Convert to LaTeX document fragment.

        Translates markdown headings, bold, and `$$` blocks into
        LaTeX equivalents.  The result can be pasted into a `.tex`
        file or included via `\\input{}`.
        """
        text = self._md

        # Headings: # → \section, ## → \subsection, ### → \subsubsection
        text = re.sub(r'^### (.+)$', r'\\subsubsection*{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'\\subsection*{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'\\section*{\1}', text, flags=re.MULTILINE)

        # Bold: **text** → \textbf{text}
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)

        # Display math: $$ ... $$ → \begin{equation*} ... \end{equation*}
        text = re.sub(
            r'\$\$\n?(.+?)\n?\$\$',
            r'\\begin{equation*}\n\1\n\\end{equation*}',
            text,
            flags=re.DOTALL,
        )

        # Bullet lists: - item → \item item (wrap in itemize)
        lines = text.split('\n')
        out = []
        in_list = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('- '):
                if not in_list:
                    out.append('\\begin{itemize}')
                    in_list = True
                out.append('  \\item ' + stripped[2:])
            else:
                if in_list:
                    out.append('\\end{itemize}')
                    in_list = False
                out.append(line)
        if in_list:
            out.append('\\end{itemize}')

        return '\n'.join(out)

    # ── File I/O ──────────────────────────────────────────────────────

    def save(self, path: str):
        """Save to file.  Format chosen by extension (.md or .tex)."""
        if path.endswith('.tex'):
            content = self.to_latex()
        else:
            content = self.to_markdown()
        with open(path, 'w') as f:
            f.write(content)
