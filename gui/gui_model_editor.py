import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')

import test as test_code
from gui_elements import MyBasicOrganizer

code_file = open('./test.py', 'r')
code_text = code_file.read()
editor = pn.widgets.CodeEditor(value=code_text, sizing_mode='stretch_width', language='python', theme='monokai', height=300)

def update_terminal_from_editor(button, terminal, editor):
    terminal.value = editor.value


# latex = pn.pane.LaTeX(r"""
# $\begin{aligned}
#   \nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
#   \nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
#   \nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
#   \nabla \cdot \vec{\mathbf{B}} & = 0
# \end{aligned}
# $""", styles={'font-size': '18'})
def wrap_latex(text):
    begin = r'$\begin{aligned}'
    end = r'\end{aligned}$'
    output = f'{begin} {text} {end}'
    return output
latex = pn.pane.LaTeX(wrap_latex(test_code.get_model()))


main = GridStack(sizing_mode='stretch_both', min_height=60, min_width=190)
main[0:9, 0:6] = editor
main[0:9, 6:12] = latex
main[9:12, 0:6] = pn.Spacer(styles=dict(background='purple'))
main[9:12, 6:12] = pn.Spacer(styles=dict(background='yellow'))

model_editor = MyBasicOrganizer(main)


checkbox_1 = pn.widgets.Checkbox(name='Show Model')
checkbox_2 = pn.widgets.Checkbox(name='Show Eigenvalues')
controls = pn.Column('# Editor controls', checkbox_1, checkbox_2)
model_editor.attach_controls(controls)
