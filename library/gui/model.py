import param
import panel as pn

from apps.guiv2.basicelements import Card, Section
from apps.gui.docstring_crawler import get_class_docstring
from library.zoomy_core.model.models.shallow_moments import ShallowMoments

pn.extension('katex')

class ModelCard(Card):

    def __init__(self, parent_app, latex_description, model_class, **params):
        super().__init__(parent_app, **params)
        self._code=model_class
        self._doc= get_class_docstring(self._code)

        latex = pn.pane.LaTeX(latex_description, width=300)

        self._layout = pn.Column(self.title, latex, self._btn, styles=self._default_style)


    def get_controls(self):
        """Optional: Return extra widgets unique to this card."""
        return [pn.widgets.StaticText(value="No extra controls for this card.")]

class ModelSection(Section):

    def __init__(self, parent_app, **params):
        super().__init__(**params)

        self.title='Model'

        description =  r'$\partial_t {Q} + \nabla \cdot {F}({Q}) + {NC}({Q}) : \nabla {Q} = {S}({Q})$'
        model1 = ModelCard(parent_app, latex_description=description, model_class=ShallowMoments, title="Model #1")
        model2 = ModelCard(parent_app, latex_description=description, model_class=ShallowMoments, title="Model #2")

        self.manager.add_card(model1)
        self.manager.add_card(model2)

        self.manager.selected_card = model1

