import os
import param
import panel as pn

from apps.guiv2.basicelements import Card, Section

main_dir = os.getenv("ZOOMY_DIR")

pn.extension()


class VisuCard(Card):

    def __init__(self, parent_app, img='', **params):
        super().__init__(parent_app, **params)

        if img.endswith("png"):
            self.fig = pn.pane.PNG(img, width=300)
        else:
            image=os.path.join(main_dir, "apps/gui/data/sample_mesh.png")
            self.fig = pn.pane.PNG(image, width=300)

        self._layout = pn.Column(self.title, self.fig, self._btn, styles=self._default_style)


    def get_controls(self):
        """Optional: Return extra widgets unique to this card."""
        return [pn.widgets.StaticText(value="No extra controls for this card.")]



class VisuSection(Section):
    def __init__(self, parent_app, **params):
        super().__init__(**params)

        self.title='Visualization'

        visu1 = VisuCard(parent_app, goto_page='paraview', img='', title="Paraview")
        visu2 = VisuCard(parent_app, img='', title="Matplotlib")

        self.manager.add_card(visu1)
        self.manager.add_card(visu2)

        self.manager.selected_card = visu1

