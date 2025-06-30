import os
import param
import panel as pn

from apps.guiv2.basicelements import Card, Section
from apps.guiv2.libmesh.load_gmsh import load_gmsh

main_dir = os.getenv("SMS")

pn.extension()


class MeshCard(Card):

    def __init__(self, parent_app, path, **params):
        super().__init__(parent_app, **params)

        if path.endswith("msh"):
            plots, cells = load_gmsh(path)
            self.fig = plots[0]
            self.path = path
        elif path.endswith("png"):
            self.fig = pn.pane.PNG(path, width=300)
        else:
            image=os.path.join(main_dir, "apps/gui/data/sample_mesh.png")
            self.fig = pn.pane.PNG(image, width=300)

        self._layout = pn.Column(self.title, self.fig, self._btn, styles=self._default_style)


    def get_controls(self):
        """Optional: Return extra widgets unique to this card."""
        return [pn.widgets.StaticText(value="No extra controls for this card.")]



class MeshSection(Section):
    def __init__(self, parent_app, **params):
        super().__init__(**params)

        self.title='Mesh'

        mesh1 = MeshCard(parent_app, path='meshes/quad_2d/mesh.msh', title="Mesh #1")
        mesh2 = MeshCard(parent_app, path='meshes/simple_openfoam/mesh_2d_mid.msh', title="Mesh #2" )

        self.manager.add_card(mesh1)
        self.manager.add_card(mesh2)

        self.manager.selected_card = mesh1

    def sidebar(self):
        """Mesh sidebar content"""
        load_gmsh = pn.widgets.Button(name="load GMSH", button_type="primary")
        return pn.Column(
            f"## {self.title} Controls",
            load_gmsh
        )

