import param
import panel as pn

from apps.guiv2.mesh import MeshSection
from apps.guiv2.model import ModelSection
from apps.guiv2.visu import VisuSection

pn.extension()

class PageTabs(param.Parameterized):
    active_tab = param.Integer(default=0, bounds=(0,3))
    parent_app = param.Parameter(default=None, doc="Reference to the main app")

    def __init__(self, **params):
        super().__init__(**params)
        # Example tabs: Mesh, Model, and Visu

        tabMesh = MeshSection(self.parent_app)
        tabModel = ModelSection(self.parent_app)
        tabVisu = VisuSection(self.parent_app)
        self.tab_list = [tabMesh, tabModel, tabVisu]

        self.tabs = pn.Tabs(*[(tab.title, tab.main_view()) for tab in self.tab_list], dynamic=True)
        self.tabs.param.watch(self._update_active_tab, 'active')

    def _update_active_tab(self, event):
        self.active_tab = event.new

    def sidebar_content(self):
        """A small placeholder for a tab-based sidebar."""
        return pn.Column(
            self.tab_list[self.active_tab].sidebar()
        )

    def view(self):
        """
        Return a plain layout (e.g., Column).
        The parent app can embed this layout into a template.
        """
        return pn.Column(
            "## Tabs Page",
            self.tabs
        )


class PageParaview(param.Parameterized):
    """
    A separate page for specialized Paraview-like content.
    """
    parent_app = param.Parameter(default=None, doc="Reference to MainApp")

    def view(self):

        html_pane = pn.pane.HTML("""
        <iframe src="http://localhost:8080" 
                width="100%" 
                height="100%" 
                style="border:none; position:absolute; top:0; left:0; width:100%; height:100%;">
        </iframe>
        """, sizing_mode="stretch_both", height_policy="max", width_policy="max")


        return pn.Column(
        html_pane
        )


class MainApp(param.Parameterized):
    """
    The top-level “router” that displays either:
      - The tab-based PageTabs
      - The Paraview page
    using a single BootstrapTemplate.
    """
    active_page = param.Selector(default='tabs', objects=['tabs', 'paraview'])

    page_tabs     = param.ClassSelector(class_=PageTabs)
    page_paraview = param.ClassSelector(class_=PageParaview)

    def __init__(self, **params):
        super().__init__(**params)
        # Instantiate sub-pages
        self.page_tabs = PageTabs(parent_app=self)
        self.page_paraview = PageParaview(parent_app=self)

    @param.depends('active_page')
    def main_content(self):
        """
        A Param-depends method returning whichever page is active.
        """
        if self.active_page == 'tabs':
            return self.page_tabs.view()
        return self.page_paraview.view()

    @param.depends('active_page', 'page_tabs.active_tab')
    def sidebar_content(self):
        """
        The top-level template's sidebar: a “global” portion plus
        either the tab's sidebar or a back button (when in paraview mode).
        """
        global_part = pn.Column(
            pn.widgets.Button(name="Save session", button_type="success"),
            pn.widgets.Button(name="Load session", button_type="success"),
        )

        if self.active_page == 'tabs':
            # Show PageTabs sidebar
            tabs_part = self.page_tabs.sidebar_content()
            return pn.Column(global_part, pn.layout.Divider(), tabs_part)

        else:
            back_button = pn.widgets.Button(name="Back to Tabs", button_type="danger")
            def go_back(_):
                self.active_page = 'tabs'
            back_button.on_click(go_back)

            return pn.Column(
                back_button,
                pn.layout.Divider(),
                global_part,
            )


    def view(self):
        """
        Only one top-level BootstrapTemplate. We embed:
        - main_content (the current page) in template.main
        - sidebar_content in template.sidebar
        """
        template = pn.template.BootstrapTemplate(title="Shallow Moment Simulation Suite")
        template.sidebar.append(self.sidebar_content)    # pass the method
        template.main.append(self.main_content)          # pass the method
        return template

def main():
    """panel serve main.py --show entry point."""
    return MainApp().view()

layout = main()
layout.servable()

