import param
import panel as pn

pn.extension()

class Card(param.Parameterized):
    title    = param.String(default="Card")
    selected = param.Boolean(default=False, doc="Is this card selected?")
    manager  = param.Parameter(default=None)
    parent_app = param.Parameter(default=None, doc="Reference to MainApp")
    goto_page = 'tabs'

    _default_style  = {'border': '1px solid white',  'border-radius': '10px'}
    _selected_style = {'border': '1px solid black', 'border-radius': '10px'}

    def __init__(self, parent_app, goto_page=None, **params):
        super().__init__(**params)
        self.parent_app = parent_app
        # A button to let the user select this card
        self._btn = pn.widgets.Button(name="Select", button_type='primary', width=300)
        self._btn.on_click(self._on_select)
        # The card's layout
        self._layout = pn.Column(self.title, self._btn, styles=self._default_style)
        self.goto_page=goto_page
        # Watch for changes in 'selected' to update style
        self.param.watch(self._update_style, 'selected')

    def _on_select(self, _event):
        """When the user clicks 'Select', notify our manager we're selected."""
        if self.manager is not None:
            self.manager.selected_card = self
        else:
            self.selected = True
        if self.parent_app:
            if self.goto_page is not None:
                self.parent_app.active_page = self.goto_page

    def _update_style(self, event):
        """Switch border style based on 'selected'."""
        if self.selected:
            self._layout.styles = self._selected_style
        else:
            self._layout.styles = self._default_style

    def panel_view(self):
        return self._layout

    def get_controls(self):
        """Optional: Return extra widgets unique to this card."""
        return [pn.widgets.StaticText(value="No extra controls for this card.")]

class CardManager(param.Parameterized):
    cards         = param.List(default=[], doc="All Card objects this manager handles.")
    selected_card = param.Parameter(default=None, doc="Which card is currently selected?")

    @param.depends('selected_card', watch=True)
    def _update_selections(self):
        """Ensure only the selected_card is marked selected."""
        for c in self.cards:
            c.selected = (c is self.selected_card)

    def add_card(self, card):
        """Register a new card with the manager."""
        card.manager = self
        self.cards.append(card)

    def view(self):
        """Row (FlexBox) of all cards."""
        return pn.FlexBox(*(c.panel_view() for c in self.cards))

    def controls(self):
        """Controls from the currently selected card."""
        if self.selected_card is None:
            return pn.Column("No card selected.")
        return pn.Column(*self.selected_card.get_controls())


class Section(param.Parameterized):
    # Because we want to watch changes in the manager, we keep it as a Param field:
    manager = param.ClassSelector(class_=CardManager)

    def __init__(self, **params):
        super().__init__(**params)

        # Create a manager and two “Model” cards
        self.manager = CardManager()
        self.title = 'Section'
        
        self.manager.selected_card = None

    @param.depends('manager.selected_card')
    def main_view(self):
        """
        This will update whenever the selected card changes.
        Renders a column with the row of cards + the selected card's controls.
        """
        selected = self.manager.selected_card
        selected_title = selected.title if selected else "(none)"

        return pn.Column(
            f"## {self.title}",
            pn.Spacer(height=10),
            self.manager.view(),
        )

    def sidebar(self):
        """If you want to show any additional mesh-specific widgets in the sidebar."""
        return pn.Column()




