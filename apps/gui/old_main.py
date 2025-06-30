import panel as pn

pn.extension(template="fast")

logout = pn.widgets.Button(name="Log out")
logout.js_on_click(code="""window.location.href = './logout'""")
pn.Column(f"Congrats `{pn.state.user}`. You got access!", logout).servable()
