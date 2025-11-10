import panel as pn
pn.extension()

# # Read the HTML viewer into a string
# with open("viewer/index.html") as f:
#     html_content = f.read()
# 
# iframe = pn.pane.HTML(f"""
#     <iframe srcdoc="{html_content}" 
#             width="100%" height="800px" style="border:none;"></iframe>
# """, height=800)

#html_pane = pn.pane.HTML("""
#<iframe src="http://localhost:5050" width="100%" height="800px" style="border:none;"></iframe>
#""", height=800)

html_pane = pn.pane.HTML("""
<iframe src="http://localhost:5050" 
        width="100%" 
        height="100%" 
        style="border:none; position:absolute; top:0; left:0; width:100%; height:100%;">
</iframe>
""", sizing_mode="stretch_both", height_policy="max", width_policy="max")

pn.Column(
    "# embedded vtk.js viewer (srcdoc)",
    html_pane
).servable()

