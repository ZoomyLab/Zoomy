from bokeh.models import FreehandDrawTool, ColumnDataSource, BoxEditTool


# Create a data source for the freehand drawings

global freehand_source
global rect_source 


freehand_source = ColumnDataSource(data=dict(xs=[], ys=[]))

def add_freehand(p):
    # Add a multi_line glyph to the figure for freehand drawing
    renderer = p.multi_line('xs', 'ys', source=freehand_source, line_width=2, line_color='red')
    
    # Create the FreehandDrawTool and associate it with the renderer
    freehand_draw_tool = FreehandDrawTool(renderers=[renderer])
    
    # Add the FreehandDrawTool to the figure's tools
    p.add_tools(freehand_draw_tool)
    
    # Set the FreehandDrawTool as the active drag tool
    p.toolbar.active_drag = freehand_draw_tool
    return renderer

rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], angle=[]))

def add_rect(p):
    # Add a rectangle glyph to the figure
    rect_renderer = p.rect('x', 'y', 'width', 'height', source=rect_source,
                           fill_color='red', fill_alpha=0.4, line_color='red')
    
    # Create the BoxEditTool and associate it with the rectangle renderer
    box_edit_tool = BoxEditTool(renderers=[rect_renderer])
    
    # Add the BoxEditTool to the figure's tools
    p.add_tools(box_edit_tool)
    
    # Set the BoxEditTool as the active drag tool
    p.toolbar.active_drag = box_edit_tool

    return rect_renderer
    

