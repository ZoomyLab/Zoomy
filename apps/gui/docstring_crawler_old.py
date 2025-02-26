import re
import ast

def print_class_docstrings(cls):
    """
    Prints the docstring of the given class and its parent classes.
    """
    if cls is None:
        return
    
    # # Print the docstring of the current class
    # print(extract_gui_dict(cls))
    
    # Get the parent class
    parent_class = cls.__base__
    
    # If the parent class is not 'object', recursively print its docstring
    if parent_class is not object:
        print_class_docstrings(parent_class)


def extract_gui_dict(cls):
    """
    Extracts the dictionary under the :gui: delimiter from the class docstring.
    """
    docstring = cls.__doc__
    if not docstring:
        return None
    
    # Regular expression to find the :gui: section
    gui_section = re.search(r':gui:\s*{', docstring, re.DOTALL)
    if not gui_section:
        return None

    # Find the start position of the dictionary
    start_pos = gui_section.end() - 1
    
    # Extract the dictionary string with matching braces
    brace_count = 0
    for i, char in enumerate(docstring[start_pos:], start=start_pos):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_pos = i + 1
                break
    else:
        print("Error: No matching closing brace found.")
        return None
    
    # Extract the dictionary string
    gui_dict_str = docstring[start_pos:end_pos]
    
    # Convert the string to a dictionary
    try:
        gui_dict = ast.literal_eval(gui_dict_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing dictionary: {e}")
        return None
    
    return gui_dict

# Example usage
class SWE:
    """
    This is the SWE class.
    
    :gui:
    {
        'parameters': { 'I': {'type': 'int', 'value': 1, 'step': 1}, 'F': {'type': 'float', 'value': 1., 'step':0.1}, 'S': {'type': 'string', 'value': 'asdf'}, 'Arr':{'type': 'array', 'value': np.array([[1., 2.], [3., 4.]])} },
    }
    """

gui_dict = extract_gui_dict(SWE)
print(gui_dict)