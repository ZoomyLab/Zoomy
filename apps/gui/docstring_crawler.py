import re
import ast
import numpy as np
import inspect


def print_class_docstring(cls):
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
        print_class_docstring(parent_class)

def get_class_docstring(cls, init_dict={}):
    """
    Extracts the docstring of the given class and its parent classes.
    """
    if cls is None:
        return
    
    # Print the docstring of the current class
    out = extract_gui_dict(cls)
    out.update(init_dict)
    
    # Get the parent class
    parent_class = cls.__base__
    
    # If the parent class is not 'object', recursively print its docstring
    if parent_class is not object:
        return get_class_docstring(parent_class, out)
    print(out)
    return out

def get_class_code(cls, func_name=None):
    """
    Extracts the code of the given class or a specific function within the class.
    
    Parameters:
    cls (type): The class to extract the code from.
    func_name (str, optional): The name of the function to extract the code for. Defaults to None.
    
    Returns:
    str: The source code of the class or the specified function.
    """
    if cls is None:
        return ""
    
    # If func_name is provided, extract the code for the specified function
    if func_name:
        try:
            func = getattr(cls, func_name)
            source_code = inspect.getsource(func)
        except (AttributeError, TypeError):
            source_code = f"Function '{func_name}' not found in class '{cls.__name__}'"
    else:
        # Get the source code of the entire class
        try:
            source_code = inspect.getsource(cls)
        except TypeError:
            source_code = f"Class '{cls.__name__}' source code not found"
    
    return source_code


def get_function_names(file_path):
    """
    Extracts all function names from a given Python file.
    
    Parameters:
    file_path (str): The path to the Python file.
    
    Returns:
    list: A list of function names defined in the file.
    """
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    function_names = get_function_names(file_path)
    return function_names


def get_class_function_names(cls):
    """
    Extracts all function names from a given class.
    
    Parameters:
    cls (type): The class to extract function names from.
    
    Returns:
    list: A list of function names defined in the class.
    """
    return [name for name, member in inspect.getmembers(cls, predicate=inspect.isfunction)if not name.startswith('_')]

    
    


def extract_gui_dict(cls):
    """
    Extracts the dictionary under the :gui: delimiter from the class docstring.
    """
    docstring = cls.__doc__
    if not docstring:
        return {}
    
    # Remove newlines from the docstring
    docstring = docstring.replace('\n', '')
    
    # Regular expression to find the :gui: section
    gui_section = re.search(r':gui:\s*{', docstring)
    if not gui_section:
        return {}
    
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
        assert False
        return {}
    
    gui_dict_str = docstring[start_pos:end_pos]
    
    # Convert the string to a dictionary
    try:
        gui_dict = eval(gui_dict_str, {"np": np})
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing dictionary: {e}")
        assert False
        return {}
    
    return gui_dict

# Example usage
# class SWE:
#     """
#     This is the SWE class.
    
#     :gui:
#     {
#         'parameters': { 'I': {'type': 'int', 'value': 1, 'step': 1}, 'F': {'type': 'float', 'value': 1., 'step':0.1}, 'S': {'type': 'string', 'value': 'asdf'}, 'Arr':{'type': 'array', 'value': np.array([[1., 2.], [3., 4.]])} },
#     }
#     """

# gui_dict = extract_gui_dict(SWE)
# print(gui_dict)