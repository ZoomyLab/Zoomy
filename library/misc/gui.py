 # go though all files in curent directory
# check the docstring of all functions and classes in thhese files
# parse the docstring using the docstring_parser module and print out all gui tags

import os
import inspect
from docstring_parser.rest import parse
import yaml

import importlib.util

def import_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Go through all files in library
# go though all files and files of subdirectories until you find a .py file
def browse_all_gui_tags(path): 
    for root, dirs, files in os.walk(path):
        for file in files:
            get_gui_tags(os.path.join(root, file))
        for dir in dirs:
            browse_all_gui_tags(os.path.join(root, dir))

def get_gui_tags(path):
    # Import the module
    if not path.endswith('.py'):
        return 
    try:
        module = import_module_from_path(path)

        # Check the docstring of all functions and classes in these files
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                docstring = inspect.getdoc(obj)
                if docstring:
                    # Parse the docstring using the docstring_parser module
                    parsed_docstring = parse(docstring)

                    # Print out all 'gui' tags
                    for meta in parsed_docstring.meta:
                        if meta.args == ['gui']:
                            yml = yaml.safe_load( parsed_docstring.meta[1].description)
                            print(yml)
    except:
        print(f"Could not import module from {path}")
        pass
    
if __name__ == "__main__":
    main_dir = os.getenv("SMS")
    browse_all_gui_tags(os.path.join(main_dir, 'library'))