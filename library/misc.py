import os
import numpy as np
import scipy.interpolate as interp
from functools import wraps
# from dotmap import DotMap

main_dir = os.getenv('SMPYTHON')

def require(requirement):
    """
    Decorator to check if a requirement is met before executing the decorated function.

    Parameters:
    - requirement (str): The requirement string to evaluate. Should evaluate to True or False.

    Returns:
    - wrapper: The decorated function that will check the requirement before executing.
    """
    # decorator to check the assertion given in requirements given the settings 
    def req_decorator(func):
        @wraps(func)
        def wrapper(settings, *args, **kwargs):
            requirement_evaluated = eval(requirement) 
            if not requirement_evaluated:
                print('Requirement {}: {}'.format(requirement, requirement_evaluated))
                assert requirement_evaluated
            return func(settings,*args, **kwargs)
        return wrapper
    return req_decorator 



def all_class_members_identical(a, b):
    members = [attr for attr in dir(a) if not callable(getattr(a, attr)) and not attr.startswith("__")]
    for member in members:
        m_a = getattr(a, member)
        m_b = getattr(b, member)
        if type(m_a) == np.ndarray:
            if not ( (getattr(a, member) == getattr(b, member)).all()):
                print(getattr(a, member))
                print(getattr(b, member))
                assert False
        else:
            if not ( (getattr(a, member) == getattr(b, member))):
                print(getattr(a, member))
                print(getattr(b, member))
                assert False

# def load_npy(filepath=main_dir + "/output/", filename="mesh.npy", filenumber=None):
#     if filenumber is not None:
#         full_filename = filepath + filename + "." + str(int(filenumber))
#     else:
#         full_filename = filepath + filename
#     if not os.path.exists(full_filename):
#         print("File or file path to: ", full_filename, " does not exist")
#         assert False
#     data = np.load(full_filename)
#     return data



# def write_field_to_npy(
#     field, filepath=main_dir + "/output/", filename="mesh.npy", filenumber=None
# ):
#     if filenumber is not None:
#         full_filename = filepath + filename + "." + str(int(filenumber))
#     else:
#         full_filename = filepath + filename
#     os.makedirs(filepath, exist_ok=True)
#     # the extra step over 'open' is to allow for a filename with filenumber
#     with open(full_filename, "wb") as f:
#         np.save(f, field)


# def interpolate_field_to_mesh(field, mesh_field, mesh_out):
#     interpolator = interp.interp1d(mesh_field, field)
#     return interpolator(mesh_out)




