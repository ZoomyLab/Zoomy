import os
import inspect
import importlib.util
import sys
from attrs import define
from typing import Callable, Optional

registry = []


@define()
class RegistryEntry:
    name: str
    path: str
    args: tuple
    kwargs: dict
    f: Callable

    def print(self):
        print("-----------")
        print(self.name)
        print(self.path)
        print(self.args)
        print(self.kwargs)
        print("-----------")


def register(func):
    def wrapper(*args, **kwargs):
        path = os.path.abspath(inspect.getfile(func))
        name = func.__name__
        f_args = args
        f_kwargs = kwargs
        r = registry.append(RegistryEntry(name, path, f_args, kwargs, func))
        # func(*args, **kwargs)
        # foo = importlib.util.module_from_spec(spec)
        # foo(*f_args, **kwargs)
        # print("Something is happening before the function is called.")
        func(*args, **kwargs)
        # print("Something is happening after the function is called.")

    return wrapper

def func1():
    """
    TAG: 1
    """
    pass

class A():
    a = 1    
    def print(self):
        print(str(self.a))




