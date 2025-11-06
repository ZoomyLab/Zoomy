import decorator_test_def as dtd
from inspect import getmembers
from inspect import isbuiltin
from inspect import isfunction

# from inspect import ismethod as isA
from inspect import isroutine as isA

# registry = []


@dtd.register
def my_func():
    def f():
        print("hi")

    return f


my_func()

for r in dtd.registry:
    r.print()
    # spec = importlib.util.spec_from_file_location('decorator_test', r.path)
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # # from mod import my_func as func
    # mod.my_func(*r.args, **r.kwargs)

for member in getmembers(dtd, isA):
    print(member[0], member[1])
    # print(member[1].__doc__)
