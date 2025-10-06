from sympy.printing.c import C99CodePrinter

from sympy.codegen.ast import Assignment, Return
from sympy.utilities.codegen import Routine
from sympy import symbols, Add
import os

class PlainCCodePrinter(C99CodePrinter):
    """Printer for plain C functions with pointer arguments."""

    def _print_Routine(self, routine: Routine):
        # Convert arguments to pointer style: double *arg
        args_str = ", ".join(f"double *{a.name}" for a in routine.arguments)
        # Convert body
        body_str = "\n".join(self._print(stmt) for stmt in routine.body)
        return f"void {routine.name}({args_str}) {{\n{body_str}\n}}"

    def _print_Assignment(self, expr: Assignment):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return f"{lhs} = {rhs};"

    def _print_Return(self, expr: Return):
        # For plain C, assign return to first output pointer
        return f"{expr.expr} /* return value ignored in void function */;"
    


def make_plain_routine(name, expr, args):
    """Create a Routine for a single-output function using pointers."""
    output_arg = args[0]
    assign = Assignment(output_arg, expr)
    local_vars = []  # no extra local variables
    global_vars = [] # no globals
    return Routine(name, [assign], args, local_vars, global_vars)


def write_plain_c_module(module_name, expression_name_tuples, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    printer = PlainCCodePrinter()
    c_code = ""

    for name, expr, args in expression_name_tuples:
        routine = make_plain_routine(name, expr, args)
        c_code += printer._print_Routine(routine) + "\n\n"

    # Write C file
    c_file = os.path.join(directory, f"{module_name}.c")
    with open(c_file, "w") as f:
        f.write(c_code)
    
    # Make __init__.py
    open(os.path.join(directory, "__init__.py"), "w").close()

