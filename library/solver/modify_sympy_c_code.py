import os


def change_file_ending_from_c_to_cpp(path, prefix="_code"):
    """
    Change file endings from .c to .cpp
    """
    for filename in os.listdir(path):
        if filename.endswith(prefix + ".c"):
            os.rename(
                os.path.join(path, filename),
                os.path.join(path, filename.replace(".c", ".cpp")),
            )


def introduce_sympy_namespace(filepath):
    """
    Introduce the sympy namespace to the c code.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "void" in line:
            lines.insert(i, "namespace sympy {\n")
            break

    for i, line in enumerate(lines):
        if "#endif" in line:
            lines.insert(i, "}\n")
            break
    else:
        lines.append("}")

    with open(filepath, "w") as f:
        f.writelines(lines)


def transform_code(settings, model_path="c_interface/Model"):
    main_dir = os.getenv("SMS")
    change_file_ending_from_c_to_cpp(
        os.path.join(main_dir, os.path.join(settings.output_dir, model_path))
    )
    introduce_sympy_namespace(
        os.path.join(
            main_dir,
            os.path.join(settings.output_dir, model_path, "model_code.cpp"),
        )
    )
    introduce_sympy_namespace(
        os.path.join(
            main_dir,
            os.path.join(settings.output_dir, model_path, "model_code.h"),
        )
    )
    introduce_sympy_namespace(
        os.path.join(
            main_dir,
            os.path.join(
                settings.output_dir, model_path, "boundary_conditions_code.cpp"
            ),
        )
    )
    introduce_sympy_namespace(
        os.path.join(
            main_dir,
            os.path.join(settings.output_dir, model_path, "boundary_conditions_code.h"),
        )
    )
