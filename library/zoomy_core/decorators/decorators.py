
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
                print("Requirement {}: {}".format(requirement, requirement_evaluated))
                assert requirement_evaluated
            return func(settings, *args, **kwargs)

        return wrapper

    return req_decorator
