class Base:
    """
    This is the parent class.

    :gui:
    { 'parameters': { 'I': {'type': 'int', 'value': 1, 'step': 1}, 'F': {'type': 'float', 'value': 1., 'step':0.1}, 'S': {'type': 'string', 'value': 'asdf'}, 'Arr':{'type': 'array', 'value': np.array([[1., 2.], [3., 4.]])} },}

    """

class SWE(Base):
    """
    This is the SWE class.

    :gui:
    {
        'parameters': { 'I': {'type': 'int', 'value': 2, 'step': 2}, 'F': {'type': 'float', 'value': 2., 'step':0.2}, 'S': {'type': 'string', 'value': 'asdfasdf'}, 'Arr':{'type': 'array', 'value': np.array([[4., 2.], [3., 4.]])} },
    }

    """

