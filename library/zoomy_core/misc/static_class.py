
import attrs
from functools import partial
from typing import Any, Tuple, Type
import attr

try:
    import jax
    import jax.numpy as jnp
    _HAVE_JAX = True
except ImportError:
    _HAVE_JAX = False


def register_static_pytree(cls: Type[Any]) -> Type[Any]:
    """
    Class decorator that registers the class as a JAX pytree node,
    treating all member variables as static.

    Parameters:
    cls (Type[Any]): The class to register.

    Returns:
    Type[Any]: The registered class.
    """
    if not _HAVE_JAX:
        # no-op decorator
        return cls
    if not attrs.has(cls):
        raise TypeError(
            "register_static_pytree can only be applied to classes decorated with @attrs.define or @attr.s"
        )

    # Extract field names from attrs
    field_names = [field.name for field in attrs.fields(cls)]

    # Define the flatten function
    def flatten(instance: Any) -> Tuple[Tuple, Tuple]:
        aux_data = tuple(getattr(instance, name) for name in field_names)
        children = ()  # No dynamic children since all are static
        return children, aux_data

    # Define the unflatten function
    def unflatten(aux_data: Tuple, children: Tuple) -> Any:
        return cls(*aux_data)

    # Register the class as a pytree node with JAX
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    return cls


# 2. Define the Mesh class using attrs and the decorator
@register_static_pytree
@attrs.define(frozen=True)
class Mesh:
    x: jnp.ndarray
    y: jnp.ndarray
    # Use factory for mutable default fields
    z: jnp.ndarray = attr.field(factory=lambda: jnp.array([0.0]))
    # Example additional fields
    w: jnp.ndarray = attr.field(factory=lambda: jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    description: str = "Default Mesh Description"


# 3. Define the SpaceOperator class with JAX's jit
class SpaceOperator:
    @partial(jax.jit, static_argnums=(0,))  # Marks 'self' as static
    def solve(self, q: jnp.ndarray, mesh: Mesh) -> jnp.ndarray:
        """
        Example operation: Element-wise multiplication of mesh.x with q,
        then add mesh.y for demonstration.
        """
        return mesh.x * q + mesh.y


if __name__ == "__main__":
    # Create mesh data using JAX arrays
    x = jnp.linspace(0, 1, 10)
    y = jnp.linspace(1, 2, 10)
    # Initialize Mesh with x and y; z and w use default_factory
    mesh = Mesh(x, y)

    # Initialize the operator
    space_op = SpaceOperator()

    # Define the input Q as a JAX array
    Q = jnp.linspace(0, 1, 10)

    # Use the solve method
    Q_result = space_op.solve(Q, mesh)

    # Print the result
    print("Q_result:", Q_result)
