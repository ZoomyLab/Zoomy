from attrs import define
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

from zoomy_core.transformation.to_numpy import NumpyRuntimeModel



@define(kw_only=False, slots=True, frozen=True)
class JaxRuntimeModel(NumpyRuntimeModel):
    module = {'ones_like': jnp.ones_like, 'zeros_like': jnp.zeros_like, 'array': jnp.array, 'squeeze': jnp.squeeze}
    printer="jax"
