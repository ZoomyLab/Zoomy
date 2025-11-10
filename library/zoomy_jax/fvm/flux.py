import jax.numpy as jnp
import jax


class Flux:
    def get_flux_operator(self, model):
        pass

class Zero(Flux):
    def get_flux_operator(self, model):
        @jax.jit
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            return jnp.zeros_like(Qi)

        return compute

