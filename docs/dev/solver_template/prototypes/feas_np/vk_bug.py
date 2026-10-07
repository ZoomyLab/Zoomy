"""Re-inject the PRE-FIX jax Venkatakrishnan negative branch (HEAD 6716f8b) at
runtime, for A/B measurement only.  No file under the repo is touched."""
import jax.numpy as jnp
from zoomy_jax.fvm import reconstruction_jax as RJ

_orig = RJ.LSQMUSCLReconstructionJAX._limit_vk

def _limit_vk_buggy(self, u, grad_v, u_min, u_max):
    nc = self._nc; eps = 1e-30; ev2 = self._eps_v2
    dA_int, dB_int = self._face_deltas_interior(grad_v)
    dA_bnd = self._face_deltas_boundary(grad_v)
    def _apply(cell_ids, deltas):
        uc = u[cell_ids]
        dm_pos = u_max[cell_ids] - uc
        dm_neg = u_min[cell_ids] - uc
        num_p = dm_pos ** 2 + ev2[cell_ids] + 2 * deltas * dm_pos
        den_p = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[cell_ids]
        cand_p = jnp.clip(num_p / jnp.maximum(den_p, eps), 0.0, 1.0)
        num_n = dm_neg ** 2 + ev2[cell_ids] + 2 * deltas * dm_neg
        den_n = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[cell_ids]
        cand_n = jnp.clip(-num_n / jnp.minimum(den_n, -eps), 0.0, 1.0)   # HEAD (buggy)
        cand = jnp.where(deltas > eps, cand_p, jnp.where(deltas < -eps, cand_n, 1.0))
        return jnp.ones(nc).at[cell_ids].min(cand)
    phi = jnp.minimum(_apply(self._iA_int, dA_int), _apply(self._iB_int, dB_int))
    phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
    return jnp.clip(phi, 0.0, 1.0)

def set_buggy(on):
    RJ.LSQMUSCLReconstructionJAX._limit_vk = _limit_vk_buggy if on else _orig
    # the instance caches the bound limiter fn at construction -> rebuild recon objects after toggling
