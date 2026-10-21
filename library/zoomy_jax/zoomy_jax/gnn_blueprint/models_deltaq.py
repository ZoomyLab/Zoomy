from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import jraph


@dataclass
class ModelConfig:
    n_fields: int = 3


class DeltaQGNN(eqx.Module):
    w_self: jnp.ndarray
    w_msg: jnp.ndarray
    w_edge: jnp.ndarray
    w_aux: jnp.ndarray
    w_coarse: jnp.ndarray
    w_gate: jnp.ndarray
    b: jnp.ndarray

    def __init__(self, n_fields: int):
        self.w_self = jnp.linspace(0.05, 0.09, n_fields)
        self.w_msg = jnp.linspace(0.02, 0.04, n_fields)
        self.w_edge = jnp.linspace(0.01, 0.03, n_fields)
        self.w_aux = jnp.linspace(0.015, 0.025, n_fields)
        self.w_coarse = jnp.linspace(0.02, 0.05, n_fields)
        self.w_gate = jnp.linspace(0.8, 1.2, n_fields)
        self.b = jnp.linspace(0.005, 0.008, n_fields)

    def __call__(self, graph: jraph.GraphsTuple, q: jnp.ndarray, dt: jnp.ndarray) -> jnp.ndarray:
        # q shape: (n_fields, n_nodes)
        edge_scalar = graph.edges[:, 0]

        dqs = []
        for i in range(q.shape[0]):
            msg_in = q[i, graph.senders] + self.w_edge[i] * edge_scalar
            msg = jraph.segment_sum(msg_in, graph.receivers, num_segments=q.shape[1])
            rhs = self.w_self[i] * q[i] + self.w_msg[i] * msg + self.b[i]
            dqs.append(dt * rhs)
        return jnp.stack(dqs, axis=0)
