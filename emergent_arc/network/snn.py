import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from .lif import LIFNeuron, LIFState

class SNNParams(NamedTuple):
    input_weights: jnp.ndarray  # (input_dim, hidden_dim)
    recurrent_weights: jnp.ndarray # (hidden_dim, hidden_dim)
    output_weights: jnp.ndarray # (hidden_dim, output_dim)
    tau_mem: jnp.ndarray # (hidden_dim,)
    tau_syn: jnp.ndarray # (hidden_dim,)

class SNNState(NamedTuple):
    lif_state: LIFState
    output_accum: jnp.ndarray # Accumulate output spikes for decision

class SpikingPolicyNetwork:
    def __init__(self, input_dim: int = 124, hidden_dim: int = 64, output_dim: int = 73):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lif = LIFNeuron(hidden_dim)

    def init_params(self, key: jax.random.PRNGKey) -> SNNParams:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        scale = 1.0 / jnp.sqrt(self.input_dim)
        
        return SNNParams(
            input_weights=jax.random.normal(k1, (self.input_dim, self.hidden_dim)) * scale,
            recurrent_weights=jax.random.normal(k2, (self.hidden_dim, self.hidden_dim)) * (0.1 / jnp.sqrt(self.hidden_dim)),
            output_weights=jax.random.normal(k3, (self.hidden_dim, self.output_dim)) * scale,
            tau_mem=jax.random.uniform(k4, (self.hidden_dim,), minval=10.0, maxval=30.0),
            tau_syn=jax.random.uniform(k5, (self.hidden_dim,), minval=2.0, maxval=10.0)
        )

    def init_state(self) -> SNNState:
        return SNNState(
            lif_state=self.lif.init_state(),
            output_accum=jnp.zeros(self.output_dim)
        )

    def forward(self, params: SNNParams, state: SNNState, input_data: jnp.ndarray) -> Tuple[SNNState, jnp.ndarray]:
        # Input current
        # I = W_in * x + W_rec * prev_spikes
        input_current = jnp.dot(input_data, params.input_weights) + \
                        jnp.dot(state.lif_state.spikes, params.recurrent_weights)
        
        # LIF Step
        lif_params = {'tau_mem': params.tau_mem, 'tau_syn': params.tau_syn}
        new_lif_state = self.lif.step(state.lif_state, input_current, lif_params)
        
        # Output spikes (simple readout: W_out * hidden_spikes)
        # Usually SNNs have output neurons too, but spec says "Output Heads: ... neurons".
        # Let's assume output neurons are also LIF or just linear readout of spikes.
        # Spec says "Output Heads: ... neurons", "Decode decisions from spike counts".
        # So outputs are spiking neurons too.
        # For simplicity, let's treat output as a linear projection of hidden spikes for now, 
        # or better, just accumulate the weighted sum of hidden spikes as "logits" or "potential".
        # Spec: "spike_counts['subroutine'] += output_spikes".
        # This implies output layer IS spiking.
        # Let's add a simple threshold for output layer or just treat W_out * hidden_spikes as output activity.
        # Given "Decode decisions from spike counts", it implies discrete events.
        # Let's assume the output layer is just a linear projection that we interpret as spikes if > threshold,
        # or we can just return the raw projection and accumulate that (rate coding).
        # Spec code: "output_spikes = snn.forward(input_state)".
        # So it returns spikes.
        
        output_current = jnp.dot(new_lif_state.spikes, params.output_weights)
        output_spikes = jnp.where(output_current > 0.5, 1.0, 0.0) # Simple threshold readout
        
        new_output_accum = state.output_accum + output_spikes
        
        return SNNState(lif_state=new_lif_state, output_accum=new_output_accum), output_spikes, output_current
