import jax
import jax.numpy as jnp
from typing import NamedTuple

class LIFState(NamedTuple):
    membrane: jnp.ndarray
    synaptic: jnp.ndarray
    spikes: jnp.ndarray

class LIFNeuron:
    def __init__(self, size: int, tau_mem: float = 20.0, tau_syn: float = 5.0, threshold: float = 1.0):
        self.size = size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.threshold = threshold

    def init_state(self) -> LIFState:
        return LIFState(
            membrane=jnp.zeros(self.size),
            synaptic=jnp.zeros(self.size),
            spikes=jnp.zeros(self.size)
        )

    def step(self, state: LIFState, input_current: jnp.ndarray, params: dict) -> LIFState:
        # Allow learnable time constants if provided in params
        tau_mem = params.get('tau_mem', self.tau_mem)
        tau_syn = params.get('tau_syn', self.tau_syn)
        
        # Synaptic dynamics
        # dI/dt = -I/tau_syn + input
        new_synaptic = state.synaptic + (-state.synaptic / tau_syn) + input_current
        
        # Membrane dynamics
        # dV/dt = -V/tau_mem + I
        new_membrane = state.membrane + (-state.membrane / tau_mem) + new_synaptic
        
        # Spike generation
        spikes = jnp.where(new_membrane >= self.threshold, 1.0, 0.0)
        
        # Reset membrane after spike (soft reset or hard reset)
        # Hard reset: V = 0
        new_membrane = jnp.where(spikes > 0, 0.0, new_membrane)
        
        return LIFState(membrane=new_membrane, synaptic=new_synaptic, spikes=spikes)
