import cirq
import numpy as np
from qas_gym.envs.qas_env import QuantumArchSearchEnv
from qas_gym.utils import *


class NoisyNQubitEnv(QuantumArchSearchEnv):
    def __init__(
        self,
        target: np.ndarray,
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
        error_single: float = 0.015,
        error_multi: float = 0.03,
        error_gates: float  = 0.0,
        error_observables = None
    ):
        n_qubits = int(np.log2(len(target)))
        qubits = cirq.LineQubit.range(n_qubits)
        state_observables = get_default_observables(qubits)
        action_gates = get_default_gates(qubits)
        super(NoisyNQubitEnv, self).__init__(target,
                                             qubits,
                                             state_observables,
                                             action_gates,
                                             fidelity_threshold,
                                             reward_penalty,
                                             max_timesteps,
                                             error_observables=error_rate,
                                             error_gates=error_rate,
                                             error_single=error_single,
                                             error_multi=error_multi,)


class NoisyTwoQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_bell_state(),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
    ):
        assert len(target) == 4, 'Target must be of size 4'
        super(NoisyTwoQubitEnv,
              self).__init__(target, fidelity_threshold, reward_penalty,
                             max_timesteps, error_rate)


class NoisyThreeQubitEnv(NoisyNQubitEnv):
    def __init__(
        self,
        target: np.ndarray = get_ghz_state(3),
        fidelity_threshold: float = 0.95,
        reward_penalty: float = 0.01,
        max_timesteps: int = 20,
        error_rate: float = 0.001,
        error_single: float = 0.015,
        error_multi: float = 0.03,  # 噪声
    ):
        assert len(target) == 8, 'Target must be of size 8'
        super(NoisyThreeQubitEnv,
              self).__init__(target, fidelity_threshold, reward_penalty,
                             max_timesteps, error_rate, error_single, error_multi,)
