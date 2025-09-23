import sys
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union, Tuple

from collections import deque
import cirq
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import os
from cirq.devices import InsertionNoiseModel
import scipy

from qas_gym.utils import get_default_gates
from qas_gym.utils import FGate,CXRVGate
import warnings
class QuantumArchSearchEnv(gym.Env):
    metadata = {'render_modes': ['ansi', 'human']}

    def __init__(
            self,
            target: np.ndarray,
            qubits: List[cirq.LineQubit],
            state_observables: List[cirq.GateOperation],
            action_gates: List[cirq.GateOperation],
            fidelity_threshold: float,
            reward_penalty: float,
            max_timesteps: int,
            error_observables: Optional[float] = None,
            error_gates: Optional[float] = None,
            seed: Optional[int] = None,  # 添加seed参数
            error_single=None, error_multi=None, #噪声
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.qubits = qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates
        # 设置种子
        self.seed(seed)  # 在这里调用seed方法
        # 窗口设置
        self.min_gate_count = max_timesteps

        self.state = None
        self.window_size = 6  # 设置时间窗口大小
        self.operation_history = deque(maxlen=self.window_size)  # 记录操作历史
        self.previous_action = None
        self.penalty_weight = 1.  # 惩罚权重
        self.state_history = None
        self.previous_state = None
        self.previous_fidelity = None
        self.previous_entanglement = None
        self.reward = 0

        self.error_single = error_single # 噪声添加
        self.error_multi = error_multi

        self.simulator = cirq.Simulator()

        self.s_tot_norm_tar, self.s_a_norm_tar, self.s_b_norm_tar = self._get_renyi_target()

        # set spaces
        self.observation_space = spaces.Dict({
            "expectations": spaces.Box(low=-1., high=1., shape=(len(state_observables),), dtype=np.float32),
            "gate_count": spaces.Discrete(self.max_timesteps + 1),
            #"entanglement": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            #"entanglement_t": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),
            "entanglement_a": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32),#duobite
            #"entanglement_b": spaces.Box(low=0., high=1., shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(n=len(action_gates))

        self.action_gates = get_default_gates(qubits)

        self.seed()
        self.render_mode = None

    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_gates = []
        #self.state = None
        #self.state_history = None
        #self.operation_history.clear()  # 清空操作历史
        self.previous_action = None  # 重置上一个动作
        #self.previous_fidelity = None
        #self.previous_entanglement = None
        #self.circuit_gates.append(cirq.H(self.qubits[0]))
        #self.circuit_gates.append(cirq.CNOT(self.qubits[0],self.qubits[1]))
        #self.circuit_gates.append(cirq.CNOT(self.qubits[1],self.qubits[2]))

        return self._get_obs(), {}

    def _get_cirq3(self, maybe_add_noise=False):
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = cirq.depolarize(
                    self.error_gates).on_each(*gate._qubits)
                circuit.append(noise_gate)
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(
                self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_cirq(self, maybe_add_noise=False):
        # 初始化电路时为每个量子比特添加单位操作
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
            if maybe_add_noise:
                # 尝试获取门作用的量子比特列表，优先使用 _qubits 属性，否则使用 qubits 属性
                if hasattr(gate, "_qubits"):
                    qubits = gate._qubits
                else:
                    qubits = gate.qubits
                # 根据门作用的量子比特数量来区分噪声模型
                if len(qubits) == 1 and (self.error_single is not None):
                    # 对单比特门添加噪声（例如使用 depolarize 噪声）
                    noise_gate = cirq.depolarize(self.error_single).on_each(*qubits)
                    circuit.append(noise_gate)
                elif len(qubits) > 1 and (self.error_multi is not None):
                    # 对多比特门添加噪声
                    noise_gate = cirq.depolarize(self.error_multi).on_each(*qubits)
                    circuit.append(noise_gate)
        # 对观测操作添加噪声
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit


    def _get_obs(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        obs_exp = self.simulator.simulate_expectation_values(
            circuit, observables=self.state_observables)
        gate_count = len(self.circuit_gates)
        current_con = self.current_concurrence()
        s_tot_norm, s_a_norm, s_b_norm = self._get_renyi()#duobite

        # Ensure the data is float32 and clip if necessary
        expectations = np.array(obs_exp).real.astype(np.float32)
        expectations = np.clip(expectations, -1.0, 1.0)

        obs = {
            "expectations": expectations,
            "gate_count": int(gate_count),
            #"entanglement": np.array([current_con], dtype=np.float32)
            #"entanglement_t": np.array([s_tot_norm], dtype=np.float32),
            "entanglement_a": np.array([s_a_norm], dtype=np.float32),#(多比特只有这个A，没有B)
            #"entanglement_b": np.array([s_b_norm], dtype=np.float32)
        }

        # Ensure entanglement is within [0, 1] range
        #obs["entanglement"] = np.clip(obs["entanglement"], 0.0, 1.0)

        return obs

    def _get_fidelity(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        pred = self.simulator.simulate(circuit).final_state_vector
        inner = np.inner(np.conj(pred), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    def set_attr(self, attr_name, value):
        setattr(self, attr_name, value)

    def step(self, action):
        # 初始化，判断历史信息状态
        #if self.state is None and self.state_history is None:
        #    self.state_history = []
        #    self.state = self._get_current_state()
        #    self.state_history.append(np.copy(self.state))
        #if self.state_history == []:
        #    self.state_history.append(np.copy(self.state))
        #if self.previous_fidelity is None:
        #    self.previous_fidelity = self._get_fidelity()
        #    self.previous_entanglement = self.current_concurrence()
        #pre_diff_E = abs(self.previous_entanglement - self.target_concurrence())#计算原有差值

        action_gate = self.action_gates[action]  # 原有，选择动作
        self.circuit_gates.append(action_gate)  # 原有，执行动作

        # compute observation
        observation = self._get_obs()

        # compute fidelity
        #fidelity = self._get_fidelity()# qubit <= 3,ues it
        fidelity = self.fidelity_measure()#多比特使用这个,qubit>3,ues it

        #diff_F = fidelity - self.previous_fidelity
        #cur_diff_E = abs(self.current_concurrence() - self.target_concurrence())
        #diff_E = -(cur_diff_E - pre_diff_E)
        #self.previous_fidelity = fidelity
        #self.previous_entanglement = self.current_concurrence()

        # 更新状态
        #self.state_history.append(np.copy(self.state))
        #penalty = self.calc_history_fidelity()

        #target_con = self.target_concurrence()  # 3bit,use it
        #current_con = self.current_concurrence()  # 3bit,use it

        # 初始化 entangle_list
        if not hasattr(self, 'entangle_list'):# Affect efficiency
            self.entangle_list = []  # 如果不存在则初始化
        # 将 entangle_list 传递给 entangle 方法
        self.entangle(self.entangle_list)

        if not hasattr(self, 'fidelity_list'):#Affect efficiency
            self.fidelity_list = []  # 如果不存在则初始化
        # 将 fidelity_list 传递给 fidelity_getlist 方法
        self.fidelity_getlist(self.fidelity_list)

        #if not hasattr(self, 'fidelity_values'):
        #    self.fidelity_values = []
        #if not hasattr(self, 'concurrence_values'):
        #    self.concurrence_values = []
        #if not hasattr(self, 'fidelity_values_mean_list'):
        #    self.fidelity_values_mean_list = []
        #if not hasattr(self, 'concurrence_values_mean_list'):
            #self.concurrence_values_mean_list = []

        max_gate_count = self.max_timesteps
        gate_count = len(self.circuit_gates)
        #con_difference = abs(current_con - target_con)
        s_tot_norm, s_a_norm, s_b_norm = self._get_renyi()#多比特加入这个
        #con_difference = (abs(self.s_tot_norm_tar - s_tot_norm) + abs(self.s_a_norm_tar - s_a_norm) + abs(self.s_b_norm_tar - s_b_norm))/3.
        con_difference = abs(self.s_a_norm_tar - s_a_norm)#（多比特只用计算这个一项）
        #con_difference = self._mixed_state_ent_diff()

        gate_score = gate_count / max_gate_count

        exploration_weight = 0.2 * (1 - fidelity)  # 0.2
        exploration_threshold = 0.6
        gate_penalty_factor = 1.
        #reward = diff_Fs

        if fidelity >= self.fidelity_threshold:
            reward = (-1.5 * gate_score + fidelity * 2.2 + (1 - exploration_weight) * (
                        1 - con_difference))
        elif self.fidelity_threshold > fidelity >= exploration_threshold:
            reward = ((0.5 * (1 - exploration_weight) * fidelity - 1. * gate_score) - 1. + 0.5 * (
                        1 - exploration_weight) * (1 - con_difference)) * 0.5
        elif exploration_threshold > fidelity >= 0.4:
            reward = ((- 0.75 * gate_score + exploration_weight * (fidelity)) - 1. + (exploration_weight) * (
                        1 - con_difference)) * 0.5
        else:
            reward = ((- 1. * (exploration_weight) * (
                    1 - fidelity) - 0.5 * gate_score) - 1.) * 0.5
        terminal = (fidelity > self.fidelity_threshold) or (
                len(self.circuit_gates) >= self.max_timesteps)
        truncated = False

        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_cirq(), 'concurrence_diff': con_difference, #current_con,
                'goal_achieved': terminal, }
        return observation, reward, terminal, truncated, info

    def entangle(self, entangle_list):
        entangle_list.append(self.current_concurrence())
        return entangle_list

    def fidelity_getlist(self, fidelity_list):
        fidelity_list.append(self._get_fidelity())
        return fidelity_list

    def _get_current_state(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        state = self.simulator.simulate(circuit).final_state_vector
        return state

    def _update_last_state(self, current_state):
        self.last_state = current_state

    def _update_last_last_state(self, last_state):
        self.last_last_state = last_state

    def render(self, mode='human'):
        if self.render_mode is not None:
            if mode != self.render_mode:
                raise AttributeError(
                    f"render mode asked to render with {mode} but environment has been initialized to render with {self.render_mode}."
                )
        else:
            self.render_mode = mode

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        return None

    def target_concurrence(self):
        d1 = (self.target[0] * self.target[0] * self.target[7] * self.target[7] + self.target[1] * self.target[1] *
              self.target[6] * self.target[6] + self.target[2] *
              self.target[2] * self.target[5] * self.target[5] + self.target[4] * self.target[4] * self.target[3] *
              self.target[3])
        d2 = (self.target[0] * self.target[7] * self.target[3] * self.target[4] + self.target[0] * self.target[7] *
              self.target[5] * self.target[2] + self.target[
                  0] * self.target[7] * self.target[6] * self.target[1]
              + self.target[3] * self.target[4] * self.target[5] * self.target[2] + self.target[3] * self.target[4] *
              self.target[6] * self.target[1]
              + self.target[5] * self.target[2] * self.target[6] * self.target[1])
        d3 = self.target[0] * self.target[6] * self.target[5] * self.target[3] + self.target[7] * self.target[1] * \
             self.target[2] * self.target[4]
        target_con = 4 * abs(d1 - 2 * d2 + 4 * d3)
        return target_con

    def current_concurrence(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        pred = self.simulator.simulate(circuit).final_state_vector
        d1 = (pred[0] * pred[0] * pred[7] * pred[7] + pred[1] * pred[1] * pred[6] * pred[6] + pred[2] *
              pred[2] * pred[5] * pred[5] + pred[4] * pred[4] * pred[3] * pred[3])
        d2 = (pred[0] * pred[7] * pred[3] * pred[4] + pred[0] * pred[7] * pred[5] * pred[2] + pred[0] * pred[7] * pred[
            6] * pred[1]
              + pred[3] * pred[4] * pred[5] * pred[2] + pred[3] * pred[4] * pred[6] * pred[1]
              + pred[5] * pred[2] * pred[6] * pred[1])
        d3 = pred[0] * pred[6] * pred[5] * pred[3] + pred[7] * pred[1] * pred[2] * pred[4]
        current_con = 4 * abs(d1 - 2 * d2 + 4 * d3)
        return current_con

    def current_unitary(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        current_unitary = cirq.unitary(circuit)
        return current_unitary


    def fidelity_measure(self):
        """
        计算当前含噪声电路产生的混合态与预定义目标纯态 self.target 之间的保真度。
        """
        # 1. 构建并获取包含噪声的电路
        try:
            circuit_with_noise = self._get_cirq(maybe_add_noise=True)
            if not isinstance(circuit_with_noise, cirq.Circuit):
                 print("Warning: _get_cirq did not return a cirq.Circuit object.")
                 return 0.0 # 或者抛出错误
        except Exception as e:
            print(f"Error getting circuit with noise: {e}")
            return 0.0 # 返回 0 保真度表示失败

        # 检查目标状态向量是否有效
        if self.target is None or not isinstance(self.target, np.ndarray):
             print("Error: Target state vector (self.target) is not defined or not a NumPy array.")
             return 0.0

        # 2. 使用密度矩阵模拟器模拟含噪声电路
        try:
            simulator = cirq.DensityMatrixSimulator(seed=42) # 可以考虑是否需要固定种子
            result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits) # 推荐指定 qubit_order
            density_matrix = result.final_density_matrix

            # 可选：强制厄米性以提高数值稳定性
            density_matrix = (density_matrix + density_matrix.conj().T) / 2
            density_matrix = density_matrix / np.trace(density_matrix)

        except Exception as e:
            print(f"Error during density matrix simulation: {e}")
            print(f"Circuit that caused error:\n{circuit_with_noise}")
            return 0.0 # 返回 0 保真度表示失败

        # 3. 计算最终密度矩阵与目标状态向量之间的保真度
        try:
            # cirq.fidelity 可以直接处理密度矩阵和状态向量
            fidelity_M = cirq.fidelity(density_matrix, self.target)

            # 保真度理论上应该在 [0, 1] 之间，但数值计算可能导致微小偏差
            fidelity_M = np.clip(np.real(fidelity_M), 0.0, 1.0) # 取实部并限制范围

        except ValueError as e:
             print(f"Error calculating fidelity: {e}")
             print(f"Density matrix shape: {density_matrix.shape}")
             print(f"Target vector shape: {self.target.shape}")
             # 检查维度是否匹配
             num_qubits = len(self.qubits)
             expected_dim = 2**num_qubits
             if density_matrix.shape != (expected_dim, expected_dim) or self.target.shape != (expected_dim,):
                 print(f"Dimension mismatch: Expected ({expected_dim}, {expected_dim}) and ({expected_dim},)")
             return 0.0
        except Exception as e:
             print(f"Unexpected error calculating fidelity: {e}")
             return 0.0


        return fidelity_M

    def _mixed_state_ent_diff(self):
        circuit_with_noise = self._get_cirq(maybe_add_noise=True)
        simulator = cirq.DensityMatrixSimulator(seed=42)  # 固定种子
        result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits)  # 指定 qubit_order
        density_matrix = result.final_density_matrix

        # 强制厄米性以提高数值稳定性
        density_matrix = (density_matrix + density_matrix.conj().T) / 2
        #print("\n计算带噪声状态的对数负度:")
        ln_noisy_0_vs_12 = self.calculate_log_negativity(density_matrix, [0], len(self.qubits))
        #print(f"  划分 (0 | 12): {ln_noisy_0_vs_12:.4f}")
        ln_noisy_1_vs_02 = self.calculate_log_negativity(density_matrix, [1], len(self.qubits))
        #print(f"  划分 (1 | 02): {ln_noisy_1_vs_02:.4f}")
        ln_noisy_2_vs_01 = self.calculate_log_negativity(density_matrix, [2], len(self.qubits))
        #print(f"  划分 (2 | 01): {ln_noisy_2_vs_01:.4f}")

        target_matrix = cirq.density_matrix_from_state_vector(self.target)
        t_0_vs_12 = self.calculate_log_negativity(target_matrix, [0], len(self.qubits))
        t_1_vs_02 = self.calculate_log_negativity(target_matrix, [1], len(self.qubits))
        t_2_vs_01 = self.calculate_log_negativity(target_matrix, [2], len(self.qubits))

        diff_ln = abs(ln_noisy_0_vs_12 - t_0_vs_12) + abs(ln_noisy_1_vs_02 - t_1_vs_02) + abs(ln_noisy_2_vs_01 - t_2_vs_01)
        diff_ln = diff_ln/3

        return diff_ln

    def partial_transpose(self,density_matrix: np.ndarray,
                          subsystem_A_indices: List[int],
                          subsystem_dims: List[int]) -> np.ndarray:
        """计算密度矩阵相对于子系统 A 的部分转置。"""
        num_subsystems = len(subsystem_dims)
        total_dim = np.prod(subsystem_dims)
        if density_matrix.shape != (total_dim, total_dim):
            raise ValueError(f"密度矩阵形状 {density_matrix.shape} 与 "
                             f"子系统维度 {subsystem_dims} (总维度 {total_dim}) 不符。")
        reshaped_dm = density_matrix.reshape(subsystem_dims + subsystem_dims)
        axes = list(range(2 * num_subsystems))
        for idx in subsystem_A_indices:
            axes[idx], axes[idx + num_subsystems] = axes[idx + num_subsystems], axes[idx]
        pt_tensor = np.transpose(reshaped_dm, axes=axes)
        pt_matrix = pt_tensor.reshape((total_dim, total_dim))
        # print(pt_matrix)
        return pt_matrix

    def calculate_log_negativity(self,density_matrix: np.ndarray,
                                 subsystem_A_indices: List[int],
                                 num_qubits: int,
                                 tol: float = 1e-9) -> float:
        """计算给定二分划下的对数负度。"""
        if density_matrix.shape != (2 ** num_qubits, 2 ** num_qubits):
            raise ValueError(f"密度矩阵形状应为 ({2 ** num_qubits}, {2 ** num_qubits})")
        subsystem_dims = [2] * num_qubits
        pt_rho = self.partial_transpose(density_matrix, subsystem_A_indices, subsystem_dims)
        singular_values = scipy.linalg.svdvals(pt_rho)
        trace_norm = np.sum(singular_values)
        if trace_norm <= 1.0 + tol:
            return 0.0
        else:
            log_negativity = np.log2(trace_norm)
            return max(0.0, log_negativity)

    def calculate_renyi_entropy(self,rho: np.ndarray, alpha: float, tol: float = 1e-12) -> float:
        """
        计算密度矩阵 rho 的 alpha 阶 Rényi 熵 S_alpha(rho)。
        Includes handling for alpha=0, 1, inf and numerical stability.
        """
        if alpha < 0:
            raise ValueError("Rényi 熵的阶数 alpha 必须是非负的。")
        rho = np.asarray(rho)
        if not np.allclose(rho, rho.T.conj(), atol=tol):  # Use tolerance for check
            warnings.warn("输入矩阵可能不是厄米矩阵（在容忍度内）。")
            # Force Hermiticity? Maybe just warn.
            # rho = 0.5 * (rho + rho.T.conj())

        # Handle alpha = 1 (Von Neumann)
        if np.isclose(alpha, 1.0):
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            positive_eigenvalues = eigenvalues[eigenvalues > tol]
            if len(positive_eigenvalues) == 0: return 0.0
            try:
                log_eigs = np.log2(positive_eigenvalues)
                von_neumann_entropy = -np.sum(positive_eigenvalues * log_eigs)
            except (ValueError, FloatingPointError):
                warnings.warn("数值误差导致 log2 计算问题 for S_1。")
                return np.nan
            if np.isnan(von_neumann_entropy) or np.isinf(von_neumann_entropy):
                warnings.warn("冯诺依曼熵计算结果为 NaN 或 Inf。")
                # Return NaN instead of potential Inf/NaN from sum
                return np.nan
            return von_neumann_entropy

        # Handle alpha = 0 (Hartley/Max-entropy)
        if np.isclose(alpha, 0.0):
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            rank = np.sum(eigenvalues > tol)
            if rank == 0: return 0.0
            return np.log2(rank)

        # Handle alpha = inf (Min-entropy)
        if alpha == np.inf:
            try:
                eigenvalues = np.linalg.eigvalsh(rho)
            except np.linalg.LinAlgError:
                warnings.warn("特征值计算失败。")
                return np.nan
            max_eigenvalue = np.max(eigenvalues)
            if max_eigenvalue <= tol:
                warnings.warn("最大特征值接近零或为负，最小熵未定义或无穷大。")
                return np.inf
            try:
                # Ensure argument to log2 is positive real
                min_entropy = -np.log2(max_eigenvalue.real)
            except (ValueError, FloatingPointError):
                warnings.warn("数值误差导致 log2 计算问题 for S_inf。")
                return np.nan
            if np.isnan(min_entropy) or np.isinf(min_entropy):
                warnings.warn("最小熵计算结果为 NaN 或 Inf。")
                return np.nan  # Return NaN for consistency
            return min_entropy

        # General alpha != 0, 1, inf
        try:
            eigenvalues = np.linalg.eigvalsh(rho)
        except np.linalg.LinAlgError:
            warnings.warn("特征值计算失败。")
            return np.nan

        positive_eigenvalues = eigenvalues[eigenvalues > tol].real  # Take real part
        positive_eigenvalues = positive_eigenvalues[positive_eigenvalues > tol]  # Filter again

        if len(positive_eigenvalues) == 0:
            warnings.warn("密度矩阵没有正特征值，Tr(rho^alpha) 为 0。")
            if alpha < 1:
                return -np.inf  # Or potentially 0 depending on convention
            else:
                return np.inf  # Or potentially 0

        try:
            # Check for potential overflow with large alpha?
            with np.errstate(over='raise'):
                powered_eigs = positive_eigenvalues ** alpha
            trace_rho_alpha = np.sum(powered_eigs)
        except FloatingPointError:
            warnings.warn(f"计算 Tr(rho^alpha) = sum(lambda^alpha) 时出错 (可能溢出)。")
            return np.nan  # Or Inf? Depends on alpha

        if trace_rho_alpha <= tol or not np.isfinite(trace_rho_alpha):
            warnings.warn(f"Tr(rho^alpha) 计算结果 {trace_rho_alpha} 非常接近于零、为负或非有限。")
            if alpha < 1:
                return -np.inf
            else:
                return np.inf

        try:
            log_trace = np.log2(trace_rho_alpha)
        except (ValueError, FloatingPointError):
            warnings.warn(f"计算 log2(Tr(rho^alpha)) 时出错。")
            return np.nan

        if np.isclose(alpha, 1.0):  # Should not happen here, but as safety
            warnings.warn("Alpha=1 case handled improperly in general calculation.")
            return np.nan
        renyi_entropy = (1.0 / (1.0 - alpha)) * log_trace

        if np.isnan(renyi_entropy):  # Don't check for Inf here, Inf is a possible valid output
            warnings.warn(f"Rényi entropy calculation resulted in NaN for alpha={alpha}.")
            pass  # Keep NaN

        return renyi_entropy

    # Manual Partial Trace
    def manual_partial_trace(self,rho: np.ndarray, trace_over_indices: List[int], dims: List[int]) -> np.ndarray:
        """
        Manually performs partial trace using numpy.einsum.
        """
        num_subsystems = len(dims)
        if not all(d >= 1 for d in dims):
            raise ValueError("所有子系统维度必须至少为 1。")
        if not rho.shape == (np.prod(dims), np.prod(dims)):
            raise ValueError("rho 维度与 dims 不匹配。")
        if not all(0 <= i < num_subsystems for i in trace_over_indices):
            raise ValueError("trace_over_indices 包含无效索引。")

        kept_indices = [i for i in range(num_subsystems) if i not in trace_over_indices]

        tensor_shape = tuple(dims + dims)
        try:
            tensor_rho = rho.reshape(tensor_shape)
        except ValueError as e:
            print(f"Reshaping rho (shape {rho.shape}) to {tensor_shape} failed: {e}")
            raise e

        all_indices_list = list(range(2 * num_subsystems))
        kept_axes_rows = kept_indices
        kept_axes_cols = [i + num_subsystems for i in kept_indices]
        output_indices_list = kept_axes_rows + kept_axes_cols

        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if 2 * num_subsystems > len(alphabet):
            raise ValueError("子系统过多，无法使用默认字母表进行 einsum。")

        input_map = {axis: alphabet[i] for i, axis in enumerate(all_indices_list)}
        output_map = {axis: alphabet[i] for i, axis in
                      enumerate(output_indices_list)}  # Use different indices for output map

        input_str_list = [input_map[i] for i in all_indices_list]
        output_str_list = [output_map[i] for i in output_indices_list]  # Map to output letters

        # Handle trace indices: make row and column indices the same letter in input
        current_input_str_list = list(input_str_list)  # Copy to modify
        for i in trace_over_indices:
            row_axis_idx = i
            col_axis_idx = i + num_subsystems
            trace_letter = current_input_str_list[row_axis_idx]
            current_input_str_list[col_axis_idx] = trace_letter  # Assign same letter

        # Reconstruct strings
        input_str = "".join(current_input_str_list)
        # Output string uses letters corresponding to kept indices only
        final_output_str_list = []
        current_output_idx = 0
        for i in kept_axes_rows:
            final_output_str_list.append(alphabet[current_output_idx])
            current_output_idx += 1
        for i in kept_axes_cols:
            final_output_str_list.append(alphabet[current_output_idx])
            current_output_idx += 1
        output_str = "".join(final_output_str_list)

        # Ensure the output letters match the positions correctly
        # Rebuild output string mapping kept row/col letters to a compact alphabet a,b,c...
        final_output_str_map = {}
        out_idx = 0
        for i in kept_axes_rows:
            final_output_str_map[input_map[i]] = alphabet[out_idx]
            out_idx += 1
        for i in kept_axes_cols:
            final_output_str_map[input_map[i]] = alphabet[out_idx]
            out_idx += 1
        output_str = "".join(alphabet[k] for k in range(out_idx))

        # Rebuild input string with trace indices identified by repetition
        final_input_str_list = list(alphabet[:2 * num_subsystems])  # Start with unique letters
        for i in trace_over_indices:
            final_input_str_list[i + num_subsystems] = final_input_str_list[i]  # Make letters same
        input_str = "".join(final_input_str_list)

        # Now build the output string using only the letters corresponding to kept indices
        final_output_str_list_v2 = []
        for i in kept_axes_rows:
            final_output_str_list_v2.append(final_input_str_list[i])
        for i in kept_axes_cols:
            final_output_str_list_v2.append(final_input_str_list[i])
        output_str = "".join(final_output_str_list_v2)

        einsum_str = f"{input_str}->{output_str}"
        # print(f"Debug: Einsum string: {einsum_str}") # Uncomment for debugging

        try:
            traced_tensor = np.einsum(einsum_str, tensor_rho, optimize='greedy')  # Added optimize flag
        except ValueError as e:
            print(f"Einsum failed with string '{einsum_str}': {e}")
            raise e

        dim_kept = np.prod([dims[i] for i in kept_indices])
        if dim_kept == 0:  # Handle case where all subsystems are traced out
            if traced_tensor.ndim == 0:  # Should be a scalar
                return traced_tensor  # Return the scalar trace result
            else:
                raise ValueError("Trace result has unexpected dimension when all traced.")

        # Ensure the result dimension matches expectation
        expected_output_dim = dim_kept * dim_kept
        if traced_tensor.size != expected_output_dim:
            # If einsum produced a scalar (dim_kept=1), reshape needs size 1
            if dim_kept == 1 and traced_tensor.size == 1:
                rho_reduced = traced_tensor.reshape(1, 1)
            else:
                print(f"Einsum output size {traced_tensor.size} != expected {expected_output_dim}")
                print(f"Einsum string: {einsum_str}")
                print(f"Input shape: {tensor_rho.shape}")
                print(f"Output shape: {traced_tensor.shape}")
                print(f"Kept indices: {kept_indices}, Dims: {dims}, Dim_kept: {dim_kept}")

                # Fallback to np.trace if einsum logic is flawed
                # This is much less efficient for large numbers of traces
                # warnings.warn("Einsum result dimension mismatch, potential issue in string generation.")
                # This fallback is complex to write correctly on the fly. Sticking with einsum debug.
                raise ValueError("Einsum result dimension mismatch.")

        else:
            rho_reduced = traced_tensor.reshape(dim_kept, dim_kept)

        return rho_reduced

    def calculate_system_and_subsystem_renyi(self,rho: np.ndarray, num_qubits: int, alpha: float) -> Tuple[
        float, float, float]:
        """
        计算整个系统以及中间划分的两个子系统的 **归一化** Rényi 熵。
        归一化因子是各自子系统的最大可能熵 log2(d) = num_qubits_in_subsystem。
        """
        expected_dim = 2 ** num_qubits
        if rho.shape != (expected_dim, expected_dim):
            raise ValueError(f"密度矩阵维度 {rho.shape} 与 {num_qubits} 个 qubit "
                             f"(期望维度 {expected_dim}x{expected_dim}) 不符。")

        # --- 1. 计算原始（未归一化）熵 ---
        s_total_raw = self.calculate_renyi_entropy(rho, alpha)

        # --- 2. 定义划分和维度 ---
        num_qubits_A = num_qubits // 2
        num_qubits_B = num_qubits - num_qubits_A
        keep_indices_A = list(range(num_qubits_A))
        keep_indices_B = list(range(num_qubits_A, num_qubits))
        trace_over_indices_A = keep_indices_B  # Indices to trace out to get rho_A
        trace_over_indices_B = keep_indices_A  # Indices to trace out to get rho_B

        s_A_raw = np.nan  # Initialize
        s_B_raw = np.nan

        if num_qubits >= 2:  # Only calculate subsystem entropy if partitioning is possible
            all_qubit_dims = [2] * num_qubits

            # --- 3/4. 计算子系统 A 的熵 ---
            try:
                # 使用 manual_partial_trace (追踪掉 B)
                rho_A = self.manual_partial_trace(rho, trace_over_indices_A, all_qubit_dims)
                s_A_raw = self.calculate_renyi_entropy(rho_A, alpha)
            except Exception as e:
                warnings.warn(f"计算子系统 A (保留 {keep_indices_A}) 的熵时出错: {e}")
                s_A_raw = np.nan  # Mark as failed

            # --- 5/6. 计算子系统 B 的熵 ---
            try:
                # 使用 manual_partial_trace (追踪掉 A)
                rho_B = self.manual_partial_trace(rho, trace_over_indices_B, all_qubit_dims)
                s_B_raw = self.calculate_renyi_entropy(rho_B, alpha)
            except Exception as e:
                warnings.warn(f"计算子系统 B (保留 {keep_indices_B}) 的熵时出错: {e}")
                s_B_raw = np.nan  # Mark as failed

        elif num_qubits == 1:
            # For 1 qubit, S_A = S_total, S_B is not applicable (or trace of 0-dim)
            s_A_raw = s_total_raw
            s_B_raw = np.nan  # Subsystem B doesn't exist meaningfully
            warnings.warn("1-qubit system: S_B is NaN.")
        else:  # num_qubits == 0
            s_A_raw = s_total_raw  # Should be 0
            s_B_raw = s_total_raw  # Should be 0

        # --- 7. 计算归一化因子 (最大熵) ---
        # log2(d) = num_qubits
        max_s_total = float(num_qubits) if num_qubits > 0 else 0.0
        max_s_A = float(num_qubits_A) if num_qubits_A > 0 else 0.0
        max_s_B = float(num_qubits_B) if num_qubits_B > 0 else 0.0

        # --- 8. 进行归一化 ---
        # Handle division by zero and non-finite raw entropies
        norm_s_total = np.nan
        if np.isfinite(s_total_raw):
            if max_s_total > 1e-9:  # Use tolerance for float comparison
                norm_s_total = s_total_raw / max_s_total
            elif np.isclose(s_total_raw, 0.0):  # Handle 0/0 case -> 0
                norm_s_total = 0.0
            # else: raw entropy is non-zero but max is zero -> leave as NaN

        norm_s_A = np.nan
        if np.isfinite(s_A_raw):
            if max_s_A > 1e-9:
                norm_s_A = s_A_raw / max_s_A
            elif np.isclose(s_A_raw, 0.0):
                norm_s_A = 0.0

        norm_s_B = np.nan
        if np.isfinite(s_B_raw):
            if max_s_B > 1e-9:
                norm_s_B = s_B_raw / max_s_B
            elif np.isclose(s_B_raw, 0.0):
                norm_s_B = 0.0

        # Special handling for Inf results from raw entropy
        if not np.isfinite(s_total_raw): norm_s_total = s_total_raw  # Propagate Inf/-Inf
        if not np.isfinite(s_A_raw): norm_s_A = s_A_raw
        if not np.isfinite(s_B_raw): norm_s_B = s_B_raw

        return norm_s_total, norm_s_A, norm_s_B

    def _get_renyi(self):
        #circuit = self._get_cirq(maybe_add_noise=True)
        #state_v = self.simulator.simulate(circuit).final_state_vector
        #rho = cirq.density_matrix_from_state_vector(state_v)
        #rho = (rho + rho.conj().T) / 2_______pure
        # 1. 构建并获取包含噪声的电路
        try:
            circuit_with_noise = self._get_cirq(maybe_add_noise=True)
            if not isinstance(circuit_with_noise, cirq.Circuit):
                print("Warning: _get_cirq did not return a cirq.Circuit object.")
                return 0.0  # 或者抛出错误
        except Exception as e:
            print(f"Error getting circuit with noise: {e}")
            return 0.0  # 返回 0 保真度表示失败

        # 检查目标状态向量是否有效
        if self.target is None or not isinstance(self.target, np.ndarray):
            print("Error: Target state vector (self.target) is not defined or not a NumPy array.")
            return 0.0

        # 2. 使用密度矩阵模拟器模拟含噪声电路
        try:
            simulator = cirq.DensityMatrixSimulator(seed=42)  # 可以考虑是否需要固定种子
            result = simulator.simulate(circuit_with_noise, qubit_order=self.qubits)  # 推荐指定 qubit_order
            density_matrix = result.final_density_matrix

            # 强制厄米性以提高数值稳定性
            density_matrix = (density_matrix + density_matrix.conj().T) / 2
            density_matrix = density_matrix / np.trace(density_matrix)
        except Exception as e:
            print(f"Error during density matrix simulation: {e}")
            print(f"Circuit that caused error:\n{circuit_with_noise}")
            return 0.0 # 返回 0 保真度表示失败

        s_tot_norm, s_a_norm, s_b_norm = self.calculate_system_and_subsystem_renyi(density_matrix, len(self.qubits), 2)
        return s_tot_norm, s_a_norm, s_b_norm

    def _get_renyi_target(self):
        rho_target = cirq.density_matrix_from_state_vector(self.target)
        rho_target = (rho_target + rho_target.conj().T) / 2
        s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar = self.calculate_system_and_subsystem_renyi(rho_target, len(self.qubits), 2)
        return s_tot_norm_tar, s_a_norm_tar, s_b_norm_tar