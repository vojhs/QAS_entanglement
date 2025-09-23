# QAS-entanglement
# Copyright (C) 2024 [vojhs]
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# This file contains modifications based on the project "quantum-arch-search",
# which is also licensed under GPLv3. The original project can be found at:
# https://github.comcom/qdevpsi3/quantum-arch-search

from typing import Dict, List, Optional, Union

import cirq
import numpy as np
import itertools
n = 3  # Example value for n
k1 = 1
k2 = 2

def get_default_gates2(qubits: List[cirq.LineQubit]) -> (List, int):
    gates = []
    num_fixed_actions = None
    for idx, qubit in enumerate(qubits):
        next_qubit = qubits[(idx + 1) % len(qubits)]
        fixed_gates = [
            cirq.rz(np.pi / 4.)(qubit),
            #cirq.ry(np.pi / -2.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            cirq.CNOT(qubit, next_qubit),
            cirq.CZ(qubit, next_qubit),
            #cirq.ry(np.pi / 2.)(qubit),
            #FGate(n, k1)(qubit, next_qubit),
            #FGate(n, k2)(qubit, next_qubit)
        ]
        if num_fixed_actions is None:
            num_fixed_actions = len(fixed_gates)
        gates += fixed_gates
        # 添加参数化旋转门：返回一个函数用于后续根据连续参数构造门操作
        def param_ry(angle, qubit=qubit):
            return cirq.ry(angle)(qubit)
        gates.append(param_ry)
    return gates, num_fixed_actions


def get_default_gates(
        qubits: List[cirq.LineQubit]) -> List[cirq.Operation]: # 返回 Operation

    gates: List[cirq.Operation] = []
    num_qubits = len(qubits)

    if num_qubits == 0:
        return []

    # --- 定义 FGate 参数 ---
    n = num_qubits # 比特数
    k1 = 1
    k2 = 2

    # --- 1. 添加单比特门 ---
    # 使用原始循环结构添加
    for qubit in qubits: # 遍历 qubit
        gates += [
            cirq.rz(np.pi / 4.)(qubit),
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
            cirq.H(qubit),
            cirq.ry(np.pi / 2.)(qubit),
            # --- 添加其他单比特门  ---
            #cirq.T(qubit),
            cirq.S(qubit),
            #cirq.rz(np.pi/8.)(qubit),
            #cirq.rz(np.pi/16.)(qubit),

        ]

    # --- 2. 添加双比特门  ---
    if num_qubits >= 2:
        # 定义双比特门类型
        two_qubit_gate_generators = [
            lambda q1, q2: cirq.CNOT(q1, q2),
            lambda q1, q2: cirq.CZ(q1, q2),
            #lambda q1, q2: FGate(n, k1)(q1, q2),
            #lambda q1, q2: FGate(n, k2)(q1, q2),
            #lambda q1, q2: cirq.SWAP(q1, q2),
            # lambda q1, q2: CXRVGate()(q1, q2), # CXRVGate
            # lambda q1, q2: cirq.ControlledGate(cirq.H)(q1, q2),
            # lambda q1, q2: cirq.ControlledGate(cirq.ry(np.pi / 8.))(q1, q2),

        ]

        # 遍历量子比特对 (i, j) where i != j
        for i, j in itertools.permutations(range(num_qubits), 2):
            q_i = qubits[i]
            q_j = qubits[j]
            for gate_gen in two_qubit_gate_generators:
                gates.append(gate_gen(q_i, q_j))

    # --- 3. 添加三比特门 ---
    if num_qubits >= 3:
        # 定义三比特门
        three_qubit_gate_generators = [
             #lambda q1, q2, q3: cirq.TOFFOLI(q1, q2, q3),
             # lambda q1, q2, q3: cirq.FREDKIN(q1, q2, q3),
        ]

        # 遍历所有可能的有序量子比特三元组 (i, j, k) where i, j, k distinct
        for i, j, k in itertools.permutations(range(num_qubits), 3):
            q_i = qubits[i]
            q_j = qubits[j]
            q_k = qubits[k]
            for gate_gen in three_qubit_gate_generators:
                 # 只有在三比特门列表不为空时才添加
                 if three_qubit_gate_generators:
                     gates.append(gate_gen(q_i, q_j, q_k))


    # --- 清理和信息输出 ---
    # 去重
    # print(f"Generated {len(gates)} total gate operations (fully connected).")

    return gates


def get_default_observables(
        qubits: List[cirq.LineQubit]) -> List[cirq.GateOperation]:
    observables = []
    for qubit in qubits:
        observables += [
            cirq.X(qubit),
            cirq.Y(qubit),
            cirq.Z(qubit),
        ]
    return observables


def get_bell_state() -> np.ndarray:
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


def get_ghz_state(n_qubits: int = 3) -> np.ndarray:
    target = np.zeros(2**n_qubits, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    return target


class FGate(cirq.Gate):
    """
    Cirq custom gate for the F_gate.
    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.theta = np.arccos(np.sqrt(1 / (n - k + 1)))

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        q0, q1 = qubits
        yield cirq.ry(-self.theta)(q1)
        yield cirq.CZ(q0, q1)
        yield cirq.ry(self.theta)(q1)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=(f"F*CTR(k={self.k})", f"F⊕TAR(k={self.k})")
        )

    def __str__(self):
        return f"FGate(n={self.n}, k={self.k})"


class CXRVGate(cirq.Gate):
    """
    Cirq custom gate for the cxrv gate (reverse CNOT).
    """

    def _num_qubits_(self):
        return 2

    def _decompose_(self, qubits):
        q0, q1 = qubits
        yield cirq.H(q0)
        yield cirq.H(q1)
        yield cirq.CNOT(q1, q0)
        yield cirq.H(q0)
        yield cirq.H(q1)

    def _circuit_diagram_info_(self, args):
        return cirq.CircuitDiagramInfo(
            wire_symbols=("CXRV_TAR⊕", "CXRV_CTR*")  # 指定 q0 为目标比特，q1 为控制比特
        )

    def __str__(self):
        return "CXRVGate"


