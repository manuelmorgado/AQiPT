#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Quantum Circuits compiler

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s):
# Created: 2024-04-01
# Last update: 2024-12-14

try:
    import cupy as cp
except:
    import numpy as np
import matplotlib.pyplot as plt
import re
from typing import List

class QuantumCircuitMPS:
    '''

        Quantum circuit simulator using the MPS method, requires cupy (i.e., nvidia cards), no tested in AMD GPUs yet.

        #example
        qc = QuantumCircuitMPS(num_qubits=16, backend='gpu')
        qc.y(0)
        qc.z(5)
        qc.h(8)
        # qc.cnot(1,2)

        #get the final state vector of the circuit
        final_state = qc.get_state_vector()

        #print the final state
        print("Final state vector:", final_state)
    '''

    def __init__(self, num_qubits, backend='cpu', use_gpu=False):
        self.num_qubits = num_qubits
        self.backend = backend
        self.use_gpu = use_gpu

        #initialize MPS tensors for each qubit
        if self.use_gpu:
            self.mps = [cp.zeros((2, 1, 1)) for _ in range(num_qubits)]
        else:
            self.mps = [np.zeros((2, 1, 1)) for _ in range(num_qubits)]

        #set each qubit to the |0> state
        for i in range(num_qubits):
            if self.use_gpu:
                self.mps[i][0, 0, 0] = 1.0
            else:
                self.mps[i][0, 0, 0] = 1.0


    #operations
    def apply_single_qubit_gate(self, gate, qubit):

        #sets the correct backend (cupy or numpy)
        if self.use_gpu and isinstance(gate, np.ndarray):
            gate = cp.array(gate)

        elif not self.use_gpu and isinstance(gate, cp.ndarray):
            gate = cp.asnumpy(gate)

        current_tensor = self.mps[qubit]
        print(f'\nCurrent tensor shape before gate: {current_tensor.shape}')
        print(f'Gate shape: {gate.shape}')

        #perform tensorproduct and reshape
        new_tensor = cp.tensordot(gate, current_tensor, axes=([1], [0])) if self.use_gpu else np.tensordot(gate, current_tensor, axes=([1], [0]))

        print(f'New tensor shape after tensordot: {new_tensor.shape}')

        #transpose to match expected dimensions
        new_tensor = new_tensor.transpose(1, 0, 2)
        print(f'New tensor shape after transpose: {new_tensor.shape}')

        #tensor reshaped to (2, 1, 1) after gate application
        new_tensor = cp.reshape(new_tensor, (2, 1, 1)) if self.use_gpu else np.reshape(new_tensor, (2, 1, 1))
        print(f'Final tensor shape after reshape: {new_tensor.shape}')

        self.mps[qubit] = new_tensor

    def apply_two_qubit_gate(self, gate, qubit1, qubit2):

        #obtain the current states of the qubits
        tensor1 = self.mps[qubit1]  #state of the control qubit
        tensor2 = self.mps[qubit2]  #state of the target qubit

        #check dimensions of tensors before applying the gate
        if tensor1.shape != (2, 1, 1) or tensor2.shape != (2, 1, 1):
            raise ValueError(f"Expected shapes (2, 1, 1) but got {tensor1.shape} and {tensor2.shape}")

        #reshape tensors to prepare for applying the CNOT gate
        tensor1 = tensor1.reshape(2, 1, 1)  #ensure it has the correct shape
        tensor2 = tensor2.reshape(2, 1, 1)  #ensure it has the correct shape

        #merge the two tensors
        merged = np.tensordot(tensor1, tensor2, axes=([1], [1]))  #merge the middle dimensions
        merged = merged.reshape(2, 2, 1)  #reshape to (2, 2, 1)

        #apply the gate (CNOT)
        merged = np.tensordot(merged, gate, axes=([1, 2], [0, 1]))  #apply CNOT gate
        merged = merged.reshape(2, 2, 1)  #reshape to (2, 2, 1)

        #perform SVD on the merged tensor
        U, S, Vh = np.linalg.svd(merged.reshape(-1, 2), full_matrices=False)
        S = np.diag(S)  #convert singular values to a diagonal matrix

        #reshape U and Vh to their respective dimensions
        U = U.reshape((2, -1, 1))  #reshape U to match the required dimensions
        Vh = Vh.reshape((-1, 2, 1))  #reshape Vh to match the required dimensions

        #update the state vector for both qubits after SVD
        self.state_vector[qubit1] = U
        self.state_vector[qubit2] = Vh

        print(f"New state vector for qubit {qubit1}: {self.mps[qubit1]}")
        print(f"New state vector for qubit {qubit2}: {self.mps[qubit2]}")

    #gates
    def x(self, qubit):
        pauli_x_gate = cp.array([[0, 1], [1, 0]]) if self.use_gpu else np.array([[0, 1], [1, 0]])
        self.apply_single_qubit_gate(pauli_x_gate, qubit)

    def y(self, qubit):
        pauli_y_gate = cp.array([[0, -1j], [1j, 0]]) if self.use_gpu else np.array([[0, -1j], [1j, 0]])
        self.apply_single_qubit_gate(pauli_y_gate, qubit)

    def z(self, qubit):
        pauli_z_gate = cp.array([[1, 0], [0, -1]]) if self.use_gpu else np.array([[1, 0], [0, -1]])
        self.apply_single_qubit_gate(pauli_z_gate, qubit)

    def h(self, qubit):
        hadamard_gate = (1 / np.sqrt(2)) * cp.array([[1, 1], [1, -1]]) if self.use_gpu else (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.apply_single_qubit_gate(hadamard_gate, qubit)

    def phase(self, qubit: int):
        S = np.array([[1, 0], [0, 1j]], dtype=np.complex64)
        self.apply_single_qubit_gate(S, qubit)

    def t(self, qubit: int):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex64)
        self.apply_single_qubit_gate(T, qubit)

    def rx(self, qubit: int, theta: float):
        Rx = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.complex64)
        self.apply_single_qubit_gate(Rx, qubit)

    def ry(self, qubit: int, theta: float):
        Ry = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=np.float32)
        self.apply_single_qubit_gate(Ry, qubit)

    def rz(self, qubit: int, theta: float):
        Rz = np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex64)
        self.apply_single_qubit_gate(Rz, qubit)

    def cnot(self, control: int, target: int):
        CNOT = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 0]], dtype=np.float32)
        self.apply_two_qubit_gate(CNOT, control, target)

    #readout
    def get_state_vector(self):
        '''
            Reconstruct the full state vector from the MPS and return the state vector as a numpy array.
        '''

        #initialize the MPS tensor of the first qubit
        state = self.mps[0].squeeze()  #remove dimensions

        for i in range(1, self.num_qubits):
          next_tensor = self.mps[i].squeeze()

          #tensor contraction (equivalent to a matrix multiplication along the proper axes)
          state = cp.tensordot(state, next_tensor, 0) if self.use_gpu else np.tensordot(state, next_tensor, 0)

        #reshape to a 1D state vector of size 2^n
        state_vector_size = 2**self.num_qubits
        state = cp.reshape(state, (state_vector_size,)) if self.use_gpu else np.reshape(state, (state_vector_size,))

        #convert back to NumPy array if using cupy
        return cp.asnumpy(state) if self.use_gpu else state


class OpenQASMParser:
    '''
        OpenQASM parser for quantum circuits
    '''

    def __init__(self, circuit: QuantumCircuitMPS):
        '''
            Initialize the parser with a quantum circuit.

            :param circuit: QuantumCircuitMPS object to which parsed commands will be applied.
        '''
        self.circuit = circuit
        self.qubit_mapping = {}

    def parse_qasm(self, qasm_code: str):
        '''
            Parse OpenQASM code and map the instructions to QuantumCircuitMPS.

            :param qasm_code: A string containing the OpenQASM code.
        '''
        # Strip comments and split into lines
        lines = qasm_code.strip().splitlines()
        lines = [re.sub(r'//.*', '', line).strip() for line in lines]
        lines = [line for line in lines if line]  # Remove empty lines

        for line in lines:
            if line.startswith('qreg'):
                self._parse_qreg(line)
            elif line.startswith('creg'):
                pass  # We'll ignore classical registers for now
            elif line.startswith('measure'):
                self._parse_measure(line)
            elif line.startswith('cx'):
                self._parse_cnot(line)
            else:
                self._parse_single_qubit_gate(line)

    def _parse_qreg(self, line: str):
        '''
            Parse a qubit register declaration.

            line: OpenQASM code line for qubit register (e.g., 'qreg q[5];').
        '''
        match = re.match(r'qreg\s+(\w+)\[(\d+)\];', line)
        if match:
            reg_name, num_qubits = match.groups()
            self.qubit_mapping[reg_name] = list(range(int(num_qubits)))
        else:
            raise ValueError(f"Invalid qreg declaration: {line}")

    def _parse_single_qubit_gate(self, line: str):
        '''
            Parse single-qubit gates like 'h', 'x', 'rz', etc.

            line: OpenQASM code line for a gate (e.g., 'h q[0];').
        '''
        match = re.match(r'(\w+)\s+(\w+)\[(\d+)\];', line)
        
        if match:

            gate, reg_name, qubit_idx = match.groups()
            qubit = self.qubit_mapping[reg_name][int(qubit_idx)]

            # Map OpenQASM gates to QuantumCircuitMPS gates
            if gate == 'h':
                self.circuit.h(qubit)

            elif gate == 'x':
                self.circuit.x(qubit)

            elif gate == 'y':
                self.circuit.y(qubit)

            elif gate == 'z':
                self.circuit.z(qubit)

            elif gate == 's':
                self.circuit.p(qubit)

            elif gate == 't':
                self.circuit.t(qubit)

            elif gate.startswith('r'):
                self._parse_rotation(gate, qubit)

            else:
                raise ValueError(f"Unsupported gate: {gate}")

    def _parse_rotation(self, gate: str, qubit: int):
        '''
            Parse rotation gates like 'rx', 'ry', 'rz'.

            gate: The rotation gate command (e.g., 'rx').
            qubit: The qubit index.
        '''
        match = re.match(r'(\w+)\(([\d\.]+)\)\s+', gate)

        if match:

            gate_type, angle = match.groups()
            angle = float(angle)

            if gate_type == 'rx':
                self.circuit.rx(qubit, angle)

            elif gate_type == 'ry':
                self.circuit.ry(qubit, angle)

            elif gate_type == 'rz':
                self.circuit.rz(qubit, angle)

            else:
                raise ValueError(f"Unsupported rotation gate: {gate_type}")

    def _parse_cnot(self, line: str):
        '''
            Parse CNOT (cx) gate in OpenQASM.

            line: OpenQASM code line for CNOT (e.g., 'cx q[0], q[1];').
        '''
        match = re.match(r'cx\s+(\w+)\[(\d+)\],\s*(\w+)\[(\d+)\];', line)

        if match:

            reg1, q1_idx, reg2, q2_idx = match.groups()

            qubit1 = self.qubit_mapping[reg1][int(q1_idx)]
            qubit2 = self.qubit_mapping[reg2][int(q2_idx)]

            self.circuit.cnot(qubit1, qubit2)

        else:
            raise ValueError(f"Invalid CNOT syntax: {line}")

    def _parse_measure(self, line: str):
        '''
            Parse measurement operations (e.g., 'measure q[0] -> c[0];').

            line: OpenQASM code line for measurement.
        '''
        match = re.match(r'measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\];', line)

        if match:

            reg_name, q_idx, _, _ = match.groups()  #ignore classical register part
            qubit = self.qubit_mapping[reg_name][int(q_idx)]
            outcome, _ = self.circuit.measure_qubit(qubit)

            print(f"Measured qubit {qubit}: {outcome}")

        else:
            raise ValueError(f"Invalid measurement syntax: {line}")


class QuantumCompiler:
    '''
        Circuit compiler with hardware awareness on the operations and instructions possible with SLMs and AODs.
    '''

    def __init__(self, r, t_swap, t_move, max_single_qubit_gates, max_multi_qubit_gates, movement_type="2D"):
        '''
            Initializes the compiler with system parameters.

            Attributes:

                r: The dimension of the qubit grid (r x r grid of qubits).
                t_swap: Time for a SWAP gate.
                t_move: Time for moving a qubit one site.
                max_single_qubit_gates: Max number of simultaneous single-qubit gates.
                max_multi_qubit_gates: Max number of simultaneous multi-qubit gates.
                movement_type: "1D" for row-wise movements, "2D" for arbitrary 2D movements.
            
            Methods:

                compile: compile the circuit given
                are_qubits_neighbors: check if they are neighbours (True) or not (False)
                optimize_movement_and_swap: optimize the moves

                Example usage of the compiler
                print('Random circuit: \n', random_circuit)
                compiler = QuantumCompiler(r=5, t_swap=2, t_move=1, max_single_qubit_gates=4, max_multi_qubit_gates=2)
                compiled_circuit = compiler.compile(random_circuit)
                print('Compiled circuit: \n', compiled_circuit)
        '''
        self.r = r
        self.t_swap = t_swap
        self.t_move = t_move
        self.max_single_qubit_gates = max_single_qubit_gates
        self.max_multi_qubit_gates = max_multi_qubit_gates
        self.movement_type = movement_type

    def compile(self, circuit):
        '''
            Compiles the quantum circuit by optimizing qubit movements and SWAP gates.

            INPUTS:
            ------
                circuit: The quantum circuit dictionary generated by the previous script.

            OUTPUT:
            ------
                compiled_circuit: An optimized version of the circuit with SWAP and movement details.
        '''
        compiled_circuit = {"initial_positions": circuit["qubits"],  #start with the initial positions
                            "operations": [],  #list to store the SWAPs, movements, and gates
                            "total_time": 0  #keep track of total time
                            }

        #[rocess each gate in the circuit
        for gate in circuit["gates"]:
            if gate["type"] == "single":
                #apply the single qubit gate directly
                compiled_circuit["operations"].append({"type": "gate",
                                                       "gate": gate["gate"],
                                                       "qubits": gate["qubits"]
                                                       })
                compiled_circuit["total_time"] += 1  #assume unit time for single qubit gates

            elif gate["type"] == "multi":
                #check if the qubits are within interaction range
                qubits = gate["qubits"]
                if self.are_qubits_neighbors(qubits):
                    #apply the multi-qubit gate directly
                    compiled_circuit["operations"].append({"type": "gate",
                                                           "gate": gate["gate"], 
                                                           "qubits": qubits
                                                           })
                    compiled_circuit["total_time"] += 1  #unit time for multi-qubit gates
                else:
                    #in case it needs to either move or SWAP qubits into place
                    compiled_circuit = self.optimize_movement_and_swap(compiled_circuit, qubits, gate)

        return compiled_circuit

    def are_qubits_neighbors(self, qubits):
        '''
            Checks if the qubits are within interaction range and return True if they are neighbors
        '''

        #calculate distance and check against neighborhood
        (q1_x, q1_y), (q2_x, q2_y) = qubits
        distance = abs(q1_x - q2_x) + abs(q1_y - q2_y)  #distance

        return distance == 1 

    def optimize_movement_and_swap(self, compiled_circuit, qubits, gate):
        '''
            Decides whether to move or SWAP qubits to optimize gate application.

            INPUTS:
            ------
                compiled_circuit: The compiled circuit data being built.
                qubits: The qubits involved in the gate.
                gate: The gate to be applied.

            OUTPUT:
            ------
                compiled_circuit: Updated circuit with optimized operations.

            #example: Apply SWAP if qubits are far apart, else move them
        '''

        
        # Include timing calculations for each choice
        q1, q2 = qubits

        #placeholder logic
        compiled_circuit["operations"].append({ "type": "SWAP", "qubits": [q1, q2] })
        compiled_circuit["total_time"] += self.t_swap  #add SWAP time

        #apply the gate after movement/SWAP
        compiled_circuit["operations"].append({"type": "gate",  "gate": gate["gate"], "qubits": qubits })
        compiled_circuit["total_time"] += 1  #unit time for gate application

        return compiled_circuit


