#Atomic Quantum information Processing Tool (AQIPT) - Gate compiler

# Author(s): 
# Angel Alvarez. Universidad Simon Bolivar. Quantum Information and Communication Group.
# Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2022-11-24
# Last update: 2024-11-24


#libs
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from functools import reduce
# import itertools
# import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union 

# from tqdm import tqdm

# from numba import jit
# import numba
# import os, time, dataclasses
# import json

# import AQiPTcore as aqipt
from AQiPT.modules.directory import AQiPTdirectory as dirPath
from qiskit import *
from qiskit.circuit import Gate, Qubit
from qiskit import Aer, assemble, transpile, QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.library.standard_gates.u3 import CU3Gate, U3Gate


from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.control.AQiPTcontrol import function
from AQiPT.modules.emulator.AQiPTemulator import bitstring2lst
from AQiPT.modules.kernel.AQiPTkernel import RydbergQubitSchedule, RydbergQRegisterSchedule, RydbergQuantumRegister, RydbergQubit, coupling_detuning_constructors, merge_waveforms,freq_given_phi

#####################################################################################################
############################            TRANSPILER SECTION               ############################
#####################################################################################################

#Based in the paper: Morgado, M., and S. Whitlock. "Quantum simulation and computing with Rydberg-interacting qubits." AVS Quantum Science 3.2 (2021).

#####################################################################################################
#qiskit transpiler classes
#####################################################################################################



class _RydbergUxy(Gate):
    '''
        Class for the unitary transformation Uxy(theta, phi) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, label: Optional[str] = None,):
    
        super().__init__('uxy', 1, [theta, phi], label=label);
    
    def _define(self):
        '''
            Set the unitary transformation of the gate using Qiskit's U3Gate()
            where the 3 angles correspond to the Euler angles. 

            TODO: revise new version of Qiskit to

                    >> circuit.u(theta, phi, lambda)

                    https://qiskit.org/documentation/stubs/qiskit.circuit.library.U3Gate.html
        '''
        qc = RydbergQuantumCircuit(1, name=self.name);
        
        t = self.params[0]; #theta
        p = -(np.pi/2 + self.params[1]); #phi
        l = +p; #lambda
        
        qc.append(U3Gate(t, p, l), [0], []);
        self.definition = qc;
        
    
    def _matrix(self, dtype=complex) -> np.array:
        '''
            Matrix definition of the transformation as an array of numpy.
        '''
        theta, phi = self.params;
        theta, phi = float(theta), float(phi);
        
        return np.array( [[  np.cos(theta/2),                       -1j*np.sin(theta/2)*np.exp(1j*phi)], 
                         [ -1j*np.sin(theta/2)*np.exp(1j*(-phi)),   np.cos(theta/2) ]], dtype=dtype)
    
class _RydbergH(Gate):
    '''
        Class for the unitary Hadamard transformation with atomic qubit.
        Sub-class of Qiskit's Gate class. 
    '''
    def __init__(self, label: Optional[str] = None):
        super().__init__('h', 1, [], label=label);
        
    def _define(self):
        '''
            Set the unitary transformation of the gate using Uxy().
        '''
        qc = RydbergQuantumCircuit(1, name=self.name);
        qc.append(RydbergUxyGate(np.pi/2, -np.pi/2), [0]);
        qc.append(RydbergUxyGate(np.pi, 0), [0]);
        self.definition = qc;

class _RydbergRx(Gate):
    '''
        Class for the unitary transformation Rx(theta) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__(self, theta: ParameterValueType, label: Optional[str]=None):
        super().__init__('rx', 1, [theta], label=label);
        
    def _define(self):
        '''
            Set the unitary transformation of the gate using Uxy().
        '''
        qc = RydbergQuantumCircuit(1, name =self.name);
        qc.append(RydbergUxyGate(self.params[0], 0), [0]);
        self.definition = qc;
            
class _RydbergRy(Gate):
    '''
        Class for the unitary transformation Ry(theta) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__( self, theta: ParameterValueType, label: Optional[str]=None ):
        super().__init__('ry', 1, [theta], label=label)
    
    def _define(self):
        '''
            Set the unitary transformation of the gate using Uxy().
        '''
        qc = RydbergQuantumCircuit(1, name =self.name);
        qc.append(RydbergUxyGate(self.params[0] ,-np.pi/2), [0]);
        self.definition = qc;
            
class _RydbergRz(Gate):
    '''
        Class for the unitary transformation Rx(theta) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__( self, theta: ParameterValueType, label: Optional[str]=None):
        super().__init__('rz', 1, [theta], label=label)
    
    def _define(self):
        '''
            Set the unitary transformation of the gate using Uxy().
        '''
        qc = RydbergQuantumCircuit(1, name =self.name);
        qc.append(RydbergUxyGate(np.pi/2, -np.pi/2), [0]);
        qc.append(RydbergUxyGate(self.params[0], 0), [0]);
        qc.append(RydbergUxyGate(np.pi/2, np.pi/2), [0]);
        self.definition = qc;

class _RydbergCUxy(Gate):
    '''
        Class for the unitary transformation Control-Uxy(theta, phi) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__( self, theta: ParameterValueType, phi: ParameterValueType, label: Optional[str]=None):
    
        super().__init__('cuxy',2, [theta, phi], label=label);
    
    def _define(self):
        '''
            Set the unitary transformation of the gate using Uxy().
        '''
        qc = RydbergQuantumCircuit(2, name=self.name);
        
        t = self.params[0];
        p = -(np.pi/2 + self.params[1]);
        l = -p;
        
        qc.append(CU3Gate(t, p, l), [0, 1], []);
        self.definition = qc;
    
    def __array__(self, dtype = complex):
        '''
            Transformation matrix 
        '''
        theta, phi = self.params;
        theta, phi = float(theta), float(phi);
        
        cos = np.cos(theta/2);
        sin = np.sin(theta/2);
        epp = np.exp(1j*phi);
        epm = np.exp(1j*(-phi));
        
        return np.array([ [ cos,  -1j*sin*epp , 0, 0], 
                          [ -1j*sin*epm, cos , 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1] ], dtype=dtype );

class _RydbergCPHASE(Gate):
    '''
        Class for the unitary transformation CPhase(phi00, phi01, phi10, phi11) with atomic qubit.
        Sub-class of Qiskit's Gate class.

    '''
    def __init__(self,
                 phi11: ParameterValueType,
                 phi00: ParameterValueType=0,
                 phi01: ParameterValueType=np.pi,
                 phi10: ParameterValueType=np.pi,
                 label: Optional[str]=None,):
    
        super().__init__('cp',2, [phi00, phi01, phi10, phi11], label=label);
    
    def _define(self):
        '''
            Definición circuital de la transformación a base de un operador matricial.
        '''
        phi00, phi01, phi10, phi11 = self.params;
        e00 = np.exp(1j*phi00);
        e01 = np.exp(1j*phi01);
        e10 = np.exp(1j*phi10);
        e11 = np.exp(1j*phi11);
        
        qc = RydbergQuantumCircuit(2, name=self.name);
        
        cp = Operator(np.array( [ [e00, 0, 0, 0],
                                  [0, e01, 0, 0],
                                  [0, 0, e10, 0],
                                  [0, 0, 0, e11], ]) );
        qc.unitary(cp, [0,1]);
        self.definition = qc;
    
    def __array__(self, dtype = complex):
        '''
            Transformation matrix
        '''
        phi00, phi01, phi10, phi11 = self.params;
        
        e00 = np.exp(1j*phi00);
        e01 = np.exp(1j*phi01);
        e10 = np.exp(1j*phi10);
        e11 = np.exp(1j*phi11);
        
        return np.array([ [ e00, 0, 0, 0], 
                          [ 0, e01, 0, 0],
                          [ 0, 0, e10, 0],
                          [ 0, 0, 0, e11] ], dtype=dtype );
    
class _RydbergCX(Gate):
    '''
        Class for the unitary transformation  Control-X with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__(self, label: Optional[str] = None,):
        super().__init__('cx',2, [], label=label);
    
    def _define(self):
        '''
            Set the unitary transformation for quantum circuit.
        '''
        phi = self.params;
        ep = np.exp(1j*phi);
        
        qc = RydbergQuantumCircuit(2, name=self.name);
        
        cp = Operator(np.array([ [1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, ep],  ]) );
        qc.unitary(cp, [0,1]);
        self.definition = qc;
        
class _RydbergCZ(Gate):
    '''
        Class for the unitary transformation Control-Phase(phi) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__( self, phi11: ParameterValueType, label: Optional[str] = None, ):
        super().__init__("cp", 2, [phi11], label=label);

    def _define(self):
        '''
            Set the unitary transformation for quantum circuit.
        '''
        phi11 = self.params;
        e00 = np.exp(1j * 0);
        e01 = np.exp(1j * np.pi);
        e10 = np.exp(1j * np.pi);
        e11 = np.exp(1j * phi11);

        qc = RydbergQuantumCircuit(2, name=self.name);

        cp = Operator( np.array( [ [e00, 0, 0, 0],
                                   [0, e01, 0, 0],
                                   [0, 0, e10, 0],
                                   [0, 0, 0, e11], ])  );
        qc.unitary(cp, [0, 1]);
        self.definition = qc;

    def __array__(self, dtype=complex):
        '''
            Transformation matrix
        '''
        phi11 = self.params;

        e00 = np.exp(1j * 0)
        e01 = np.exp(1j * np.pi);
        e10 = np.exp(1j * np.pi);
        e11 = np.exp(1j * phi11);

        return np.array([  [e00, 0, 0, 0],
                           [0, e01, 0, 0],
                           [0, 0, e10, 0],
                           [0, 0, 0, e11],], dtype=dtype, );
        
class _RydbergSWAP(Gate):
    '''
        Class for the unitary transformation Swap with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''
    def __init__(self, label: Optional[str]=None,):
        super().__init__('swap',2, [], label=label);
    
    def _define(self):
        '''
            Set the unitary transformation for quantum circuit.
        '''
        qc = RydbergQuantumCircuit(2, name=self.name);
        
        cp = Operator(np.array([ [1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],  ]) );
        qc.unitary(cp, [0,1]);
        self.definition = qc;
        
        self.define = qc

class _RydbergXY(Gate):
    '''
        Class for the unitary transformation XY(theta, phi) with atomic qubit.
        Sub-class of Qiskit's Gate class.
    '''

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, ):
        super().__init__("xy", 2, [theta], label=label);

    def _define(self):
        '''
            Set the unitary transformation for quantum circuit.
        '''
        theta = self.params

        qc = RydbergQuantumCircuit(2, name=self.name)

        xy = Operator(
            np.array([ [1, 0, 0, 0],
                       [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                       [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                       [0, 0, 0, 1], ] ) );
        qc.unitary(xy, [0, 1]);
        self.definition = qc;

    def __array__(self, dtype=complex):
        '''
           Transformation matrix
        '''
        theta = self.params;

        return np.array([ [1, 0, 0, 0],
                          [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
                          [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
                          [0, 0, 0, 1], ], dtype=dtype, );


class RydbergQuantumCircuit(QuantumCircuit):
    '''
        AQiPT class that contains native gates based in Rydberg and cold atoms experiment. Subclass
        of QuantumCircuit from Qiskit.
        

        Parameters
        ----------
        (# Quantum bits , # Classical bits) : int tuple
        Number of quantum bits and classsical bits to generate the quantum circuit.

        Methods
        -------
        uxy : QuantumCircuit (Gate)
        cuxy : QuantumCircuit (Gate)
        cphase : QuantumCircuit (Gate)
        ryd_h : QuantumCircuit (Gate)
        ryd_rx : QuantumCircuit (Gate)
        ryd_ry : QuantumCircuit (Gate)
        ryd_rz : QuantumCircuit (Gate)
        ryd_cx : QuantumCircuit (Gate)
        ryd_cz : QuantumCircuit (Gate)
        ryd_swap : QuantumCircuit (Gate)


        Returns
        -------
        QuantumCircuit : quantum circuit based on native gates
    
    '''

    def uxy(self, theta: float, phi: float, qubit: Union[int, List[int]] ):
        '''
            Implements Uxy(theta, phi) over 'qubit'.

            Args:
                theta (float): theta rotation angle
                phi (float): phi rotation angle
                qubit (Qubit [int]): qubit index

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Uxy(theta, phi) transformation over 'qubit'

        '''
        return self.append(_RydbergUxy(theta, phi), [qubit])
    
    def cuxy(self, theta: float, phi: float, control_qubit: int, target_qubit: Union[int, List[int]]):
        '''
            Implements Controled-Uxy(theta, phi) over target qubit accordingly to the state of control qubit.

            Args:
                theta (float):theta rotation angle
                phi (float): phi rotation angle
                control_qubit (Qubit [int]): control qubit
                target_qubit (Qubit [int]): target qubit

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented CUxy(theta, phi) over target qubit conditional
                to control qubit

        '''
        return self.append(_RydbergCUxy(theta, phi), [control_qubit, target_qubit])
    
    def cphase(self, phi00: float, phi01: float, phi10: float, phi11:float, qubit1:Union[int, List[int]], qubit2: Union[int, List[int]]):
        '''
            Implements Controled-Uxy(theta, phi) over target qubit accordingly to the state of control qubit.

            Args:
                phi00 (float): phi00 rotation angle
                phi01 (float): phi01 rotation angle
                phi10 (float): phi10 rotation angle
                phi11 (float): phi11 rotation angle
                qubit1 (Qubit [int]): Qubit 1
                qubit2 (Qubit [int]): Qubit 2

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented CPhase(phi00, phi01, phi10, phi11)
                between 'qubit1' and 'qubit2'.

        '''
        return self.append(_RydbergCPHASE(phi00, phi01, phi10, phi11), [qubit1, qubit2])
    
    def cz(self, qubit1: Union[int, List[int]], qubit2: Union[int, List[int]]
    ):
        """
        Método que aplica la transformación CPhase(phi11) entre
        'qubit1' y 'qubit2'.

        Args:
            phi11 (float): Ángulo de rotación phi11
            qubit1 (Qubit [int]): Qubit 1
            qubit2 (Qubit [int]): Qubit 2

        Returns:
            RydbergQuantumCircuit: Circuito de Rydberg con CPhase(phi11)
            aplicada sobre 'qubit1' y 'qubit2'.

        """
        return self.append(_RydbergCPHASE(np.pi), [qubit1, qubit2])

    def ryd_h(self, qubit: Union[int, List[int]]):
        '''
            Implements Hadamard over qubit 

            Args:
                qubit (Qubit [int]): Qubit 

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Hadamard over 'qubit'.

        '''
        return self.append(_RydbergH(), [qubit])
    
    def ryd_rx(self, theta: float, qubit: Union[int, List[int]]):
        '''
            Implements Rotation over x over qubit 

            Args:
                theta (float): theta rotation angle around X axis 
                qubit (Qubit [int]): Qubit 

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Rx(theta) over 'qubit'.

        '''
        return self.append(_RydbergRx(theta), [qubit])
    
    def ryd_ry(self, theta: float, qubit: Union[int, List[int]]):
        '''
            Implements Rotation over y over qubit 

            Args:
                theta (float): theta rotation angle around Y axis 
                qubit (Qubit [int]): Qubit 

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Ry(theta) over 'qubit'.

        '''
        return self.append(_RydbergRy(theta), [qubit])
    
    def ryd_rz(self, theta: float, qubit: Union[int, List[int]]):
        '''
            Implements Rotation over z over qubit 

            Args:
                theta (float): theta rotation angle around Z axis 
                qubit (Qubit [int]): Qubit 

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Rz(theta) over 'qubit'.

        '''
        return self.append(_RydbergRz(theta), [qubit])
    
    def ryd_cx(self, ctrl_qubit: int, target_qubit: Union[int, List[int]]):
        '''
            Implements Control-X over target qubit accordingly to the state of control qubit.

            Args:
                control_qubit (Qubit [int]): control qubit
                target_qubit (Qubit [int]): target qubit

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented Control-X over target qubit conditional
                to control qubit

        '''
        return self.append(_RydbergCUxy(np.pi,0), [ctrl_qubit, target_qubit]);
    
    def ryd_cp(self, phi: float, ctrl_qubit: int, target_qubit: Union[int, List[int]]):
        '''
            Implements Control-P over target qubit accordingly to the state of control qubit.

            Args:
                phi (float): phi rotation angle
                control_qubit (Qubit [int]): control qubit
                target_qubit (Qubit [int]): target qubit

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented  Control-P(phi) over target qubit conditional
                to control qubit

        '''
        return self.append(_RydbergCPHASE(phi), [ctrl_qubit, target_qubit]);
    
    def ryd_swap(self, qubit1: int, qubit2: int):
        '''
            Implements SWAP between 'qubit1' and 'qubit2'.

            Args:
                qubit1 (Qubit [int]): Qubit 1
                qubit2 (Qubit [int]): Qubit 2

            Returns:
                RydbergQuantumCircuit: returns circuit with implemented SWAP
                between 'qubit1' and 'qubit2'.
        '''
        return self.append(_RydbergXY(), [qubit1, qubit2])

    def xy(self, theta: float, qubit1: Union[int, List[int]], qubit2: Union[int, List[int]]):
        """
        Método que aplica la transformación XY(theta) entre
        'qubit1' y 'qubit2'.

        Args:
            theta (float): Ángulo de rotación theta
            qubit1 (Qubit [int]): Qubit 1
            qubit2 (Qubit [int]): Qubit 2

        Returns:
            RydbergQuantumCircuit: Circuito de Rydberg con XY(theta)
            aplicada sobre 'qubit1' y 'qubit2'.

        """
        return self.append(_RydbergXY(theta), [qubit1, qubit2])


    def qft(self,
            num_qubits: Optional[int] = None,
            approximation_degree: Optional[int] = 0,
            do_swaps: Optional[bool] = True,
            insert_barriers: Optional[bool] = True,
            name="qft",):
        """
        Método que construye la subcircuito de QFT de 'num_qubits' a base de las
        transformaciones en átomos de Rydberg

        Args:
            num_qubits (int): Número de qubits de la QFT
            approximation_degree (int): Nivel de entrelazamiento de los estados en la QFT
            do_swapps (bool): Hacer o no los swaps finales de la QFT
            inverse (bool): Hacer o no el circuito inverso de la QFT
            inser_barries (bool): Insertar las barreras entre cada etapa del circuit
            name (str): Nombre del circuito

        Returns:
            RydbergQuantumCircuit: Modelo circuital cuántico de Rydberg con la QFT de 'num_qubits'
            aplicada
        """
        num_qubits = self.num_qubits

        if num_qubits == 0:
            return

        for j in reversed(range(num_qubits)):
            self.h(j)
            num_entanglements = max(
                0, j - max(0, approximation_degree - (num_qubits - j - 1))
            )
            for k in reversed(range(j - num_entanglements, j)):
                # Use negative exponents so that the angle safely underflows to zero, rather than
                # using a temporary variable that overflows to infinity in the worst case.
                lam = np.pi * (2.0 ** (k - j))
                self.cp(lam, j, k)

            if insert_barriers:
                self.barrier()

        if do_swaps:
            for i in range(num_qubits // 2):
                self.swap(i, num_qubits - i - 1)

#LUT for gates
# Gates instance LUT (Look-UP-Table)
#NATIVE
ALG_UXY = hex(id(_RydbergUxy));
ALG_CUY = hex(id(_RydbergCUxy));
ALG_CPHASE = hex(id(_RydbergCPHASE));

# #CANONICAL
ALG_RYD_H = hex(id(_RydbergH));
ALG_RYD_RX = hex(id(_RydbergRx));
ALG_RYD_RY = hex(id(_RydbergRy));
ALG_RYD_RZ = hex(id(_RydbergRz));
ALG_RYD_CX = hex(id(_RydbergCX));
ALG_RYD_CZ = hex(id(_RydbergCZ));
ALG_RYD_SWAP = hex(id(_RydbergSWAP));


config = aqipt.transpiler_config
freq = config.normal_frequency
shape = config.shape
TIME_SLEEP = config.t_wait
t_start = config.t_start
high_freq = config.high_frequency


def transpilation_rule(func: Callable) -> Callable:
    '''(name)_rule(args):

    Transpilation rule for the (name) gate.

    Args:
        name (str): Name of the gate inside the QuantumCircuit.
        params (List[float]): List of parameters, normally angles, that some gates need.
        num_qubits (int): Number of qubits that the gate is applied
        qubits (List[int]): List that contains the number of the qubit(s) which is applied on.
        circuit_schedule (dict): Dictionary than contains the schedule of the circuit so far.

    Raises:
        ValueError: If name does not match.
        ValueError: If the number of qubits does not match.

    Decorator for transpilation rules. It extracts common arguments and the backend.

    Args:
        func (_type_): _description_
    '''

    def extract_backend(*args, **kwargs):
        if "backend" in kwargs.keys():
            backend_config = kwargs["backend"];
        else:
            backend_config = aqipt.backend_config;

        assert isinstance(backend_config, aqipt.BackendConfig)

        transpiler_config = aqipt.backend_config.transpiler_config;
        t_wait = transpiler_config.t_wait;
        freq = transpiler_config.normal_frequency;
        shape = transpiler_config.shape;

        atomic_config = backend_config.atomic_config;
        c6 = atomic_config.c6_constant;
        R = atomic_config.R;

        func(t_wait=t_wait, freq=freq, shape=shape, c6=c6, R=R, *args, **kwargs)

    return extract_backend

@transpilation_rule
def uxy_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the Uxy transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "uxy"
        params (List[float]): List of parameters for the gate, theta and phi
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "uxy"
        ValueError: The number of qubits is different of 1
    '''
    t_wait = kwargs["t_wait"];
    freq = kwargs["freq"];
    shape = kwargs["shape"];

    if name != "uxy":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    # Get the gate angles and qubit
    theta = params[0];
    phi = params[1];
    qubit = qubits[0];

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)];
    qubit_t_end = max(qubit_info[1], t_wait);

    # Construct the gate schedule
    Uxy = UxySchedule(theta=theta,
                      phi=phi,
                      t_start=qubit_t_end,
                      freq=freq,
                      shape=shape,
                      backend=kwargs["backend"],)

    # Update the circuit schedule
    qubit_info[0].append(Uxy);
    qubit_info[1] = Uxy.t_end + t_wait;

@transpilation_rule
def rx_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the rx transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "rx"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "rx"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "rx":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    theta = params[0]
    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Rx = RxSchedule(
        theta=theta,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Rx)
    qubit_info[1] = Rx.t_end + t_wait

@transpilation_rule
def ry_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the ry transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "ry"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "ry"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "ry":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    theta = params[0]
    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Ry = RySchedule(
        theta=theta,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Ry)
    qubit_info[1] = Ry.t_end + t_wait

@transpilation_rule
def rz_rule( name: str,  params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the rz transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "rz"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "rz"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "rz":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    theta = params[0]
    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Rz = RzSchedule(
        theta=theta,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Rz)
    qubit_info[1] = Rz.t_end + t_wait

@transpilation_rule
def x_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the x transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "x"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "x"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "x":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Rx = RxSchedule(
        theta=np.pi,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Rx)
    qubit_info[1] = Rx.t_end + t_wait

@transpilation_rule
def y_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the y transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "y"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "y"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "y":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_schedule = qubit_info[0]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Ry = RySchedule(
        theta=np.pi,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Ry)
    qubit_info[1] = Ry.t_end + t_wait

@transpilation_rule
def z_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the z transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "z"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "z"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "z":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    theta = np.pi;
    qubit = qubits[0]

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)]
    qubit_t_end = max(qubit_info[1], t_wait)

    # Construct the gate schedule
    Rz = RzSchedule(
        theta=theta,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    # Update the circuit schedule
    qubit_info[0].append(Rz)
    qubit_info[1] = Rz.t_end + t_wait

@transpilation_rule
def h_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the h transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "h"
        params (List[float]): List of parameters for the gate, usually angles
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "h"
        ValueError: The number of qubits is different of 1
    '''

    t_wait = kwargs["t_wait"];
    freq = kwargs["freq"];
    shape = kwargs["shape"];

    if name != "h":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 1:
        raise ValueError(f"Number of qubits {num_qubits} != 1")

    qubit = qubits[0];

    # Get the qubit list of schedules and end time
    qubit_info = circuit_schedule[str(qubit)];
    qubit_t_end = max([qubit_info[1] for qubit_info in circuit_schedule.values()]);

    Uxy1 = UxySchedule(
        theta=np.pi / 2,
        phi=-np.pi / 2,
        t_start=qubit_t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    );
    Uxy2 = UxySchedule(
        theta=np.pi,
        t_start=Uxy1.t_end,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    );

    # Update the circuit schedule
    qubit_info[0].append(Uxy1);
    qubit_info[0].append(Uxy2);
    qubit_info[1] = Uxy2.t_end + t_wait;

@transpilation_rule
def cuxy_rule( name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the cuxy transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "cuxy"
        params (List[float]): List of parameters for the gate, theta and phi
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "cuxy"
        ValueError: The number of qubits is different of 2
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "cuxy":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")

    theta = params[0]
    phi = params[1]
    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    t_start = max(
        control_t_end, target_t_end, t_wait
    )  # We must wait for both qubits to be free
    CUxy = CUxySchedule(
        t_start=t_start,
        theta=theta,
        phi=phi,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    circuit_schedule[str(ctrl)][0].append(CUxy.q_schedule[0])
    circuit_schedule[str(targt)][0].append(CUxy.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = CUxy.t_end + t_wait
    circuit_schedule[str(targt)][1] = CUxy.t_end + t_wait

@transpilation_rule
def cx_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the cx transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "cx"
        params (List[float]): List of parameters for the gate
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "cx"
        ValueError: The number of qubits is different of 2
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "cx":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")
    print(params)
    theta = np.pi;
    phi = 0
    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    t_start = max(
        control_t_end, target_t_end, t_wait
    )  # We must wait for both qubits to be free
    CUxy = CUxySchedule(
        t_start=t_start,
        theta=theta,
        phi=phi,
        freq=freq,
        shape=shape,
        backend=kwargs["backend"],
    )

    circuit_schedule[str(ctrl)][0].append(CUxy.q_schedule[0])
    circuit_schedule[str(targt)][0].append(CUxy.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = CUxy.t_end + t_wait
    circuit_schedule[str(targt)][1] = CUxy.t_end + t_wait

@transpilation_rule
def cp_rule(name: str, params: List[float],  num_qubits: int, qubits: List[int],  circuit_schedule: Dict[str, List],  **kwargs,
):
    '''
        The application for the cp transpilation rule.

        Args:
            name (str): Name of the gate received. Must be equalt to "cp"
            params (List[float]): List of parameters for the gate, phi_11
            num_qubits (int): Number of qubits where the gate is applied
            qubits (List[int]): List of integer index for the qubits
            circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

        Raises:
            ValueError: The name does not match "cp"
            ValueError: The number of qubits is different of 2
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "cp":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")

    phi11 = params[0]
    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    t_start = max(
        [qubit_info[1] for qubit_info in circuit_schedule.values()]
    )  # We must wait for both qubits to be free
    CP = CphaseSchedule(
        t_start=t_start, phi11=phi11, freq=freq, shape=shape, backend=kwargs["backend"]
    )

    circuit_schedule[str(ctrl)][0].append(CP.q_schedule[0])
    circuit_schedule[str(targt)][0].append(CP.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = CP.t_end + t_wait
    circuit_schedule[str(targt)][1] = CP.t_end + t_wait

@transpilation_rule
def cz_rule(
    name: str,
    params: List[float],
    num_qubits: int,
    qubits: List[int],
    circuit_schedule: Dict[str, List],
    **kwargs,
):
    """The application for the cp transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "cp"
        params (List[float]): List of parameters for the gate, phi_11
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "cp"
        ValueError: The number of qubits is different of 2
    """

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "cz":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")

    phi11 = np.pi
    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    t_start = max(
        [qubit_info[1] for qubit_info in circuit_schedule.values()]
    )  # We must wait for both qubits to be free
    CP = _RydbergCPHASE(
        t_start=t_start, phi11=phi11, freq=freq, shape=shape, backend=kwargs["backend"]
    )

    circuit_schedule[str(ctrl)][0].append(CP.q_schedule[0])
    circuit_schedule[str(targt)][0].append(CP.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = CP.t_end + t_wait
    circuit_schedule[str(targt)][1] = CP.t_end + t_wait

@transpilation_rule
def iswap_rule( name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the iswap transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "iswap"
        params (List[float]): List of parameters for the gate, theta and phi
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "iswap"
        ValueError: The number of qubits is different of 2
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "swap" and name != "iswap":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")

    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    t_start = max(
        [qubit_info[1] for qubit_info in circuit_schedule.values()]
    )  # We must wait for both qubits to be relaxed
    XY = XYSchedule(t_start=t_start, freq=freq, shape=shape, backend=kwargs["backend"])

    circuit_schedule[str(ctrl)][0].append(XY.q_schedule[0])
    circuit_schedule[str(targt)][0].append(XY.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = XY.t_end + t_wait
    circuit_schedule[str(targt)][1] = XY.t_end + t_wait

@transpilation_rule
def xy_rule(name: str, params: List[float], num_qubits: int, qubits: List[int], circuit_schedule: Dict[str, List], **kwargs,
):
    '''The application for the xy transpilation rule.

    Args:
        name (str): Name of the gate received. Must be equalt to "xy"
        params (List[float]): List of parameters for the gate, Theta
        num_qubits (int): Number of qubits where the gate is applied
        qubits (List[int]): List of integer index for the qubits
        circuit_schedule (Dict[str, List]): Dictionary containing the prototype of the circuit schedule.

    Raises:
        ValueError: The name does not match "xy"
        ValueError: The number of qubits is different of 2
    '''

    t_wait = kwargs["t_wait"]
    freq = kwargs["freq"]
    shape = kwargs["shape"]

    if name != "xy":
        raise ValueError(f"Name {name} does not match for this rule")

    if num_qubits != 2:
        raise ValueError(f"Number of qubits {num_qubits} != 2")

    ctrl, targt = qubits[0], qubits[1]

    # Get the qubit list of schedules and end time
    control_info = circuit_schedule[str(ctrl)]
    control_t_end = control_info[1]

    # Get the qubit list of schedules and end time
    target_info = circuit_schedule[str(targt)]
    target_t_end = target_info[1]

    theta = params[0]

    t_start = max(
        control_t_end, target_t_end, t_wait
    )  # We must wait for both qubits to be relaxed
    XY = XYSchedule(
        theta=theta, t_start=t_start, freq=freq, shape=shape, backend=kwargs["backend"]
    )

    circuit_schedule[str(ctrl)][0].append(XY.q_schedule[0])
    circuit_schedule[str(targt)][0].append(XY.q_schedule[1])

    circuit_schedule[str(ctrl)][1] = XY.t_end + t_wait
    circuit_schedule[str(targt)][1] = XY.t_end + t_wait

transpilation_rules = {
    # One qubit rules
    "uxy": uxy_rule,
    "rx": rx_rule,
    "ry": ry_rule,
    "rz": rz_rule,
    "x": x_rule,
    "y": y_rule,
    "z": z_rule,
    "h": h_rule,
    # Two qubit rules
    "cuxy": cuxy_rule,
    "cx": cx_rule,
    "cz": cz_rule,
    "cp": cp_rule,
    "xy": xy_rule,
    "iswap": iswap_rule,
    "swap": iswap_rule,}

#####################################################################################################
#Shaped waveforms class
#####################################################################################################

WAVEFORM_PARAMS = aqipt.backend_config.simulation_config.WAVEFORM_PARAMS;

class ShapedFunction:
    def __init__(
        self,
        t_o: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        amp: float = 1,
        width: float = 0,
        name: Optional[str] = None,
        color: Optional[str] = None,
        area: Optional[float] = None,
        **kwargs,
    ):
        self.t_o = t_o
        self.t_start = t_start
        self.t_end = t_end
        self.width = width
        self.tg = 2.0 * self.width
        self.amp = amp
        self.name = name
        self.color = color
        self.type = None
        self.area = area
        self.args = None
        self.function = None

        if "backend" in kwargs:
            backend_config = kwargs["backend"]
            assert isinstance(backend_config, aqipt.BackendConfig)

            self.backend_config = backend_config

        else:
            self.backend_config = aqipt.backend_config

        simulation_config = self.backend_config.simulation_config
        t_max = simulation_config.time_simulation

        self.tp_window = t_max

    def info(self):
        """Imprime la información completa del pulso."""
        print(
            f"{self.type} ({self.name}) - Amp:{self.amp:0.5f}, Center: {self.t_o:0.2f}, Gate time: {self.tg:0.5f}"
        )
          
class GaussianFunction(ShapedFunction):

    def __init__(
        self,
        t_o: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        amp: int = 1,
        g_std: float = np.pi / 40,
        name: Optional[str] = None,
        color: Optional[str] = None,
        area: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(t_o, t_start, t_end, amp, 4 * g_std, name, color, area, **kwargs )
        self.type = "Gaussian Pulse"
        self.g_std = g_std

        self._set_parameters()
        self.function = self._function()

    def _set_parameters(self):
        '''
            Generates the parameters of the function.
            1) length mode: given an start and final time, the waveform area is calculated given the waveform length.
                
            2) area mode: given an area and a start time, the length is calculated based in the standard deviation.

        '''
        if self.t_start is not None and self.t_end is not None:  # Primer modo
            self.width = (self.t_end - self.t_start) / 2;
            self.t_o = self.t_start + self.width;
            self.g_std = self.width / 4;
            self.area = self.g_std * np.power(5 * np.pi, 1 / 2) * np.abs(self.amp);

        elif self.area is not None and self.t_start is not None:  # Segundo modo
            self.g_std = self.area / (np.power(5 * np.pi, 1 / 2) * np.abs(self.amp));
            self.width = self.g_std * 4;
            self.t_o = self.t_start + self.width;
            self.t_end = self.t_start + 2 * self.width;
            self.tg = self.t_end - self.t_start;

    def _function(self):
        '''
            Generates the function instance from the control module i.e., Gaussian.
        '''
        args_list = {
            "g_Amp": self.amp,
            "g_center": self.t_o,
            "g_std": self.g_std,
            "tp_window": self.tp_window,
            "name": self.name,
            "color": self.color,
            "type": self.type,
        }
        self.args = args_list;

        simulation_config = self.backend_config.simulation_config;
        sampling = simulation_config.sampling;
        t_max = simulation_config.time_simulation;

        self.tp_window = t_max;

        t_p = np.linspace(0, self.tp_window, int((self.tp_window - 0) * sampling / t_max));

        return function(t_p, args_list).gaussian()[0]
           
class SquareFunction(ShapedFunction):
    r"""Pulso de forma cuadrada"""

    def __init__(
        self,
        t_o: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        amp: float = 1,
        width: float = 0,
        name: Optional[str] = None,
        color: Optional[str] = None,
        area: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(t_o, t_start, t_end, amp, width, name, color, area, **kwargs)
        self.type = "Square Pulse"

        self._set_parameters()
        self.function = self._function()

    def _set_parameters(self):
        '''
            Generates the parameters of the function.
            1) length mode: given an start and final time, the waveform area is calculated given the waveform length.
                
            2) area mode: given an area and a start time, the length is calculated based in the standard deviation.

        '''  

        if self.t_start is not None and self.t_end is not None: #length mode
            self.width = (self.t_end - self.t_start) / 2; # The area carries a factor of 1/2 over the calculations given the form of the Hamiltonian
            self.t_o = self.t_start + self.width;
            self.tg = self.width * 2;
            self.area = self.tg * np.abs(self.amp);

        # Area and starting setting
        elif self.area is not None and self.t_start is not None: #area mode
            self.width = 1 / 2 * (self.area / (np.abs(self.amp)));
            self.t_o = self.t_start + self.width;
            self.tg = self.width * 2;
            self.t_end = self.t_start + self.tg;

    def _function(self):
        '''
            Generates the function instance from the control module i.e., Square.
        '''
        args_list = { "amp": self.amp,
                      "t_o": self.t_o,
                      "width": self.width,
                      "tp_window": self.tp_window,
                      "name": self.name,
                      "color": self.color,
                      "type": self.type,}

        self.args = args_list
        simulation_config = self.backend_config.simulation_config
        sampling = simulation_config.sampling
        t_max = simulation_config.time_simulation

        self.tp_window = t_max

        t_p = np.linspace( 0, self.tp_window, int((self.tp_window - 0) * sampling / t_max) )

        return function(t_p, args_list).step()[0]

#Zero function
ZERO_FUNCTION = SquareFunction(t_start = 0, t_end=0, amp=0)



#####################################################################################################
# gates schedules classes
#####################################################################################################
class GateSchedule:
    def __init__(
        self, t_start: float, freq: float, pair: list, shape: str, **kwargs
    ) -> None:
        r"""Clase que contiene las bases fundamentales del schedule de una compuerta.

        Args:
            t_start (float): Tiempo de inicio del schedule.
            freq (float): Frecuencia normal para los pulsos dentro del schedule.
            pair (list): Par de estados necesarios paral el schedule.
            shape (str): Forma normal de los pulsos dentro del schedule.
        """
        self.t_start = t_start
        self.t_end = t_start
        self.freq = freq
        self.pair = pair
        self.shape = shape
        self.n_qubits = len(pair)
        self.omega = 2 * np.pi * freq
        self.q_schedule = None

        if "backend" in kwargs.keys():
            backend_config = kwargs["backend"]
            assert isinstance(backend_config, aqipt.BackendConfig)

            self.backend_config = backend_config

        else:
            self.backend_config = aqipt.backend_config

    def __call__(self):
        return self.q_schedule

def ZeroSchedule() -> RydbergQubitSchedule:
    """Definición del schedule de valor nulo.

    Returns:
        RydbergQubitSchedule: Retorno del schedule.
    """
    couplings = [
        ([0, 1], ZERO_FUNCTION.function),
    ]

    detunings = [([1, 1], ZERO_FUNCTION.function)]

    coupling1 = {}
    for i, coupling in enumerate(couplings):
        levels, coupling = coupling
        coupling1["Coupling" + str(i)] = [levels, 0, coupling]

    detuning1 = {}
    for i, detuning in enumerate(detunings):
        levels, detuning = detuning
        detuning1["Detuning" + str(i)] = [levels, 0, detuning]

    return RydbergQubitSchedule(coupling_waveforms=coupling1, detuning_waveforms=detuning1)

class UxySchedule(GateSchedule):
    r"""Este es el schedule para la compuerta XY en átomos de Rydberg.
    
    **Representación matricial:**

    .. math::

        U_{xy}(\theta, \varphi) = 
        \qty(\begin{array}{cc}
         \cos{(\theta/2)} & -i \sin{(\theta/2)e^{-i\varphi}  }  \\
         -i \sin{(\theta/2)e^{+i\varphi}} & \cos{(\theta/2)}, 
        \end{array})
    )

    **Operador evolución:**

    .. math::
        \hat{H}_j^{ab} = \qty(\frac{\Omega_j(t)}{2}e^{i\varphi_j(t)}\ket{a}\bra{b}+\text{h.c})  
        - \Delta_j(t)\ket{b}_j\bra{b}.
    """

    def __init__(
        self,
        theta: float = np.pi,
        phi: float = 0,
        t_start: float = 1,
        freq: float = 1,
        shape: str = "square",
        pair: Optional[list] = None,
        **kwargs,
    ) -> None:
        if pair is None:
            pair = [0, 1]
        super().__init__(t_start, freq, pair, shape, **kwargs)
        self.theta = theta
        self.phi = phi

        self._schedule()

    def _schedule(self) -> RydbergQubitSchedule:
        omega = 2 * np.pi * self.freq
        self.omega = omega
        if self.shape == "square":
            ShapedFunction = SquareFunction
        elif self.shape == "gaussian":
            ShapedFunction = GaussianFunction
        else:
            raise ValueError(f"{self.shape} is not a valid shape.")

        pulse_1 = ShapedFunction(
            t_start=self.t_start, area=self.theta / omega, backend=self.backend_config
        )

        pulse_t1 = SquareFunction(
            t_start=self.t_start, t_end=pulse_1.t_end, backend=self.backend_config
        )
        complx_pulse_1 = pulse_1.function * np.exp(-1j * self.phi * pulse_t1.function)

        self.t_end = pulse_1.t_end

        coupling = [(self.pair, complx_pulse_1)]

        detuning = [([1, 1], ZERO_FUNCTION.function)]

        coup, detun = coupling_detuning_constructors(
            coupling, detuning, omega_coup=omega
        )
        qubit_schedule1 = RydbergQubitSchedule(
            coupling_waveforms=coup, detuning_waveforms=detun, backend=self.backend_config
        )

        self.q_schedule = qubit_schedule1
        return qubit_schedule1

class RxSchedule(UxySchedule):
    r"""Este es el schedule para la compuerta Rx. 
    
    **Representación matricial:**

    .. math::
        R_{x}(\theta)_{\mathfrak{R}} = U_{x,y}(\theta, 0) = 
        \begin{pmatrix}
            \cos\qty(\frac{\theta}{2}) & -i\sin\qty(\frac{\theta}{2}) \\
            -i\sin\qty(\frac{\theta}{2}) & \cos\qty(\frac{\theta}{2})
        \end{pmatrix},
    """

    def __init__(
        self,
        theta: float = np.pi,
        t_start: float = 1,
        freq: float = 1,
        shape: str = "square",
        pair: Optional[list] = None,
        **kwargs
    ) -> None:
        if pair is None:
            pair = [0, 1]
        super().__init__(theta, 0, t_start, freq, shape, pair, **kwargs)

    def _schedule(self):
        Rx = UxySchedule(
            self.theta,
            0,
            self.t_start,
            self.freq,
            self.shape,
            self.pair,
            backend=self.backend_config,
        )
        self.t_end = Rx.t_end
        self.q_schedule = Rx.q_schedule
        return Rx

class RySchedule(UxySchedule):
    r"""Este es el schedule para la compurta Ry. 
    
    **Representación matricial:**

    .. math::
        R_{y}(\theta)_{\mathfrak{R}} = U_{x,y}(\theta, -\pi/2) = 
        \begin{pmatrix}
            \cos\qty(\frac{\theta}{2}) & \sin\qty(\frac{\theta}{2}) \\
            -\sin\qty(\frac{\theta}{2}) & \cos\qty(\frac{\theta}{2})
        \end{pmatrix}.

    """

    def __init__(
        self,
        theta: float = np.pi,
        t_start: float = 1,
        freq: float = 1,
        shape: str = "square",
        pair: Optional[list] = None,
        **kwargs
    ) -> None:
        if pair is None:
            pair = [0, 1]
        super().__init__(theta, -np.pi / 2, t_start, freq, shape, pair, **kwargs)

    def _schedule(self):
        Ry = UxySchedule(
            self.theta,
            -np.pi / 2,
            self.t_start,
            self.freq,
            self.shape,
            backend=self.backend_config,
        )
        self.t_end = Ry.t_end
        self.q_schedule = Ry.q_schedule
        return Ry

class RzSchedule(UxySchedule):
    r"""Este es el schedule para la compurta Rz. 
    
    **Representación matricial:**

    .. math::
        R_z(\theta)_\mathfrak{R} = U_{x,y}(\pi/2,\pi/2)U_{x,y}(\theta,0)U_{x,y}(\pi/2, -\pi/2) = \begin{pmatrix}
            e^{i\theta/2} & 0 \\
            0 & e^{-i\theta/2}
        \end{pmatrix} = R_z(-\theta)

    """

    def __init__(
        self,
        theta: float = np.pi,
        t_start: float = 1,
        freq: float = 1,
        shape: str = "square",
        pair: Optional[list] = None,
        **kwargs
    ) -> None:
        if pair is None:
            pair = [0, 1]
        super().__init__(theta, 0, t_start, freq, shape, pair, **kwargs)

    def _schedule(self):
        Ry = UxySchedule(
            np.pi / 2,
            -np.pi / 2,
            t_start=self.t_start,
            freq=self.freq,
            shape=self.shape,
            backend=self.backend_config,
        )
        Rx = UxySchedule(
            self.theta,
            0,
            t_start=Ry.t_end,
            freq=self.freq,
            shape=self.shape,
            backend=self.backend_config,
        )
        R = UxySchedule(
            np.pi / 2,
            +np.pi / 2,
            t_start=Rx.t_end,
            freq=self.freq,
            shape=self.shape,
            backend=self.backend_config,
        )

        R.q_schedule.add_function(
            Rx.q_schedule.coupling_waveforms["Coupling0"][2], "Coupling0"
        )
        R.q_schedule.add_function(
            Ry.q_schedule.coupling_waveforms["Coupling0"][2], "Coupling0"
        )

        self.q_schedule = R.q_schedule
        self.t_end = R.t_end
        return R

class CphaseSchedule(GateSchedule):
    r"""Este es el schedule ede la compuerta diagonal que introduce una fase en el estado del 
    qubit objetivo dependiento del estado del control, llamda CPHASE.

    **Representación matricial:**

    .. math::

        \text{CPHASE}(\Phi_{11}) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & -1 & 0 & 0 \\
                0 & 0 & -1 & 0 \\
                0 & 0 & 0 & e^{i\Phi_{11}}
            \end{pmatrix}

    **Operador evolución:**

    .. math::
        
        \begin{split}
            \hat{U} = &\exp\qty[-i\hat{H}^{r1}_c(\Omega=\Omega_1,\Delta=0)\tau_1] \\
                &\cross \exp\qty[-i\qty(\hat{H}^{r1}_t(\Omega=\Omega_2, \Delta=0) + \hat{H}^{rrrr}_{c,t})\tau_2] \\
                &\cross \exp\qty[-i\hat{H}^{r1}_c(\Omega=\Omega_1,\Delta=0)\tau_1] 
        \end{split}    

    
    """

    def __init__(
        self,
        t_start: float = 1,
        phi11: float = 0,
        freq: float = 1,
        pair: Optional[list] = None,
        shape: str = "gaussian",
        **kwargs
    ) -> None:
        if pair is None:
            pair = [[1, 3], [1, 3]]
        super().__init__(t_start, freq, pair, shape, **kwargs)

        self.phi11 = phi11

        self._schedule()

    def _schedule(self):
        t_start = self.t_start
        freq = self.freq
        shape = self.shape
        c_pair, t_pair = self.pair[0], self.pair[1]

        atomic_config = self.backend_config.atomic_config
        c_6 = atomic_config.c6_constant
        r_dist = atomic_config.R
        v_ct = c_6 / np.power(r_dist, 6)

        freq_int = freq_given_phi(self.phi11, v_ct)

        # 1 -> r
        rc1 = RxSchedule(
            theta=np.pi,
            t_start=t_start,
            freq=freq,
            shape=shape,
            pair=c_pair,
            backend=self.backend_config,
        )

        # r -> r
        rt1 = RxSchedule(
            theta=2 * np.pi,
            t_start=rc1.t_end,
            freq=freq_int,
            shape="gaussian",
            pair=t_pair,
            backend=self.backend_config,
        )
        # r -> 1
        rc2 = RxSchedule(
            theta=np.pi,
            t_start=rt1.t_end,
            freq=freq,
            shape=shape,
            pair=c_pair,
            backend=self.backend_config,
        )

        rc1.q_schedule.add_function(
            rc2.q_schedule.coupling_waveforms["Coupling0"][2], "Coupling0"
        )

        self.t_end = rc2.t_end

        self.q_schedule = (rc1, rt1)  # control and target schedules

class CUxySchedule(GateSchedule):
    r"""
    Este es el schedule de la versión controlada de CUxy. 
    
    **Representación matricial:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        CU_{x,y}(\theta, \varphi)\ q_0, q_1 =
            U_{xy}(\theta,\varphi) \otimes |0\rangle\langle 0| +
            I \otimes |1\rangle\langle 1| =
        \begin{pmatrix}
            \cos{(\th)} & -i \sin{(\th)e^{-i\varphi}} & 0 & 0 \\
            -i\sin{(\th)e^{+i\varphi}} & \cos{(\th)} & 0 & 0 \\
            0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 1 \\
        \end{pmatrix}
    
    **Operador evolución:**

    .. math::
        
        \hat{U} = \exp{-i\qty(\hat{H}^{01}_t+\hat{H}^{1111}_{c,t})\tau_g}.    

    """

    def __init__(
        self,
        theta: float = np.pi,
        phi: float = 0,
        t_start: float = 1,
        freq: float = 1.0,
        pair: Optional[list] = None,
        shape: str = "square",
        **kwargs
    ) -> None:
        if pair is None:
            pair = [[1, 3], [0, 3]]
        super().__init__(t_start, freq, pair, shape)

        self.theta = theta
        self.phi = phi

        self._schedule()

    def _schedule(self):
        r1 = RxSchedule(
            theta=np.pi,
            t_start=self.t_start,
            freq=self.freq,
            shape=self.shape,
            pair=self.pair[0],
            backend=self.backend_config,
        )

        r2 = UxySchedule(
            theta=self.theta,
            phi=self.phi,
            t_start=r1.t_end,
            freq=self.freq,
            pair=self.pair[1],
            shape=self.shape,
            backend=self.backend_config,
        )

        r3 = RxSchedule(
            theta=np.pi,
            t_start=r2.t_end,
            freq=self.freq,
            pair=self.pair[0],
            shape=self.shape,
            backend=self.backend_config,
        )

        r1.q_schedule.add_function(
            r2.q_schedule.coupling_waveforms["Coupling0"][2], "Coupling0"
        )
        r1.q_schedule.add_function(
            r3.q_schedule.coupling_waveforms["Coupling0"][2], "Coupling0"
        )
        target_schedule = r1

        control_schedule = UxySchedule(
            theta=0,
            t_start=self.t_start,
            freq=self.freq,
            pair=self.pair[0],
            shape=self.shape,
            backend=self.backend_config,
        )

        self.t_end = target_schedule.t_end
        self.q_schedule = (control_schedule, target_schedule)        

class XYSchedule(GateSchedule):
    r"""Este es el schedule para la compuerta XY en átomos de Rydberg.
    
    **Representación matricial:**

    .. math::

        \text{XY}(\Theta) =
        \begin{pmatrix}
             1 & 0 & 0 & 0 \\
             0 & \cos\qty(\Theta/2) & - i\sin\qty(\Theta/2) & 0 \\
             0 & - i\sin\qty(\Theta/2) & \cos\qty(\Theta/2) & 0 \\
             0 & 0 & 0 & 1
        \end{pmatrix}
    )

    **Operador evolución:**

    .. math::
        
        \begin{split}
            \hat{U} = & \exp\qty[-i \sum_\alpha \hat{H}_\alpha^{r0} \qty(\Omega_\alpha = \Omega_1, \varphi = \pi) \tau_1] \\
            & \times \exp\qty[-i\qty(\hat{H}_{c,t}^{rr'r'r} + \sum_\alpha \hat{H}_\alpha^{r'1} \qty(\Omega_\alpha = \Omega_2, \varphi = \pi))\tau_2] \\
            & \times \exp\qty[-i\qty(\hat{H}_{c,t}^{rr'r'r} + \sum_\alpha \hat{H}_\alpha^{r'1} \qty(\Omega_\alpha = \Omega_2, \varphi = 0))\tau_2] \\
            & \times \exp\qty[-i \sum_\alpha \hat{H}_\alpha^{r0} \qty(\Omega_\alpha = \Omega_1, \varphi = 0) \tau_1]
        \end{split}
    """

    def __init__(
        self,
        theta: float = 3 * np.pi,
        t_start: float = 1,
        freq: float = 1,
        pair: Optional[list] = None,
        shape: str = "square",
        **kwargs,
    ) -> None:
        if pair is None:
            pair = [[0, 2], [1, 3]]
        super().__init__(t_start, freq, pair, shape, **kwargs)
        self.theta = theta
        self._schedule()

    def _schedule(self):
        theta = self.theta
        t_start = self.t_start
        freq = self.freq
        shape = self.shape
        c3 = self.backend_config.atomic_config.c3_constant
        R = self.backend_config.atomic_config.R
        Vct = c3 / (np.sqrt(R) ** 3)

        omega1 = 2 * np.pi * freq

        omega2 = (2 * np.pi / theta) * (-Vct)

        if shape == "square":
            ShapedFunction = SquareFunction
        elif shape == "gaussian":
            ShapedFunction = GaussianFunction
        else:
            raise ValueError(f"{shape} is not a valid shape")

        # 1: Pulse from 0 -> r at Omega 1, Pi pulse
        p1 = ShapedFunction(
            t_start=t_start, area=np.pi / omega1, backend=self.backend_config
        )

        # 2: Pulse from 1 -> r' at Omega 2, 2Pi Pulse
        p2 = SquareFunction(
            t_start=p1.t_end, area=2 * np.pi / omega2, backend=self.backend_config
        )

        # 3: Same as before but phase -Pi
        p3 = SquareFunction(
            t_start=p2.t_end, area=2 * np.pi / omega2, backend=self.backend_config
        )
        p3_t = SquareFunction(
            t_start=p3.t_start, t_end=p3.t_end, backend=self.backend_config
        )
        p3_com = p3.function * np.exp(-1j * np.pi * p3_t.function)

        # 4: Same as first but phase -Pi
        p4 = ShapedFunction(
            t_start=p3.t_end, area=np.pi / omega1, backend=self.backend_config
        )
        p4_t = SquareFunction(
            t_start=p4.t_start, t_end=p4.t_end, backend=self.backend_config
        )
        p4_com = p4.function * np.exp(-1j * np.pi * p4_t.function)

        self.t_end = p4.t_end

        p, q = self.pair[0], self.pair[1]
        couplings = [(p, p1.function), (q, p2.function), (q, p3_com), (p, p4_com)]

        detuning = [([1, 1], ZERO_FUNCTION.function)]

        coup, detun = coupling_detuning_constructors(
            couplings, detuning, omega_coup=[omega1, omega2, omega2, omega1]
        )
        qubit_schedule = RydbergQubitSchedule(
            coupling_waveforms=coup, detuning_waveforms=detun, backend=self.backend_config
        )

        self.q_schedule = (qubit_schedule, qubit_schedule)

        return (qubit_schedule, qubit_schedule)

class PCUxySchedule(GateSchedule):
    r"""Este es el schedule para la compuerta pCUxy en átomos de Rydberg.

    **Representación matricial:**

    .. math::

        pCU_{xy} = \qty( 
        \begin{array}{c c c c}
            \cos{(\theta/2)} & s(\theta, \varphi) & s(\theta, \varphi) & 0  \\
            s(\theta, -\varphi) & \cos^2{(\theta/4)} & -\sin^2{(\theta/4)} & 0 \\
            s(\theta, -\varphi) & -\sin^2{(\theta/4)} & \cos^2{(\theta/4)} & 0 \\
            0 & 0 & 0 & 1 
        \end{array}
    ) 
    
    s(\theta, \varphi) = -i\sin{(\theta/2)e^{i\varphi}}/\sqrt{2}

    **Operador evolución:**

    .. math::
        
         \hat{U} = \exp{-i\qty(\hat{H}^{01}_c + \hat{H}^{01}_t + \hat{H}^{1111}_{c,t})\tau_g}.

    
    """

    def __init__(
        self,
        theta: float = np.pi,
        phi: float = 0,
        t_start: float = 1,
        freq: float = 1,
        shape: str = "square",
        pair: Optional[list] = None,
        **kwargs
    ) -> None:
        if pair is None:
            pair = [0, 1]
        super().__init__(t_start, freq, pair, shape, **kwargs)

        self.theta = theta
        self.phi = phi

        self._schedule()

    def _schedule(self):
        target_schedule = UxySchedule(
            theta=self.theta,
            phi=self.phi,
            t_start=self.t_start,
            freq=self.freq,
            shape=self.shape,
            backend=self.backend_config,
        )

        control_schedule = UxySchedule(
            theta=self.theta,
            phi=self.phi,
            t_start=self.t_start,
            freq=self.freq,
            shape=self.shape,
            backend=self.backend_config,
        )

        self.t_end = target_schedule.t_end
        self.q_schedule = (target_schedule, control_schedule)


#####################################################################################################
# transpiler utils classes
#####################################################################################################

def get_transpilation_rule(name: str, transpilation_rules: dict) -> Callable:
    r"""Función que retorna la regla de transpilación asociada al nombre
    de una compuerta.

    Args:
        name (str): Nombre de la compuerta.
        transpilation_rules (dict): Reglas de transpilación disponibles.

    Raises:
        ValueError: Si la regla de transpilación para el nombre no existe.

    Returns:
        Callable: Regla de transpilación asociada al nombre.
    """
    try:
        return transpilation_rules[name]
    except Exception as exc:
        raise ValueError(f"No transpilation rule for {name}") from exc

def extract_qc_data(
    qc: RydbergQuantumCircuit,
) -> List[Tuple[str, List[float], int, List[int]]]:
    r"""Extrae los datos de un RydbergQuantumCircuit y los retorna
    en una lista que contiene el nombre, lista de parametros, numero de qubits
    y lista de los indices de qubits.

    Args:
        qc (RydbergQuantumCircuit): Circuito donde se sacarán los datos

    Returns:
        List[Tuple[str, List[float], int, List[int]]]: Lista de datos
    """
    gates = []

    for d in qc.data:
        name = d.operation.name
        params = d.operation.params
        num_qubits = d.operation.num_qubits
        qubits = [d.qubits[i].index for i in range(0, num_qubits)]
        gates.append((name, params, num_qubits, qubits))

    return gates

def circuit_schedule_init(num_qubits: int) -> dict:
    r"""Inicializa la estructura auxiliar del circuit schedule

    circuit_schedule = {
        "Qubit_#" : [ List[Schedules] , t_end ]
    }

    Args:
        num_qubits (int): Número de qubits del circuito.

    Returns:
        dict: La estructura auxiliar vacia construidoa.
    """
    circuit_schedule = {}

    for q in range(0, num_qubits):
        circuit_schedule[str(q)] = [[], 0]

    return circuit_schedule

def transpile_circ_sch(
    gates: list, transpilation_rules: dict, num_qubits: int, **kwargs
) -> dict:
    r"""Transpila el circuito sobre un circuit_schedule.

    Args:
        gates (list): Lista de los datos del circuito.
        transpilation_rules (dict): Reglas de transpilación disponibles.
        num_qubits (int): Número de qubits.

    Returns:
        dict: circuit_schedule con todo el circuito transpilado.
    """
    circuit_schedule = circuit_schedule_init(num_qubits)
    for gate in gates:
        name, params, num_qubits, qubits = gate
        if name == "barrier":
            continue
        apply_rule = get_transpilation_rule(name, transpilation_rules)
        args = {
            "name": name,
            "params": params,
            "num_qubits": num_qubits,
            "qubits": qubits,
            "circuit_schedule": circuit_schedule,
        }
        apply_rule(**args, **kwargs)

    return circuit_schedule

def construct_register_schedule(
    circuit_schedule: dict, num_qubits: int, **kwargs
) -> RydbergQRegisterSchedule:
    r"""Función que convierte un circuit_schedule en un RydbergQRegisterSchedule

    Args:
        circuit_schedule (dict): circuit_schedule que contienen el circuito
        transpilado.
        num_qubits (int): Número de qubits del circuito.

    Returns:
        RydbergQRegisterSchedule: Contiene todo el schedule del circuito.
    """
    register_schedule = []
    for i in range(0, num_qubits):
        # get the list of schedules of the qubit

        qubit_schedules = circuit_schedule[str(i)][0]
        qubit_couplings = {}
        qubit_detunings = {}
        if qubit_schedules != []:
            q_couplings = [q_s.q_schedule.coupling_waveforms for q_s in qubit_schedules]

            for j, q_c in enumerate(q_couplings):
                for i, value in enumerate(q_c.values()):
                    pair = value[0]
                    freq = value[1]
                    func = value[2]
                    qubit_couplings[f"Coupling{i+j}"] = [pair, freq, func]

            q_detunings = [q_s.q_schedule.detuning_waveforms for q_s in qubit_schedules]

            for j, q_c in enumerate(q_detunings):
                for i, value in enumerate(q_c.values()):
                    pair = value[0]
                    freq = value[1]
                    func = value[2]

                    qubit_detunings[f"Detuning{i+j}"] = [pair, freq, func]

            # Construct qubit schedule
            qubit_schedule = RydbergQubitSchedule(
                coupling_waveforms=qubit_couplings,
                detuning_waveforms=qubit_detunings,
                **kwargs,
            )
        else:
            qubit_couplings = {"Coupling0": [[0, 1], 0, ZERO_FUNCTION.function]}
            qubit_schedule = RydbergQubitSchedule(
                coupling_waveforms=qubit_couplings,
                detuning_waveforms=qubit_couplings,
                **kwargs,
            )

        register_schedule.append(qubit_schedule)

    return RydbergQRegisterSchedule(register_schedule)

def qc_to_ryd(
    qc: RydbergQuantumCircuit,
    transpilation_rules: dict = transpilation_rules,
    backend: aqipt.BackendConfig = aqipt.backend_config,
) -> RydbergQRegisterSchedule:
    r"""Transpila un RydbergQuantumCircuit y lo devuelve en su forma
    RydbergQRegisterSchedule.

    Args:
        qc (RydbergQuantumCircuit): Circuito a transpilar.
        transpilation_rules (dict, optional): Reglas de transpilación a utilizar. Defaults to transpilation_rules.
        backend (BackendConfig, optional): Configuración del backend a utilizar. Defaults to aqipt.backend_config.

    Returns:
        RydbergQRegisterSchedule: Schedule del circuito transpilado.
    """
    gates = extract_qc_data(qc);
    num_qubits = qc.qregs[0].size;
    circuit_schedule = {};
    circuit_schedule = transpile_circ_sch(gates, transpilation_rules, num_qubits, backend=backend);
    register_sch = construct_register_schedule(circuit_schedule, num_qubits, backend=backend);

    return register_sch


#####################################################################################################
# transpiler classes
#####################################################################################################

class Transpiler:
    def __init__(
        self,
        backend_config: aqipt.BackendConfig = aqipt.backend_config,
        transpilation_rules: Dict = transpilation_rules,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.backend_config = backend_config
        self.transpilation_rules = transpilation_rules

        self.qc = None
        self.quantum_register = None
        self.transpilation_time = None
        self.rydberg_schedule = None

    def transpile(self, qc) -> RydbergQRegisterSchedule:
        self.qc = qc
        time_start = time.time()
        rydberg_schedule = qc_to_ryd(
            qc, self.transpilation_rules, backend=self.backend_config
        )
        self.rydberg_schedule = rydberg_schedule
        time_end = time.time()

        self.transpilation_time = time_end - time_start
        self.rydberg_schedule = rydberg_schedule
        return rydberg_schedule

    def build_transpiled_circuit(
        self, init_state
    ) -> Union[RydbergQuantumRegister, RydbergQubit]:
        schedules = self.rydberg_schedule.schedules
        atomic_config = self.backend_config.atomic_config
        qubits = []

        if len(schedules) == 1:
            qubit = RydbergQubit(
                nr_levels=atomic_config.nr_levels,
                name="Qubit 0",
                initial_state=init_state,
                schedule=schedules[0],
                rydberg_states={
                    "RydbergStates": atomic_config.rydberg_states,
                    "l_values": atomic_config.l_values,
                },
                backend=self.backend_config,
            )

            self.quantum_register = qubit
            qubit._compile();
            qubit.simulate();
            return qubit

        if isinstance(init_state, str):
            _states_lst = bitstring2lst(init_state);
        else:
            _state_lst = init_state;

        for i, sch in enumerate(schedules):

            qubit = RydbergQubit(
                nr_levels=atomic_config.nr_levels,
                name=f"Qubit {i}",
                initial_state=0,
                schedule=sch,
                rydberg_states={
                    "RydbergStates": atomic_config.rydberg_states,
                    "l_values": atomic_config.l_values,
                },
                backend=self.backend_config,
            )

            qubit._compile()
            qubits.append(qubit)

        qr = RydbergQuantumRegister(
            qubits=qubits,
            layout=atomic_config.layout[: len(qubits)],
            init_state=init_state,
            connectivity=atomic_config.connectivity,
            c3=atomic_config.c3_constant,
            c6=atomic_config.c6_constant,
            backend=self.backend_config,
        )

        qr.compile()
        qr.simulate()

        self.quantum_register = qr

        return qr

    def __call__(self, qc) -> Any:
        return self.transpile(qc)

default_transpiler = Transpiler()

