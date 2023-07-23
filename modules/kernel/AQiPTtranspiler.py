#Atomic Quantum information Processing Tool (AQIPT) - DAQ module

# Author(s): 
# Angel Alvarez. Universidad Simon Bolivar. Quantum Information and Communication Group.
# Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2022-11-24
# Last update: 2022-11-24


#libs
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from functools import reduce
# import itertools
# import copy

# from tqdm import tqdm

# from numba import jit
# import numba
# import os, time, dataclasses
# import json

# import AQiPTcore as aqipt
from AQiPT.modules.directory import AQiPTdirectory as dirPath
# from qiskit import *
from qiskit.circuit import Gate, Qubit
from qiskit import QuantumCircuit, transpile, assemble, Aer
# from qiskit.quantum_info.operators import Operator
from qiskit.circuit.parameterexpression import ParameterValueType
# from qiskit.circuit.library.standard_gates.u3 import U3Gate, CU3Gate

from typing import List, Optional, Union


'''
    TO DO LIST
    ----------

        - Change _rydbergGATE(Gate) classes for an embedded instance in each method 
        while adding an instance of Gate() as an object and then bound the new methods 
        _define and _array to the new instance such it is created in the process.

        -we have to do a decomposegate() then added to decomposecircuit(), with rydbergqubit
        rydbergqregister() and quantumcircuit()
        
'''


#Qiskit custom gate class of native and compose canonical gates (parametrized)

class _rydbergUXY(Gate):
    '''
        AQiPT class of U_xy(theta, phi) Rydberg gate as Qiskit gate class.

    '''
    def __init__(self, theta: ParameterValueType, phi: ParameterValueType,
                 label: Optional[str] = None):
    
        super().__init__('Uxy', 1, [theta, phi], label=label)
    
    def _define(self):
        """
            U3Gate Qiskit gate definition of the unitary transformation
        """
        qc = RydbergQuantumCircuit(1, name=self.name)
        
        t = self.params[0]
        p = -(np.pi/2 + self.params[1])
        l = +p
        
        qc.append(U3Gate(t, p, l), [0], [])
        self.definition = qc
        
    
    def __array__(self, dtype=complex) -> np.array:
        """
            Matrix unitary definition as Numpy array 
        """
        theta, phi = self.params
        theta, phi = float(theta), float(phi)
        
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)
        epp = np.exp(1j*phi)
        epm = np.exp(1j*(-phi))
        
        return np.array(
            [
                [ cos,  -1j*sin*epp], 
                [ -1j*sin*epm, cos ]
            ],
        dtype=dtype
        )
    
class _rydbergCUXY(Gate):
    '''
        Clase que contiene la transformación de Control-Uxy(theta, phi) en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit
    '''
    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        label: Optional[str] = None,
    ):
    
        super().__init__('CUxy',2, [theta, phi], label=label)
    
    def _define(self):
        """
            Definición circuital de la transformación a base de CU3Gate() de Qiskit
        """
        qc = RydbergQuantumCircuit(2, name=self.name)
        
        t = self.params[0]
        p = -(np.pi/2 + self.params[1])
        l = -p
        
        
        qc.append(CU3Gate(t, p, l), [0, 1], [])
        self.definition = qc
    
    def __array__(self, dtype = complex):
        """
            Definición matricial de la transformación como un numpy.array 
        """
        theta, phi = self.params
        theta, phi = float(theta), float(phi)
        
        cos = np.cos(theta/2)
        sin = np.sin(theta/2)
        epp = np.exp(1j*phi)
        epm = np.exp(1j*(-phi))
        
        return np.array(
            [
                [ cos,  -1j*sin*epp , 0, 0], 
                [ -1j*sin*epm, cos , 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
        dtype=dtype
        )

class _rydbergCPHASE(Gate):
    '''
        Clase que contiene la transformación de CPhase(phi00, phi01, phi10, phi11) en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit.
    '''
    def __init__(
        self,
        phi00: ParameterValueType,
        phi01: ParameterValueType,
        phi10: ParameterValueType,
        phi11: ParameterValueType,
        label: Optional[str] = None,
    ):
    
        super().__init__('CPhase',2, [phi00, phi01, phi10, phi11], label=label)
    
    def _define(self):
        '''
            Definición circuital de la transformación a base de un operador matricial.
        '''
        phi00, phi01, phi10, phi11 = self.params
        e00 = np.exp(1j*phi00)
        e01 = np.exp(1j*phi01)
        e10 = np.exp(1j*phi10)
        e11 = np.exp(1j*phi11)
        
        qc = RydbergQuantumCircuit(2, name=self.name)
        
        cp = Operator(np.array(
                    [
                    [e00, 0, 0, 0],
                    [0, e01, 0, 0],
                    [0, 0, e10, 0],
                    [0, 0, 0, e11],
                   ])
                  )
        qc.unitary(cp, [0,1])
        self.definition = qc
    
    def __array__(self, dtype = complex):
        '''
            Definición matricial de la transformación como un numpy.array
        '''
        phi00, phi01, phi10, phi11 = self.params
        
        e00 = np.exp(1j*phi00)
        e01 = np.exp(1j*phi01)
        e10 = np.exp(1j*phi10)
        e11 = np.exp(1j*phi11)
        
        return np.array(
            [
                [ e00, 0, 0, 0], 
                [ 0, e01, 0, 0],
                [ 0, 0, e10, 0],
                [ 0, 0, 0, e11]
            ],
        dtype=dtype
        )

class _rydbergCZ(Gate):
    '''
        Clase que contiene la transformación Control-Phase(phi) en átomos de Rydberg.
    '''
    def __init__(self, phi: ParameterValueType ,label: Optional[str] = None,):
        super().__init__('CZ',2, [phi], label=label)
    
    def _define(self):
        '''
            Definición circuital a base de CPhase en átomos de Rydberg.
        '''
        qc = RydbergQuantumCircuit(2, name=self.name)
        qc.append(RydbergCPhase(0,0,0,self.params[0]), [0,1], [])
        self.define = qc
    
class _rydbergH(Gate):
    '''
        Clase que contiene la transformación de Hadamard en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit
    '''
    def __init__(self, label: Optional[str] = None):
        super().__init__('H ryd', 1, [], label=label)
        
    def _define(self):
        """
            Definición circuital de la transformación a base de Uxy() de Rydberg
        """
        qc = RydbergQuantumCircuit(1, name=self.name)
        qc.append(RydbergUxyGate(np.pi/2, -np.pi/2), [0])
        qc.append(RydbergUxyGate(np.pi, 0), [0])
        self.definition = qc

class _rydbergRX(Gate):
    '''
        Clase que contiene la transformación de Rx(theta) en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit
    '''
    def __init__(
        self, 
        theta: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__('Rx ryd$', 1, [theta], label=label)
        
    def _define(self):
        """
            Definición circuital de la transformación a base de Uxy() de Rydberg
        """
        qc = RydbergQuantumCircuit(1, name =self.name)
        qc.append(RydbergUxyGate(self.params[0], 0), [0])
        self.definition = qc
            
class _rydbergRY(Gate):
    '''
        Clase que contiene la transformación de Ry(theta) en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit
    '''
    def __init__(
        self, 
        theta: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__('Ry ryd', 1, [theta], label=label)
    
    def _define(self):
        """
            Definición circuital de la transformación a base de Uxy() de Rydberg
        """
        qc = RydbergQuantumCircuit(1, name =self.name)
        qc.append(RydbergUxyGate(self.params[0] ,-np.pi/2), [0])
        self.definition = qc
            
class _rydbergRZ(Gate):
    '''
        Clase que contiene la transformación de Rz(theta) en átomos de Rydberg.
        Hereda todas las cualidades de la clase Gate de Qiskit
    '''
    def __init__(
        self, 
        theta: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__('Rz ryd', 1, [theta], label=label)
    
    def _define(self):
        """
            Definición circuital de la transformación a base de Uxy() de Rydberg
        """
        qc = RydbergQuantumCircuit(1, name =self.name)
        qc.append(RydbergUxyGate(np.pi/2, -np.pi/2), [0])
        qc.append(RydbergUxyGate(self.params[0], 0), [0])
        qc.append(RydbergUxyGate(np.pi/2, np.pi/2), [0])
        self.definition = qc
 
class _rydbergCX(Gate):
    '''
        Clase que contiene la transformación Control-X en átomos de Rydberg.
    '''
    def __init__(self, label: Optional[str] = None,):
        super().__init__('CX',2, [], label=label)
    
    def _define(self):
        '''
            Definición de la transformación a base de un operador matricial
        '''
        phi = self.params
        ep = np.exp(1j*phi)
        
        qc = RydbergQuantumCircuit(2, name=self.name)
        
        cp = Operator(np.array(
                    [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, ep],
                   ])
                  )
        qc.unitary(cp, [0,1])
        self.definition = qc
               
class _rydbergSWAP(Gate):
    '''
        Clase que contiene la transformación Swap en átomos de Rydberg
    '''
    def __init__(self, label: Optional[str] = None,):
        super().__init__('Swap',2, [], label=label)
    
    def _define(self):
        '''
            Definición circuital a base de un operador matricial
        '''
        qc = RydbergQuantumCircuit(2, name=self.name)
        
        cp = Operator(np.array(
                    [
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                   ])
                  )
        qc.unitary(cp, [0,1])
        self.definition = qc
        
        self.define = qc

class NativeQuantumCircuit(QuantumCircuit):
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


    """Native Rydberg quantum gates"""
    def uxy(self, theta: float, phi: float, qubit: Union[int, List[int]] ):
        """
            Apply the transformation Uxy(theta, phi) over 'qubit'.

            INPUTS:
                theta : float
                    Rotation angle theta
                phi : float
                    Rotation angle phi
                qubit : Qubit [int]
                    Applied transformation's qubit

            OUTPUTS:
                NativeQuantumCircuit : QuantumCircuit
                    Rydberg circuit with Uxy(theta, phi) applied over 'qubit'

        """
        return self.append(_rydbergUXY(theta, phi), [qubit])
    
    def cuxy(self, theta: float, phi: float, control_qubit: int, target_qubit: Union[int, List[int]]):
        """
            Apply the transformation Controlled-Uxy(theta, phi) over 'target_qubit' accordingly to
            'control_qubit' state.

            INPUTS:
                theta : float
                    Rotation angle theta
                phi : float
                    Rotation angle phi
                control_qubit : Qubit [int]
                    Control qubit
                target_qubit : Qubit [int]
                    Applied transformation's qubit (target qubit)

            OUTPUTS:
                NativeQuantumCircuit : QuantumCircuit
                    Rydberg circuit with CUxy(theta, phi) applied over 'target_qubit' accordingly to
                    'control_qubit' state

        """
        return self.append(_rydbergCUXY(theta, phi), [control_qubit, target_qubit])
    
    def cphase(self, phi00: float, phi01: float, phi10: float, phi11:float, qubit1:Union[int, List[int]], qubit2: Union[int, List[int]]):
        """
            Método que aplica la transformación CPhase(phi00, phi01, phi10, phi11) entre 
            'qubit1' y 'qubit2'.

            Args:
                phi00 (float): Ángulo de rotación phi00
                phi01 (float): Ángulo de rotación phi01
                phi10 (float): Ángulo de rotación phi10
                phi11 (float): Ángulo de rotación phi11
                qubit1 (Qubit [int]): Qubit 1
                qubit2 (Qubit [int]): Qubit 2

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con CPhase(phi00, phi01, phi10, phi11)
                aplicada sobre 'qubit1' y 'qubit2'.

        """
        return self.append(_rydbergCPHASE(phi00, phi01, phi10,phi11), [qubit1, qubit2])
    
    """Canonical quantum gates composed by native Rydberg quantum gates"""

    def ryd_h(self, qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Hadamard sobre 'qubit'

            Args:
                qubit (Qubit [int]): Qubit donde se aplica la transformación

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Hadamard aplicada sobre 
                'qubit'

        """
        return self.append(_rydbergH(), [qubit])
    
    def ryd_rx(self, theta: float, qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Rotación en X sobre 'qubit'

            Args:
                theta (float): Ángulo de rotación sobre el eje X 
                qubit (Qubit [int]): Qubit donde se aplica la transformación

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Rx(theta) aplicada sobre 
                'qubit'

        """
        return self.append(_rydbergRX(theta), [qubit])
    
    def ryd_ry(self, theta: float, qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Rotación en Y sobre 'qubit'

            Args:
                theta (float): Ángulo de rotación sobre el eje Y 
                qubit (Qubit [int]): Qubit donde se aplica la transformación

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Ry(theta) aplicada sobre 
                'qubit'

        """
        return self.append(_rydbergRY(theta), [qubit])
    
    def ryd_rz(self, theta: float, qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Rotación en Z sobre 'qubit'

            Args:
                theta (float): Ángulo de rotación sobre el eje Z 
                qubit (Qubit [int]): Qubit donde se aplica la transformación

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Rz(theta) aplicada sobre 
                'qubit'

        """
        return self.append(_rydbergRZ(theta), [qubit])
    
    def ryd_cx(self, ctrl_qubit: int, target_qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Control-X o CNOT sobre 'control_qubit' y
            'target_qubit'

            Args:
                control_qubit (Qubit [int]): Qubit de control
                target_qubit (Qubit [int]): Qubit objetivo

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Control-X aplicada sobre
                'control_qubit' y 'target_qubit'
        """
        self.append(_rydbergCUXY(np.pi,0), [ctrl_qubit, target_qubit])
    
    def ryd_cz(self, phi: float, ctrl_qubit: int, target_qubit: Union[int, List[int]]):
        """
            Método que aplica la transformación Control-P o CP sobre 'control_qubit' y
            'target_qubit'

            Args:
                phi (float): Ángulo de rotación
                control_qubit (Qubit [int]): Qubit de control
                target_qubit (Qubit [int]): Qubit objetivo

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con Control-P(phi) aplicada sobre
                'control_qubit' y 'target_qubit'
        """
        self.append(_rydbergCZ(phi), [ctrl_qubit, target_qubit])
    
    def ryd_swap(self, qubit1: int, qubit2: int):
        """
            Método que aplica la transformación Swap entre 'qubit1' y 'qubit2'

            Args:
                qubit1 (Qubit [int]): Primer qubit de SWAP
                qubit2 (Qubit [int]): Segundo qubit de SWAP

            Returns:
                RydbergQuantumCircuit: Circuito de Rydberg con SWAP aplicada entre
                'qubit1' y 'qubit2'
        """
        self.append(_rydbergSWAP(), [qubit1, qubit2])


#look up table for gates
# Gates instance LUT (Look-UP-Table)
#NATIVE
ALG_UXY = hex(id(_rydbergUXY));
ALG_CUY = hex(id(_rydbergCUXY));
ALG_CPHASE = hex(id(_rydbergCPHASE));

# #CANONICAL
ALG_RYD_H = hex(id(_rydbergH));
ALG_RYD_RX = hex(id(_rydbergRX));
ALG_RYD_RY = hex(id(_rydbergRY));
ALG_RYD_RZ = hex(id(_rydbergRZ));
ALG_RYD_CX = hex(id(_rydbergCX));
ALG_RYD_CZ = hex(id(_rydbergCZ));
ALG_RYD_SWAP = hex(id(_rydbergSWAP));