#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Emulator module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): 
# Created: 2021-10-04
# Last update: 2024-12-14


#libs
import numpy as np
import qutip as qt

import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

from functools import reduce
import itertools, ctypes
from typing import Iterator, List
import copy, ctypes

import datetime

# import warnings
# warnings.filterwarnings('ignore')
np.seterr(under='ignore')

import networkx as nx
from tqdm import tqdm
import pandas as pd

import multiprocessing
from multiprocessing import Pool

from AQiPT import AQiPTcore as aqipt
import AQiPT.modules.kernel.AQiPTkernel as kernel

HZ_2_MHZ = 1/1e6;
HZ_2_KHZ = 1/1e3;

###################################################################################################
#######################                 Frontend DAQ                  #############################
###################################################################################################

#####################################################################################################
#atomicModel AQiPT class
#####################################################################################################

def eucdist(xa,ya,xb,yb,za=0,zb=0):
    '''
        Euclidean distance in 2D 
        Measure of the distance between the coordinates of two positions

        INPUTS:
        -------

            xa, ya, xb, yb : coordiantes of the object 

        OUTPUTS:
        --------

            float :  Euclidean distance given by $\sqrt((xa-xb)**2 + (ya-yb)**2)$


    '''

    return np.sqrt( (xb-xa)**2 + (yb-ya)**2 + (zb-za)**2)

def basis_nlvl(n):
    '''
        Basis state for n-lvl system

        Creates the eigenbasis of the n-level system

        INPUTS:
        -------
        
            n : number of levels of the system

        
        OUTPUTS:
        --------
        
            basis_lst (array) : eigenbasis as array object in the order |1>, |2>...

    '''
    qt_basis_lst = [qt.basis(n, state) for state in range(n)];
    return np.array(qt_basis_lst, dtype=object), qt_basis_lst

def ops_nlvl(n, basis_lst = None):
    '''
        Operators basis for n-lvl system

        Creates the operators basis of the n-level system

        INPUTS:
        ------
            n : number of levels of the system
            basis_lst : list of the system eigenbasis

        
        OUTPUTS:
        -------
            proyectors (array) : operator basis as array object in the order |1><1|, |1><2|...
            basis (array) : eigenbasis as array object in the order |1>, |2>...

    '''
    if basis_lst == None:
        np_basis, qt_basis = basis_nlvl(n);
        qt_proyectors = [ket1*(ket2.dag()) for ket1 in qt_basis for ket2 in qt_basis];

        return np.array(qt_proyectors, dtype=object), qt_basis, qt_proyectors
    else:
        qt_basis  = basis_lst;
        qt_proyectors = [ket1*(ket2.dag()) for ket1 in qt_basis for ket2 in qt_basis];

        return np.array(qt_proyectors, dtype=object), qt_basis, qt_proyectors

def iden(n):
    '''
        n-lvl identity operator

        Creates the identity operator for the n-level system

        INPUTS:
        -------
            n : number of levels of the system

        OUTPUTS:
        --------
            (QuTip object) : n by n square matrix with diagonal 1
    '''
    return qt.identity(n) #nxn identity operator

def lst2str(lst):
    '''
        List to string

        Transform a list into a string object

        INPUTS:
        -------
            lst (list) : list of characters to be converted into string.
    '''
    return ''.join(str(e) for e in lst)

#caculate operatos of interaction
def intBlockade_ops(at_nr, interacting_atoms, qdim=2):
    '''
        Internal Rydberg Blockade operators

        Calculates the blockade interaction operator coming from all-to-all interactions 
        with certain number of interacting atoms

        INPUTS:
        -------
            at_nr (int) : atom number in the blockade radius / (sub-)ensemble
            interacting_atoms (int) : number of remaining atoms in the GS and within
            qdim (int) : dimension of the atoms conside

        OUTPUTS:
        --------
    '''
    rr_op = qt.basis(qdim,qdim-1)*qt.basis(qdim,qdim-1).dag()

    rri_rrj_op = [];
    for i in interacting_atoms:
        op_list = [qt.qeye(qdim) for i in range(at_nr)];
        op_list[i] = rr_op;
        rri_rrj_op.append(qt.tensor(op_list))
    return rri_rrj_op

#calculate total blockade interation between all combinations of atoms within the ensemble
def totBlockadeInt(atoms_pos, at_nr, excitations_idx=None, qdim=2, c_val=1):

    if excitations_idx==None:
        interacting_atoms = range(at_nr);
    else:
        interacting_atoms = exc_at;

    block_ope = intBlockade_ops(at_nr, interacting_atoms, qdim); #basis of interaction operators of i-th atom
    BlockadeInt_op = qt.Qobj();

    for i in interacting_atoms:

        try:
            pos_atA = [atoms_pos[0][i], atoms_pos[1][i], atoms_pos[2][i]]; #list of positions of atoms i
        except:
            pos_atA = [atoms_pos[0][i], atoms_pos[1][i]]; #list of positions of atoms i

        for j in interacting_atoms:
            if i!=j:
                
                try:
                    pos_atB = [atoms_pos[0][j], atoms_pos[1][j], atoms_pos[2][j]]; #list of positions of atoms j
                except:
                    pos_atB = [atoms_pos[0][j], atoms_pos[1][j]]; #list of positions of atoms j
                
                try:    
                    intStrenght = c_val*2*np.pi/(eucdist(pos_atA[0], pos_atA[1], pos_atB[0], pos_atB[1], pos_atA[2],pos_atB[2]))**6; #strength coefficient of interaction
                except:
                    intStrenght = c_val*2*np.pi/(eucdist(pos_atA[0], pos_atA[1], pos_atB[0], pos_atB[1]))**6; #strength coefficient of interaction
                
                BlockadeInt_op += intStrenght*(block_ope[i]*block_ope[j]); #total blockade interaction operator sum(Vij |...ri...><...rj...|)
    return BlockadeInt_op


class atomicModel:
    
    '''
        A class for develope models based in the n-level system of an atomic model. AQiPT atomicModel()
        class contain the basic toolbox for solve, time-dependent and -independent dynamics for 1 or
        more level physical atomic registers with the help of the AQiPT control-module using the class of 
        functions, pulses, tracks, sequence etc.

        The atomicModel class is the AQiPT core of the emulator for the dynamics of the quantum registers. 
        From this class is possible to generate and store the Hamiltonians, Lindbladians, Observables, quantum
        registers maps and plot results for later use of other modules of AQiPT.


        Parameters
        ----------
        times : array
            Time of dynamics to be emulated.
        Nrlevels : int
            Number of levels of the quantum system.
        initState : int
            Initial state for the dynamics starting from 0 to any n.
        params : dict
            Dictionary with parameters of dynamcis e.g., couplings, detunings, dissipators, pulses
        name : str
            Label for the model

        Attributes
        ----------
        times : array
            Time of dynamics to be emulated.
        Nrlevels : int
            Number of levels of the quantum system.
        initState : Qobj() [QuTiP]
            Initial density matrix.
        dynParams : dict
            Dictionary with parameters of dynamcis e.g., couplings, detunings, dissipators, pulses
        _lstHamiltonian : list_like
            List of the single body Hamiltonian
        Hamiltonian : Qobj() [QuTiP]
            Hamiltonian as QuTiP object
        Hpulses : list_like
            List of time-dependent function numpy array for QobjEvo() [QuTiP]
        tHamiltonian : QobjEvo() [QuTiP]
            Total time-dependent Hamiltonian of the atomicModel()
        cops : Qobj() list [QuTiP]
            List of Lindbladians as QuTiP objects
        mops : Qobj() list [QuTiP]
            List of Observables as QuTiP objects
        _ops : list
            Full list of operators that spam the system
        _basis : list 
            Full list of eigenstates of the system
        atomicMap :
            Graph object of the n-level system
        RydbergMap :
            Graph object of Rydberg states
        _graph
            netWorkx graph object.
        _name : str
            Name of the model AQiPT object
        simOpts : Qobj() [QuTiP]
            Options QuTiP object for the mesolve() master equation solver
        simRes : Qobj() list [QuTiP]
            List of Qobj() related to the density matrix rho as function of time
        __mode : str
            Mode of Hamiltonian, 'control' for pulsed Hamiltonian (i.e., time-dependent) or 'free' for no time-dependent

        Methods
        -------
        __init__()
            Constructor of atomicModel() AQiPT class
        playSim()
            Solve dynamics for the Nrlevels-system at initState using params.
        buildHamiltonian()
            Construct Hamiltonians as Qobj() QuTiP object (Hermitian)
        buildTHamiltonian()
            Construct time-dependent Hamiltonian as QobjEvo() QuTiP object (Hermitian)
        buildLindbladians()
            Construct Lindbladians as QuTiP object (Non-Hermitian)
        buildObservables()
            Construct Observables as QuTiP object (Hermitian)
        getResult()
            Return result values from simulation as a list of QuTiP objects (Hermitian)
        showResults()
            Plot results coming from the simulation
        modelMap()
            Plot the graph associated to the n-level system of the atomic quantum register
        
    '''
    
    
    def __init__(self, times, Nrlevels, initState, params, name='atomicModel-defaultName', simOpt=qt.Options(nsteps=120000, rtol=1e-6, max_step=10e-6)):
        '''
            Constructor of the atomicQRegister() object of AQiPT
        '''
                
        #atributes
        
        self._np_ops, self._basis, self._ops = ops_nlvl(Nrlevels); #eigenoperators and eigenbasis
        
        self.times = times;
        self.Nrlevels = Nrlevels;
        self._psi0 = initState;
        if isinstance(initState, int):
            self.initState = self._basis[initState];
        else:
            self.initState = initState;

        self.dynParams = params;
        self._atom = None;
        self._states = None;
        self._lstHamiltonian = []; #list of Hamiltonians of the system
        self.Hamiltonian = None; #total Hamiltonian of the model (single matrix)
        self.Hpulses = None; #time-dependency of the Hamiltonian a.k.a pulses
        self.tHamiltonian = None; #time-dependent Hamiltonian as QobjEvo() of QuTiP
        self.internalInteraction = None; #internal interaction of the qubit in case of ensemble qubits

        self.zeeman_splitting_values = [];
        self._bfield = None;
        self.cops = None; #lindbladians
        self.mops = []; #observables


        self.atomicMap = {};
        self._graph = None;
        
        self._blochsphere = None;

        self._name = name;
        self.simOpts = simOpt;
        self.simRes = None;
        self.simRes_history = [];
        self.qubit_simRes = None;
        self.qubit_simRes_history = [];


        self.__mode = 'free';

    
    def playSim(self, mode='free', psi0='density-matrix', store_states=False):
        '''
            Execute simulation

            Play the simulation of the dynamics of the atomicModel() object and store the results in the attribute simRes. Using the solver:

                QuTiP-QME : Quantum master equation solver by QuTiP

            with two possible modes:

                free : continuos drived Hamiltonian solved by the Quantum master equation solver by QuTiP
                control : using time-dependent Hamiltonian, controlled by pulses; solved by the Quantum master equation solver by QuTiP
        '''

        if self.__mode=='free':
            if psi0=='state-vector':
                self.simOpts.store_states = store_states;
                self.simRes = qt.mesolve(self.Hamiltonian, self.initState, self.times, c_ops=self.cops, e_ops=self.mops, options=self.simOpts);
                self.simRes_history.append({str(datetime.datetime.now()) : self.simRes});
                print('Solving for \'free\' state-vector initial state.')

            elif psi0=='density-matrix':
                self.simOpts.store_states = store_states;
                self.simRes = qt.mesolve(self.Hamiltonian, qt.ket2dm(self.initState), self.times, c_ops=self.cops, e_ops=self.mops, options=self.simOpts);
                self.simRes_history.append({str(datetime.datetime.now()) : self.simRes});
                print('Solving for \'free\' density-matrix initial state')

        elif self.__mode=='control':

            if self.cops==None:
                self.simOpts.store_states = store_states;
                self.simRes = qt.mesolve(self.tHamiltonian, self.initState, self.times, e_ops=self.mops, options=self.simOpts);
                self.simRes_history.append({str(datetime.datetime.now()) : self.simRes});

            else:
                self.simOpts.store_states = store_states;
                self.simRes = qt.mesolve(self.tHamiltonian, self.initState, self.times, c_ops=self.cops, e_ops=self.mops, options=self.simOpts);
                self.simRes_history.append({str(datetime.datetime.now()) : self.simRes});

    def add_ZeemanSplitting(self, values_lst:list=None, atom=None, Bfield=None, state_lst=None, buildHamiltonian=False, buildTHamiltonian=False, printON=False):
        
        self._bfield = Bfield;
        self._atom = atom;
        self._states = state_lst;

        mu = 13996244936.1; #Hz/T
        _zeeman_splitting_values = [];

        if values_lst==None:
            
            _state_idx=0;
            for detuning in self.dynParams['detunings']:
                
                state = state_lst[_state_idx];
                
                gfactor = atom.getLandegfExact(state['l'], state['j'], state['f'], state['s']);
                _HF_zeeman = mu*Bfield*gfactor*state['mf']*HZ_2_MHZ; #for weak field in MHz and relative to the center of energy at mf=0
                
                _zeeman_splitting_values.append(_HF_zeeman);

                self.dynParams['detunings'][str(detuning)][1]+=_HF_zeeman;
                _state_idx+=1; 
        
        elif values_lst!=None:
            
            _value_idx=0;
            for detuning in self.dynParams['detunings']:
                self.dynParams['detunings'][str(detuning)][1]+=values_lst[_value_idx];
                _value_idx+=1;    

                _zeeman_splitting_values+=values_lst;
        else:
            print('No method added.')
        
        self.zeeman_splitting_values+=_zeeman_splitting_values;
        
        if printON==True:
            print(self.zeeman_splitting_values)

        if buildHamiltonian==True:
            self.buildHamiltonian();
        elif buildTHamiltonian==True:
            self.buildTHamiltonian();
        else:
            print('Re-build Hamiltonian required.')

    def buildTHamiltonian(self):
        '''
            Build time-dependent Hamiltonian 

            Construct the time-dependent Hamiltonian of the atomicModel() and store it in the attribute tHamiltonian, the time-dependt pulses
            in Hpulses and the parts structure of the Hamiltonian in _lstHamiltonian.
            .
        '''
        
        self.__mode = 'control'

        self._lstHamiltonian=[]; #reset attribute for recompilation
        
        _HQobjEVO = [];
        _HAQiPTpulses = [];
        
        if self.internalInteraction == None:
            for element in range(len(self.dynParams['couplings'])):
                
                # _HStruct = 0.5*self.dynParams['couplings']['Coupling'+str(element)][1] * (self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]] + self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]].dag());
                _HStruct = 0.5*self.dynParams['couplings']['Coupling'+str(element)][1] * (self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]] );
                _HStruct_dag = 0.5*self.dynParams['couplings']['Coupling'+str(element)][1] *self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]].dag();
                _HtDependency = self.dynParams['couplings']['Coupling'+str(element)][2];
                _HtDependency_dag = np.conjugate(self.dynParams['couplings']['Coupling'+str(element)][2]);
                _HAQiPTpulses.append(_HtDependency);
                _HAQiPTpulses.append(_HtDependency_dag);
                _HQobjEVO.append([_HStruct, _HtDependency]);
                _HQobjEVO.append([_HStruct_dag, _HtDependency_dag]);
                self._lstHamiltonian.append(_HStruct);
                self._lstHamiltonian.append(_HStruct_dag);
            
            for element in range(len(self.dynParams['detunings'])):
                
                _HStruct = 0.5*self.dynParams['detunings']['Detuning'+str(element)][1]*(self._ops[(self.dynParams['detunings']['Detuning'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['detunings']['Detuning'+str(element)][0])[1]] );
                _HtDependency = self.dynParams['detunings']['Detuning'+str(element)][2];
                _HAQiPTpulses.append(_HtDependency);
                _HQobjEVO.append([_HStruct, _HtDependency]);
                self._lstHamiltonian.append(_HStruct);
        
        else:

            for element in range(len(self.dynParams['couplings'])):

                _Fullspace = [iden(self.Nrlevels)]*self.dynParams['Ensembles']['Atom_nr'];

                _Fullspace_lst=[];
                for atom_idx in range(self.dynParams['Ensembles']['Atom_nr']):
                    _bufFullspace = _Fullspace.copy();
                    _bufFullspace[atom_idx] = self.dynParams['couplings']['Coupling'+str(element)][1] * (self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]] + self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]].dag());
                    _Fullspace_lst.append(qt.tensor(_bufFullspace));

                _HStruct = sum(_Fullspace_lst);
                _HtDependency = self.dynParams['couplings']['Coupling'+str(element)][2];

                _HAQiPTpulses.append(_HtDependency);
                _HQobjEVO.append([_HStruct, _HtDependency]);
                self._lstHamiltonian.append(_HStruct);
            
            for element in range(len(self.dynParams['detunings'])):
                
                _Fullspace = [iden(self.Nrlevels)]*self.dynParams['Ensembles']['Atom_nr'];

                _Fullspace_lst=[];
                for atom_idx in range(self.dynParams['Ensembles']['Atom_nr']):
                    _bufFullspace = _Fullspace.copy();
                    _bufFullspace[atom_idx] = self.dynParams['detunings']['Detuning'+str(element)][1]*(self._ops[(self.dynParams['detunings']['Detuning'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['detunings']['Detuning'+str(element)][0])[1]] );
                    _Fullspace_lst.append(qt.tensor(_bufFullspace));

                _HStruct = sum(_Fullspace_lst);
                _HtDependency = self.dynParams['detunings']['Detuning'+str(element)][2];
    
                _HAQiPTpulses.append(_HtDependency);
                _HQobjEVO.append([_HStruct, _HtDependency]);
                self._lstHamiltonian.append(_HStruct);


        self.tHamiltonian = qt.QobjEvo(_HQobjEVO, tlist=self.times);
        self.Hpulses = _HAQiPTpulses;
    
    def buildHamiltonian(self):
        '''
            Build Hamiltonian

            Construct and assign the Hamiltonian operator of the atomicModel() in two parts: 

            1) the total diagonal terms and 
            2) the total off-diagonal terms, finally sum them up and store it in the attribute Hamiltonian.
        '''
        HD = sum([self.dynParams['couplings']['Coupling'+str(element)][1] * (self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]] + self._ops[(self.dynParams['couplings']['Coupling'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['couplings']['Coupling'+str(element)][0])[1]].dag()) for element in range(len(self.dynParams['couplings'])) ]);
        HoffD = sum([self.dynParams['detunings']['Detuning'+str(element)][1]*(self._ops[(self.dynParams['detunings']['Detuning'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['detunings']['Detuning'+str(element)][0])[1]] ) for element in range(len(self.dynParams['detunings'])) ]);
        
        if self.internalInteraction == None:
            self.Hamiltonian = 0.5*(2*HD + 4*HoffD);
        else:
            # self.Hamiltonian = HD + HoffD;
            _Fullspace = [iden(self.Nrlevels)]*self.dynParams['Ensembles']['Atom_nr']; #empty string of operators for ensemble
            
            _Fullspace_lst=[];
            for atom_idx in range(self.dynParams['Ensembles']['Atom_nr']):
                _bufFullspace = _Fullspace.copy();
                _bufFullspace[atom_idx] = (HD + HoffD);
                _Fullspace_lst.append(qt.tensor(_bufFullspace));

            self.Hamiltonian = sum(_Fullspace_lst)  + self.internalInteraction;

    def buildLindbladians(self):
        '''
            Build Lindbladians

            Construct and assign Lindbland operators for the Quantum master equation solver of QuTiP stored in the cops attribute of the atomicModel.
        '''
        if self.internalInteraction == None:
            self.cops = [np.sqrt(self.dynParams['dissipators']['Dissipator'+str(element)][1])*(self._ops[(self.dynParams['dissipators']['Dissipator'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['dissipators']['Dissipator'+str(element)][0])[1]]) for element in range(len(self.dynParams['dissipators']))];
        else:
            _lindblandians = [np.sqrt(self.dynParams['dissipators']['Dissipator'+str(element)][1])*(self._ops[(self.dynParams['dissipators']['Dissipator'+str(element)][0])[0]*self.Nrlevels + (self.dynParams['dissipators']['Dissipator'+str(element)][0])[1]]) for element in range(len(self.dynParams['dissipators']))];
            _Fullspace = [iden(self.Nrlevels)]*self.dynParams['Ensembles']['Atom_nr']; #empty string of operators for ensemble
            
            _Fullspace_lst=[];
            for atom_idx in range(self.dynParams['Ensembles']['Atom_nr']):
                _bufFullspace = _Fullspace.copy();
                _bufFullspace[atom_idx] = sum(_lindblandians);
                _Fullspace_lst.append(qt.tensor(_bufFullspace));

            self.cops = _Fullspace_lst;

    def buildObservables(self, observables=None):
        '''
            Build Observables

            Construct and assign Observable operators for the Quantum master equation solver of QuTiP stored in the mops attribute of the atomicModel.
        '''
        if observables!=None:
            self.mops = observables;

        elif observables==[]:
            self.mops = [];
        else:
            if self.internalInteraction == None:
                
                self.mops = [self._ops[(self.Nrlevels+1)*i] for i in range(self.Nrlevels)];
            
            else:
                
                _bufbasis = ops_nlvl(self.Nrlevels**self.dynParams['Ensembles']['Atom_nr']);
                _ops = [];
                
                _counter=0;
                for i in range((self.Nrlevels**self.dynParams['Ensembles']['Atom_nr'])**2):
                    
                    if i%self.Nrlevels**self.dynParams['Ensembles']['Atom_nr'] == 0:
                        _ops.append(_bufbasis[1][i+_counter%self.Nrlevels**self.dynParams['Ensembles']['Atom_nr']])
                        _counter+=1;

                self.mops = _ops;
    
    def buildOnsiteInteraction(self, c6_val=1):

        '''
            Build on-site interaction operators

            Assign the interaction operator into the intalInteraction attribute of the atomicModel

            INPUTS:
            -------

            c6_val : value of the C6 coefficient

        '''
        self.internalInteraction = totBlockadeInt(self.dynParams['Ensembles']['Atom_pos'], self.dynParams['Ensembles']['Atom_nr'], qdim=self.Nrlevels, c_val=c6_val)

    def getResult(self, states='qudit'):
        '''
            Get results of atomicModel

            Assign the values of the simulation executed with playSim() into simRes attribute.
            
            mode (str): 'qudit' will show full dynamics of all levels
                        'qubit' will show dynamics of qubit computational states
        '''

        if states=='qudit':
            return self.simRes
        elif states=='qubit':
            return self.qubit_simRes
    
    def getTHamiltonian(self, time_dependency=False):

        if time_dependency:

            return qt.Qobj(np.sum(self._lstHamiltonian, axis=0)), self.Hpulses

        else:

            return qt.Qobj(np.sum(self._lstHamiltonian, axis=0))
        
    def _check_qubitstate(self):

        if 'qubitstate' not in self.dynParams:
            self.dynParams['qubitstate'] = [0, 1];

    def qudit2qubit(self):

        _result_states = self.getResult().states;
        
        self._check_qubitstate();
        #TODO: to be replace for allocated states
        states=[];
        for _state_idx in range(self.Nrlevels):
            if _state_idx in self.dynParams['qubitstate']:
                states.append(qt.basis(self.Nrlevels, _state_idx));#TODO: fix index: dont correspond to state index

        _new_results = []
        for _result in _result_states:
            _reshaped_operator = qt.Qobj([[_result.matrix_element(bra, ket) for ket in states] for bra in states]);
            _reshaped_operator.dims = [[2],[2]];

            _new_results.append(_reshaped_operator);
            
        self.qubit_simRes = _new_results;
        self.qubit_simRes_history.append(_new_results);


    def showResults(self, plot_mode='matplotlib', resultseq=None, resultTitle=None, 
                    figure_size=(15,9), 
                    labels=None,
                    fontsize=18, 
                    figure=None, 
                    axis=None, 
                    legendON=True, 
                    report=False, 
                    time_unit=r'$\mu$s', 
                    bloch_step=100,
                    states= 'qudit' ):
        '''
            Show Results of atomicModel

            Returns the matplotlib fig and axis, showing the results from the Simulation using playSim().

            INPUTS:
            -------

            resultseq : results from simulation given externally
            sizefig : size of the matplotlib figure, set as (15,9) by default
            labels : list of labels for legend

            
            OUTPUS:
            -------
            fig : matplotlib figure
            axs : matplotlib axis

        '''
        
        if plot_mode=='plotly':
            if figure_size==(10,6):
                figure_size=(1650,500);
            resultseq = self.simRes
            if labels == None:
                labels = [lst2str(i) for i in list(itertools.product(*[range(self.Nrlevels)]))]
                self._basis = labels

            if figure==None and axis==None:
                fig = go.Figure()

                for i in range(len(resultseq.expect)):
                    if labels != None:
                        fig.add_trace(go.Scatter(x=self.times, y=resultseq.expect[i], name=labels[i]))
                    else:
                        fig.add_trace(go.Scatter(x=self.times, y=resultseq.expect[i], name=str(i)))

                if legendON:
                    fig.update_layout(showlegend=True)
                else:
                    fig.update_layout(showlegend=False)

                fig.update_layout(
                    xaxis_title='Time [{TIME_UNIT}]'.format(TIME_UNIT=time_unit),
                    yaxis_title='Population',
                    title=resultTitle if resultTitle != None else self._name,
                    width=figure_size[0],
                    height=figure_size[1]
                )
                return fig

        elif plot_mode=='bloch-sphere':

            if self.Nrlevels==2:
                self.buildObservables(observables=[qt.sigmax(), qt.sigmay(), qt.sigmaz()]);    #building Observables
                self.playSim(psi0='state-vector', store_states=True);

                if figure!=None and axis==None:
                    self._blochsphere = qt.Bloch(fig=figure);
                elif figure==None and axis!=None:
                    self._blochsphere = qt.Bloch(axes=axis);
                else:
                    self._blochsphere = qt.Bloch(fig=figure, axes=axis);

                self._blochsphere.vector_color = ['b']; #color of the initial state vector

                f_point = (bloch_step)*(int(len(self.times)/bloch_step)-1);
                for _idx in range(0,len(self.times), bloch_step):

                    psi_0 = (self.simRes_history[-1])[list(self.simRes_history[-1])[-1]].states[_idx];
                    
                    #adding initial and final vector
                    if _idx==0 or _idx==f_point:
                        self._blochsphere.add_states(psi_0);

                    _point_0  = [qt.expect(qt.sigmax(),psi_0), qt.expect(qt.sigmay(),psi_0), qt.expect(qt.sigmaz(),psi_0)];
                    self._blochsphere.add_points(_point_0);

                self._blochsphere.vector_alpha = [0.4 for _ in self._blochsphere.vector_alpha];
                self._blochsphere.vector_color = ['b','r'];
                self._blochsphere.point_color = ['b']+['gray']*(int(len(self.times)/bloch_step)-2)+['r'];
                self._blochsphere.point_marker = ['.']*int(len(self.times)/bloch_step);


                self._blochsphere.xlabel = [r'$\frac{|0\rangle + |1\rangle}{\sqrt{2}}$', r'$\frac{|0\rangle - |1\rangle}{\sqrt{2}}$'];
                self._blochsphere.ylabel = [r'$\frac{|0\rangle + i|1\rangle}{\sqrt{2}}$', r'$\frac{|0\rangle - i|1\rangle}{\sqrt{2}}$'];
                self._blochsphere.frame_color = 'gray';
                self._blochsphere.frame_alpha = 0;
                self._blochsphere.background = False;

                self._blochsphere.render();


            elif self.Nrlevels>2:
                self.qudit2qubit();


                self._blochsphere = qt.Bloch(fig=figure); 
                self._blochsphere.vector_color = ['b']; #color of the initial state vector

                f_point = (bloch_step)*(int(len(self.times)/bloch_step)-1);
                for _idx in range(0,len(self.times), bloch_step):

                    psi_0 = (self.qubit_simRes_history[-1])[_idx];
                    
                    #adding initial and final vector
                    if _idx==0 or _idx==f_point:
                        self._blochsphere.add_states(psi_0);

                    _point_0  = [qt.expect(qt.sigmax(),psi_0), qt.expect(qt.sigmay(),psi_0), qt.expect(qt.sigmaz(),psi_0)];
                    self._blochsphere.add_points(_point_0);

                self._blochsphere.vector_alpha = [0.4 for _ in self._blochsphere.vector_alpha];
                self._blochsphere.vector_color = ['b','r'];
                self._blochsphere.point_color = ['b']+['gray']*(int(len(self.times)/bloch_step)-2)+['r'];
                self._blochsphere.point_marker = ['.']*int(len(self.times)/bloch_step);


                self._blochsphere.xlabel = [r'$\frac{|0\rangle + |1\rangle}{\sqrt{2}}$', r'$\frac{|0\rangle - |1\rangle}{\sqrt{2}}$'];
                self._blochsphere.ylabel = [r'$\frac{|0\rangle + i|1\rangle}{\sqrt{2}}$', r'$\frac{|0\rangle - i|1\rangle}{\sqrt{2}}$'];
                self._blochsphere.frame_color = 'gray';
                self._blochsphere.frame_alpha = 0;
                self._blochsphere.background = False;

                self._blochsphere.render();

        else:

            if states=='qudit':
                if resultseq==None:
                    resultseq = self.simRes;
                
                fig, axs = plt.subplots(figsize=figure_size);

                if labels is None:
                    for i in range(len(resultseq.expect)):
                        axs.plot(self.times, resultseq.expect[i], label=i, alpha=0.5, linewidth=1.5);
                else:
                    for i in range(len(resultseq.expect)):
                        axs.plot(self.times, resultseq.expect[i], label=labels[i], alpha=0.5, linewidth=1.5);
                plt.legend();
                plt.xlabel('Time [{TIME_UNIT}]'.format(TIME_UNIT=time_unit), fontsize=fontsize);
                plt.ylabel('Population', fontsize=fontsize);

                if resultTitle!=None:
                        plt.title(resultTitle);
                else:
                    plt.title(self._name);

                return fig, axs

            elif states=='qubit':
                if resultseq==None:
                    resultseq = self.qubit_simRes;
                
                fig, axs = plt.subplots(figsize=figure_size);

                if labels is None:
                    for i in range(len(resultseq)):
                        axs.plot(self.times, resultseq[i], label=i, alpha=0.5, linewidth=1.5, fontsize=fontsize);
                else:
                    for i in range(len(resultseq)):
                        axs.plot(self.times, resultseq[i], label=labels[i], alpha=0.5, linewidth=1.5);
                plt.legend();
                plt.xlabel('Time [{TIME_UNIT}]'.format(TIME_UNIT=time_unit), fontsize=fontsize);
                plt.ylabel('Population', fontsize=fontsize);

                if resultTitle!=None:
                        plt.title(resultTitle);
                else:
                    plt.title(self._name);

                return fig, axs
    
    def showHamiltonian(self, plot_mode='qutip', figure_size=(15,15), qutip_style="scaled", unit='MHz', color_map='plasma'):

        '''

        '''

        if plot_mode=='qutip':

            qt.hinton(self.Hamiltonian, title='Hamiltonian', label_top=False, color_style=qutip_style)

        elif plot_mode=='matplotlib':

            fig, axs = plt.subplots(1,2, figsize=(15,15));

            ticks = range(self.Nrlevels);
            tick_labels = [r'$|{VALUE}\rangle$'.format(VALUE=i) for i in range(self.Nrlevels)];
            

            rHamiltonian = axs[0].imshow(np.real(self.Hamiltonian), cmap=color_map);
            cbar1 = fig.colorbar(rHamiltonian, ax=axs[0],fraction=0.046, pad=0.046);
            cbar1.set_label(r' {UNIT}/$2\pi$'.format(UNIT=unit));
            axs[0].set_title('Real part Hamiltonian: '+self._name);
            axs[0].set_xticks(ticks); # `ticks` is a list here!
            axs[0].set_yticks(ticks); # `ticks` is a list here!

            axs[0].set_xticklabels(tick_labels, rotation=90, fontsize=8);  # Set custom tick labels
            axs[0].set_yticklabels(tick_labels, rotation=0, fontsize=8);  # Set custom tick labels
            axs[0].grid();

            iHamiltonian = axs[1].imshow(np.imag(self.Hamiltonian), cmap=color_map);
            cbar2 = fig.colorbar(iHamiltonian, ax=axs[1], fraction=0.046, pad=0.046);
            cbar2.set_label(r'{UNIT}/$2\pi$'.format(UNIT=unit));
            axs[1].set_title('Imaginary part Hamiltonian: '+self._name);
            axs[1].set_xticks(ticks); # `ticks` is a list here!
            axs[1].set_yticks(ticks); # `ticks` is a list here!

            axs[1].set_xticklabels(tick_labels, rotation=90, fontsize=8);  # Set custom tick labels
            axs[1].set_yticklabels(tick_labels, rotation=0, fontsize=8);  # Set custom tick labels
            axs[1].grid();

    def modelMap(self, plotON=True, figure_size=(8,8)):
        '''
            Construct and return the Graph plot and Lindbland terms for the Quantum master equation solver of QuTiP.
        '''
        #
        for state in range(self.Nrlevels):
            self.atomicMap[state] = [];
            
        #setting edges
        for coupling in self.dynParams['couplings'].items():
            i, j = coupling[1][0];
            self.atomicMap[i].append(j);
            self.atomicMap[j].append(i);
        
        edge_list = [];
        for dissipator in self.dynParams['dissipators'].items():    
            i,j = dissipator[1][0];
            if i==j:
                edge_list.append(tuple([i,j]));

        plt.figure(figsize=figure_size) 
        G = nx.MultiDiGraph(self.atomicMap, create_using=nx.DiGraph, seed = 100);
        
        #edges for dephasing
        G.add_edges_from(edge_list);
              
        if 'rydbergstates' in self.dynParams:
            color_map = ['m' if node in self.dynParams['rydbergstates']['RydbergStates'] else 'dodgerblue' for node in G] 
        else:
            color_map = 'mediumseagreen'
            
        #plotting
        if plotON==True:
            pos = nx.circular_layout(G);
            nx.draw(G, with_labels=True, font_weight='regular', node_color=color_map, node_size=400, linewidths=7, font_size=15);
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, arrowstyle="<|-", style="solid");
            
        self._graph = {'graph': G, 'colormap':color_map, 'edges': edge_list}; #store the graph in attribute

        return self.atomicMap
   
    def recompile(self, mode):
        if mode=='free':

            self.buildHamiltonian();
            self.buildLindbladians();
            self.buildObservables();

            print('Recompilation complete. Mode: '+mode)

        elif mode=='control':

            self.buildTHamiltonian();
            self.buildLindbladians();
            self.buildObservables();

            print('Recompilation complete. Mode: '+mode)

#####################################################################################################
#atomicQRegister AQiPT class
#####################################################################################################

def AM_superpositionState(state:str, Nrlevels:int):

    qtotal=0; qidx=0;
    for q in [qt.basis(Nrlevels, state) for state in bitstring2lst(state)]:
        qtotal+=q; qidx+=1;
    
    return (1/np.sqrt(qidx))*qtotal;

def AM_mixedState(state:str, weights:list, Nrlevels:int):

    qtotal=0; qidx=0;
    for q in [weight*(qt.basis(Nrlevels, state)*qt.basis(Nrlevels, state).dag()) for state, weight in zip(bitstring2lst(state), weights)]:
        qtotal+=q; qidx+=1;
    
    return (1/np.sqrt(qidx))*qtotal;

def append_to_list_if_not_exists(lst, arr):
    for l in lst:
        if np.array_equal(l, arr):
            return
    lst.append(arr)

def perm_w_repl(n: int, m: int) -> Iterator[List[int]]:
    cur = []

    def perm_w_repl_recur(n_rec: int, m_rec: int) -> Iterator[List[int]]:
        nonlocal cur

        if n_rec == 0:
            yield cur
            return

        for i in range(1, m_rec + 1):
            cur = cur + [i]
            yield from perm_w_repl_recur(n_rec - 1, m_rec)
            cur.pop()

    yield from perm_w_repl_recur(n, m)

#transform from list to string
def lst2string(lst):
    return ''.join(str(e) for e in lst)

#convert bit-string to string
def bitstring2lst(string): 
    list1=[] 
    list1[:0]=string 
    return [ int(x) for x in list1 ] 

#transformation of qudit bit-string to qudit ket
def nqString2nqKet(psi0_string, bitdim=2, bitsdim=None):
    if bitsdim==None:
        ket_lst = [c for c in psi0_string];
        psi = qt.Qobj();    
        counter=0;
        for i in range(len(ket_lst)):
            if counter==0:
                psi = qt.ket(ket_lst[i],bitdim);
                counter+=1;
            else:
                psi= qt.tensor(psi, qt.ket(ket_lst[i],bitdim));
        return psi
    else:
        ket_lst = [c for c in psi0_string];
        psi = qt.Qobj();    
        counter=0;
        for i in range(len(ket_lst)):
            if counter==0:
                psi = qt.ket(ket_lst[i],bitsdim[i]);
                counter+=1;
            else:
                psi= qt.tensor(psi, qt.ket(ket_lst[i],bitsdim[i]));
        return psi

#generate binary bit-strings of n-bits and k-ones in bit
def bitCom(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result

#generate non-binary bit-strings of nr_at atoms and qdim qudit dimension
def mbitCom(nr_at=1, qdim=3):
    lst_comb=[];

    for i in perm_w_repl(nr_at, qdim):
        lst_comb.append([e - 1 for e in [*i]])
    return lst_comb

#set observables of dim^n Hilbert space system
def obs(at_nr, qdim=2):
    if qdim==2:
        bit_lst = [];
        for i in range(at_nr+1):
            bit_lst+=bitCom(at_nr,i);
        obs_lst = [];
        for e in bit_lst:
            ket = qt.basis([qdim]*at_nr, bitstring2lst(e));
            obs_lst.append((ket* ket.dag()) )
        obs_tot = sum(obs_lst)
        #possible to graphically check with: qt.matrix_histogram_complex(obs_tot)
        return obs_lst, bit_lst
    if qdim!=2:
        bit_lst = [lst2string(i) for i in mbitCom(at_nr, qdim)];
        obs_lst = [qt.basis(qdim, i)*qt.basis(qdim, i).dag() for i in range(qdim)];
        return obs_lst, bit_lst

#caculate operatos of interaction
def rydbergInteraction(qubit_nr, interacting_atoms, qdim=2):
    '''
        Rydberg Interaction operators

        Calculates the blockade interaction operator coming from all-to-all interactions 
        with certain number of interacting atoms

        INPUTS:
        -------
            at_nr (int) : atom number in the blockade radius / (sub-)ensemble
            interacting_atoms (int) : number of remaining atoms in the GS and within
            qdim (int) : dimension of the atoms conside

        OUTPUTS:
        --------
    '''
    rr_op = qt.basis(qdim,qdim-1)*qt.basis(qdim,qdim-1).dag()

    rri_rrj_op = [];
    for i in interacting_atoms:
        op_list = [qt.qeye(qdim) for i in range(qubit_nr)];
        op_list[i] = rr_op;
        rri_rrj_op.append(qt.tensor(op_list))
    return rri_rrj_op

#swap elements of a list
def _swapElements(LST, pos1, pos2):
     
    LST[pos1], LST[pos2] = LST[pos2] , LST[pos1];
    return LST

#combinations of list lst of values without elements i and j
def generate_combinations(lst, i, j, values):
    combinations = list(itertools.product(*values));

    results = [];
    
    for combination in combinations:
        result = lst.copy(); 
        for index, value in enumerate(combination):
            if index != i and index != j:
                result[index] = value;
                append_to_list_if_not_exists(results, result);

    return results

#eliminate duplicates in list
def eliminate_duplicates(lst):
    unique_list = []
    for item in lst:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

class atomicQRegister:
    
    '''
        A class for develope models based in the n-level system of an atomic register. AQiPT atomicQRegister()
        class contain the basic toolbox for solve, time-dependent and -independent dynamics for 1 or
        more physical atomic atomicModel() objects with the help of the AQiPT control-module using the class of 
        functions, pulses, tracks, sequence etc.

        The pulse class is the AQiPT core of the emulator for the dynamics of quantum registers based in atomic
        systems. From this class is possible to generate and store the Hamiltonians, Lindbladians, Observables, quantum
        registers maps and plot results for later use of other modules of AQiPT.


        Parameters
        ----------
        physicalRegisters : array
            Time of dynamics to be emulated.
        Nrlevels : int
            Number of levels of the quantum system.
        initnState : int
            Initial state for the dynamics starting from 0 to any n.
        params : dict
            Dictionary with parameters of dynamcis e.g., couplings, detunings, dissipators, pulses
        name : str
            Label for the model

        Attributes
        ----------
        _AMs : list_like
            List of atomicModel() objects
        AMconfig : 
            Configuration attribute of the registers given by spacial disposition.
        times : array
            Time of dynamics to be emulated from the first atomicModel object.
        lstNrlevels : list_like
            List of the number of levels of the atomicModels() or registers
        Nrlevels : int
            Number of levels of the quantum system.
        NrQReg : int
            Number of atomicModel() or registers
        initnState : Qobj() [QuTiP]
            Initial state of the density matrix of the full atomicQRegister.
        lstinitState : list_like
            List of the initial state of the atomicModel() AQiPT class.
        dynParams : dict
            Dictionary with parameters of dynamic parameter of the registers i.e., couplings, detunings, dissipators, pulses
        lstHamiltonian : list_like
            List of full Hamiltonian of the atomicModel class.
        _lsttHamiltonian : list_like
            List of the time-dependent Hamiltonian Qobj() [QuTiP] class
        lstPulses : list_like
            List of time-dependent function numpy array
        nHamiltonian : Qobj() [QuTiP]
            Hamiltonian of the full Hilbert space as QuTiP object
        tnHamiltonian : Qobj() [QuTiP]
            Time-dependent Hamiltonian of the full Hilbert space
        Hpulses : list_like
            List of time-dependent function numpy array for QobjEvo() [QuTiP]
        lstcops : list_like
            List of collapse operators of the system for the Lindbland term
        ncops : Qobj() list [QuTiP]
            List of Lindbladians as Qobj() QuTiP class
        lstmops : list_like
            List of measurement operators of the system for the Lindbland term
        nmops : Qobj() list [QuTiP]
            List of Observables as Qobj() QuTiP class
        _ops : list
            Full list of operators that spam the system
        _basis : list 
            Full list of eigenstates of the system
        atomicRegister :
            Graph object of the n-level registers
        _graph :
            networkx Graph object of the NrQReg-atomicModel()
        _name : str
            Name of the atomicQRegister() AQiPT class
        _homogeneous : Bool 
            Quantum register homogeneity of the number of levels in each register            
        simOpts : Qobj() [QuTiP]
            Options QuTiP object for the mesolve() master equation solver
        simRes : Qobj() list [QuTiP]
            List of Qobj() related to the density matrix rho as function of time
        __mode : str
            Mode of Hamiltonian, 'control' for pulsed Hamiltonian (i.e., time-dependent) or 'free' for no time-dependent
        connectivity : list
            Map of connectivity between physical registers via Rydberg states. Blind to the interaction strength
        layout : list
            Map of spatial distribution of physical registers
        _HSlist : list
            Hilbert space indexes of the interacting states of the qudits that belongs to the QRegister (label of Rydberg states)
        _AMslevels : list
            List of atomicModel's levels in the atomicQRegister.

        Methods
        -------
        __init()__
            Contructor of the atomicQResgister() AQiPT class
        playSim()
            Solve dynamics of the physicalRegisters of atomicModel() with different or same Nrlevels-system at initState using dynparams.
        buildNinitState()
            Construct NrQReg initial state of the atomicQRegister as Qobj() [QuTiP] class
        buildNHamiltonian()
            Construct  NrQReg dimensional  Hamiltonian of the atomicQRegister as Qobj() [QuTiP] class (Hermitian)
        buildTNHamiltonian()
            Construct time-dependent  NrQReg dimensional  Hamiltonian of the atomicQRegister as QobjEvo() [QuTiP] class (Hermitian)   
        buildNLindbladians()
            Construct  NrQReg dimensional Lindbladians as Qobj() [QuTiP] class (Non-Hermitian)
        buildNObservables()
            Construct NrQReg dimensional Observables as Qobj() [QuTiP] class (Hermitian)
        add2QRegister()
            Add new model to the atomicQRegister() from parameters or from a predefined atomicModel()
        getNHamiltonian()
            Return NrQReg dimensional Hamiltonian of the atomicQRegister as QobjEvo() [QuTiP] class (Hermitian)   
        getNLindbladian()
            Return NrQReg dimensional Lindbladians as [QuTiP] class (Non-Hermitian)
        getNObservables()
            Return NrQReg dimensional Observables as [QuTiP] class (Hermitian)
        getResult()
            Return result values from simulation as a list of QuTip objects (Hermitian)
        showResults()
            Plot results coming from the simulation
        registerMap()
            Plot the graph associated atomicQRegister()
            
    '''

    def __init__(self, physicalRegisters, initnState=None, name='atomicQRegister-DefaultName', 
                 times=None, NrQReg=None, homogeneous=True, lstNrlevels=None,
                 connectivity=['All'], layout=None, map=[]):
        '''
            Constructor of the atomicQRegister() object of AQiPT
        '''
                
        #atributes
        
        self._AMs = physicalRegisters;
        self.AMconfig = None;
        

        if times == None:
            self.times = self._AMs[0].times;
        else:
            self.times = times;
        
        if lstNrlevels==None:
            self.lstNrlevels = [AM.Nrlevels for AM in self._AMs];
        else:
            self.lstNrlevels = lstNrlevels;
        self.Nrlevels = reduce(lambda x, y: x * y, self.lstNrlevels);
        self._levels = None;
        self._AMslevels = [list(np.arange(lvlAM)) for lvlAM in self.lstNrlevels];
        self._AMslevels_ket = [[qt.basis(self.lstNrlevels[k1], k2) for k2 in self._AMslevels[k1]] for k1 in range(len(self._AMslevels))];
        self.NrQReg = len(self._AMs);

        self.__HilbertSpaceSize = sum(self.lstNrlevels);
        self._HSlist = [];

        self.initnState = initnState;
        if initnState==None:
            self.lstinitState = [AM.initState for AM in self._AMs];
            self.initnState = initnState;
        if isinstance(initnState, str):
            self.lstinitState = initnState;
        if isinstance(initnState, qt.Qobj):
            self.initnState = initnState;
        
        self.dynParams = [AM.dynParams for AM in self._AMs];
        
        self.lstHamiltonian = [AM.Hamiltonian for AM in self._AMs];
        self._lsttHamiltonian = [];
        self.lstPulses = [AM.Hpulses for AM in self._AMs];
        self.nHamiltonian = None;
        self.tnHamiltonian = None;
        self._Vint = None;
        self.Hpulses = None;
        
        self.lstcops = [AM.cops for AM in self._AMs];
        self.ncops = None;
        
        self.lstmops = [AM.mops for AM in self._AMs];
        self.nmops = [];
        
        # self._np_ops, self._ops, self._basis = ops_nlvl(self.Nrlevels);
        self._basisString = [lst2str(i) for i in list(itertools.product(*[range(AM.Nrlevels) for AM in self._AMs]))];
        self._basis = [];
        self._intbasis = [];

        self._pairInteraction_idx = [];
        self._pairInteraction_lval = [];

        self.nC6Interaction = None;

        self._C3pairInteraction_idx = [];
        self.nC3Interaction = None;
        self._combinationsC3 = None;

        self._graphRegister = None;
        self._graphscolors = None;
        self._rydbergstates = [];
        self._groundstates = [];
        self._rydbergstatesAM = [];
        self._groundstatesAM = [];
        
        _Hcount=0;
        for i in range(len(self._AMs)):
            _HS_AMS=[];
            _HS_adj_AMS=[];
            for _rydState in self._AMs[i].dynParams['rydbergstates']['RydbergStates']:
                self._rydbergstates.append( _rydState + _Hcount);
                _HS_adj_AMS.append(_rydState)
                _HS_AMS.append(_rydState + _Hcount);
            self._HSlist.append(_HS_AMS);
            self._rydbergstatesAM.append(_HS_adj_AMS);
            self._groundstatesAM.append([j for j in self._AMslevels[i] if j not in _HS_adj_AMS]);
            _Hcount+=self._AMs[i].Nrlevels;

        self._groundstates = [item for sublist in self._groundstatesAM for item in sublist];

        if connectivity[0]=='All':
            self.connectivity = connectivity;
        elif connectivity[0]=='Bidirected':
            self.connectivity = connectivity[1] + [tuple(np.flip(i)) for i in connectivity[1]];
        elif connectivity[0]=='Directed':
            self.connectivity = connectivity[1];
        self._connectivityType = connectivity[0];

        self.layout = layout;

        self.map = [];
        self._graphMap = None;

        self.compileQRegister = {};

        self._name = name;
        self._homogeneous = homogeneous;
        self.simOpts = None; #qt.Options(nsteps=500, rtol=1e-7, max_step=10e-1);
        self.simRes = None;
        self.__mode = 'free';

        
    def compile(self, plotON=False):
        Q = nx.MultiDiGraph({}, create_using=nx.DiGraph, seed=100);

        for connection in self.connectivity:
            #define nodes ith and jth
            node_i = connection[0];
            node_j = connection[1];

            #if the nodes does not coincide
            if node_i!=node_j:

                _pairNodes = [None, None]; #set of node pairs
                
                for node in self._HSlist:
                    if node_i in node:
                        _pairNodes[0] = self._HSlist.index(node); #for the ith node in the node of qubit kth, connect to the jth node in the node of qubit nth
                for node in self._HSlist:
                    if node_j in node:
                        _pairNodes[1] = self._HSlist.index(node);#for the jth node in the node of qubit kth, connect to the ith node in the node of qubit nth
                
                if _pairNodes not in self.map:
                    if _pairNodes[0]!=_pairNodes[1]:
                        self.map.append(_pairNodes);

        Q.add_edges_from(self.map);
        
        # plt.figure();
        # nx.draw(Q, with_labels=True);

        self._graphMap = Q; #set graph of the qubit map

        for n in range(self.NrQReg):
            
            #positions of atoms
            _nqPosition =self.layout[n];

            #QRegister map
            _qMap = [];
            for edge in self.map:
                if n in edge:
                    if [n,n] != edge:
                        _qMap.append(edge);

            #connectivity
            _connectivity=[];
            for kqstates in self._HSlist[n]:

                _buf = [edge for edge in self.connectivity if kqstates in edge];
                
                try:
                    _buf.remove((kqstates, kqstates));
                except:
                    pass
                _connectivity.append(_buf);


            _nqDetails = {'Position': _nqPosition, 'qMap':_qMap, 'Connectivity': _connectivity};
            _nqLabel = 'q'+str(n);

            self.compileQRegister.update({_nqLabel: _nqDetails}); #store QRegister details/specs

        pos = {i: tuple(val) for i, val in enumerate(self.layout)};
        # pos = nx.set_node_attributes(Q, pos, 'pos');

        #extract node positions
        # pos = nx.get_node_attributes(Q, 'pos');
        # print(pos)
        x, y, z = zip(*pos.values());

        #define custom labels for the nodes
        labels =  {i: 'q'+str(i) for i in range(len(self.layout))};

        #create a list of labels for the nodes
        node_labels = [labels.get(node, '') for node in Q.nodes()];

        #extract edge endpoints and assign labels
        edge_x = [];edge_y = [];edge_z = [];
        edge_labels = [];

        for u, v, d in Q.edges(data=True):
            x0, y0, z0 = pos[u];
            x1, y1, z1 = pos[v];
            midpoint = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);  # calculate midpoint
            edge_x.extend([x0, x1, midpoint[0], None]);
            edge_y.extend([y0, y1, midpoint[1], None]);
            edge_z.extend([z0, z1, midpoint[2], None]);
            edge_labels.append(midpoint);  # add midpoint coordinates only

        # if plotON==True:
        #     # Create 3D plot with nodes and edges
        #     fig = go.Figure(data=[go.Scatter3d(
        #         x=x,
        #         y=y,
        #         z=z,
        #         mode='markers+text',
        #         marker=dict(size=10, color='dodgerblue'),
        #         text=node_labels
        #     ), go.Scatter3d(
        #         x=edge_x,
        #         y=edge_y,
        #         z=edge_z,
        #         mode='lines+text',
        #         line=dict(color='black', width=2),
        #         text=[f"({i},{j})" for (i, j) in Q.edges()],
        #         textposition='middle center',  # set text position to middle center
        #         textfont=dict(size=10, color='black'),
        #         hoverinfo='none'
        #     )])

        #     # Set plot layout
        #     fig.update_layout(margin=dict(l=20,
        #                                   r=20,
        #                                   b=20,
        #                                   t=20,
        #                                   pad=4),
        #                       width=300, 
        #                       height=300, 
        #                       title='3D QRegister graph')

        #     # Show the plot
        #     fig.show()

    def playSim(self, mode='free', solver='QuTiP-QME'):
        '''
            Play the simulation of the dynamics of the atomicQRegister() object and store the results in the attribute simRes. Using the solver:
            
                QuTiP-QME : Quantum master equation solver by QuTiP
            
        '''
        
        if solver=='QuTiP-QME':
            if self.__mode=='free':
                self.simRes = qt.mesolve(self.nHamiltonian, qt.ket2dm(self.initnState), self.times, c_ops=self.ncops, e_ops=self.nmops, options=self.simOpts)

            elif self.__mode=='control':
                # self.simRes = qt.mesolve(self.tnHamiltonian, qt.ket2dm(self.initnState), self.times, c_ops=self.ncops, e_ops=self.nmops, options=self.simOpts)

                if self.ncops==None:
                    self.simRes = qt.mesolve(self.tnHamiltonian, self.initnState, self.times, e_ops=self.nmops, options=self.simOpts);
                else:
                    self.simRes = qt.mesolve(self.tnHamiltonian, self.initnState, self.times, c_ops=self.ncops, e_ops=self.nmops, options=self.simOpts);
        if solver=='QuantumOptics-QME':
            pass

    def buildNBasis(self):

        _basis_set = list(itertools.product(*self.lstNrlevels));

        for basis in _basis_set:
            _basisString.append(lst2string(basis));
            _basis.append(nqString2nqKet(lst2string(basis), None, bitsdim=self.lstNrlevels));
    
    def _buildInteractingBasis(self, _psi):

        '''
            Build the basis (states and operators) for the interactions basis in the connectivity map, that represent the 
            interactions between Rydberg states. 
        '''

        _pairInteraction_idx = [];
        _interaction_ops = [[], [], []]; #[list_vdW, list_dd, list_selfInt]
        _interaction_sts = [];   

        for connection in self.connectivity: #at the full Hilbert space basis

            ri = connection[0]; #state i-th rydberg state
            rj = connection[1]; #state j-th rydberg state

            _idx_ij = []; #index pair list for the connection
            
            #build basis accordingly with the available interactions
            for k in range(len(self._HSlist)): #qbit index over the first rydberg             

                _PSI = _psi; #[qt.basis(atMod.Nrlevels, 0) for atMod in self._AMs]; #wavefunction structure of the multi-atomicModel

                if ri in self._HSlist[k]: #check if ri is in list of k-th qbit interacting state levels

                    _ri_ket = qt.basis(self.lstNrlevels[k], self._AMslevels[k].index(ri-sum(self.lstNrlevels[:k])));
                    _ri_ket_idx = k;
                    _PSI[k] = _ri_ket;

                    _idx_ij.append(_ri_ket_idx);

                    for m in range(k,len(self._HSlist)): #qbit index over the second rydberg

                        _PSI_m = copy.deepcopy(_PSI); #wavefunction structure of the multi-atomicModel for m-th qubit state

                        if k==m: #same bits of the register

                            if rj in self._HSlist[m]: #check if rj is in list of m-th qbit interacting state levels

                                _rj_ket = qt.basis(self.lstNrlevels[m], self._AMslevels[m].index(rj-sum(self.lstNrlevels[:m])));
                                _rj_ket_idx = m;
                                _PSI_m[m] = _rj_ket;

                                _interaction_sts.append(_PSI_m);

                                _idx_ij.append(_rj_ket_idx);
                            
                                _VIJ = qt.tensor(_PSI_m)*(qt.tensor(_PSI_m).dag());
                                _VIJ.dims = [[self.Nrlevels],[self.Nrlevels]];
                                _interaction_ops[2].append(_VIJ);
                        
                        else: #different bits of the register

                            if rj in self._HSlist[m]: #check if rj is in list of m-th interacting state levels

                                _rj_ket = qt.basis(self.lstNrlevels[m], self._AMslevels[m].index(rj-sum(self.lstNrlevels[:m])));
                                _rj_ket_idx = m;
                                _PSI_m[m] = _rj_ket;

                                _interaction_sts.append(_PSI_m);

                                _idx_ij.append(_rj_ket_idx);


                                li = self.dynParams[k]['rydbergstates']['l_values'][self._AMslevels[k].index(ri-sum(self.lstNrlevels[:k]))- (self._AMs[k].Nrlevels-len(self.dynParams[k]['rydbergstates']['l_values']))]; #l-value for ri
                                lj = self.dynParams[m]['rydbergstates']['l_values'][self._AMslevels[m].index(rj-sum(self.lstNrlevels[:m]))- (self._AMs[m].Nrlevels-len(self.dynParams[m]['rydbergstates']['l_values']))]; #l-value for rj

                                if li==lj: #if li=lj check V_{vdW} 

                                    _VIJ = qt.tensor(_PSI_m)*(qt.tensor(_PSI_m).dag());
                                    _VIJ.dims = [[self.Nrlevels],[self.Nrlevels]];
                                    _interaction_ops[0].append(_VIJ);

                                else: #or V_{d-d} otherwise

                                    if self.NrQReg==2:
                                        _PSI_m_dag = copy.deepcopy(_PSI_m)
                                        _VIJ = qt.tensor(_PSI_m)*(qt.tensor(_swapElements(_PSI_m_dag, _ri_ket_idx, _rj_ket_idx)).dag());
                                        _VIJ.dims = [[self.Nrlevels],[self.Nrlevels]];
                                        _interaction_ops[1].append(_VIJ);
                                    else:
                                        _combinations4C3 = eliminate_duplicates(generate_combinations(_PSI_m, _ri_ket_idx, _rj_ket_idx, self._AMslevels_ket));
                                        self._combinationsC3 = _combinations4C3.copy();
                                        _VIJ=0;
                                        for combi in _combinations4C3:
                                            _PPSI_m =copy.deepcopy(combi);
                                            _PPSI_m_dag = _swapElements(copy.deepcopy(_PPSI_m), _rj_ket_idx, _ri_ket_idx);
                                            _vij = qt.tensor(_PPSI_m)*(qt.tensor(_PPSI_m_dag).dag());
                                            _vij.dims = [[self.Nrlevels],[self.Nrlevels]];
                                            _VIJ+=_vij;
                                    _interaction_ops[1].append(_VIJ);


                                    # _VIJ=0;
                                    # for l in range(len(self._HSlist)):

                                    #     if l!=m and l!=k:
                                    #         for rk in self._AMs[l]._basis:
                                    #             _PPSI_m =copy.deepcopy(_PSI_m);
                                    #             _PPSI_m[l]=rk;
                                    #             _PPSI_m_dag = _swapElements(copy.deepcopy(_PPSI_m), _rj_ket_idx, _ri_ket_idx);
                                    #             _VIJ += qt.tensor(_PPSI_m)*(qt.tensor(_PPSI_m_dag).dag());

                                    # _VIJ.dims = [[self.Nrlevels],[self.Nrlevels]];
                                    # _interaction_ops[1].append(_VIJ);
                                
                                self._pairInteraction_lval.append([li, lj]);
                                _pairInteraction_idx.append(_idx_ij);

        self._intbasis = [_interaction_ops, _interaction_sts];
        self._pairInteraction_idx = _pairInteraction_idx;

        for i in self._pairInteraction_idx:
            append_to_list_if_not_exists(self._C3pairInteraction_idx, np.array(i).tolist())

    def buildNinitState(self):
        '''
            Construct the initial state for the N atomicModel() that constitute the atomicQRegister() and store it in the attribute initnState. Either 
            from a given 
        '''
        if self.initnState is None:
            for elements in self.lstinitState:
                if isinstance(elements, qt.Qobj):
                    self.initnState = qt.tensor(self.lstinitState);
            
                elif isinstance(elements, int):
                    self.initnState = qt.tensor([qt.basis(self.lstNrlevels[i],  self.lstinitState[i]) for i in range(len(self.lstinitState))]) 
            
            self.initnState.dims= [[self.Nrlevels],[1]];
            self.initnState.reshape = (self.Nrlevels, 1);

        elif isinstance(self.lstinitState, str):
            self.initnState = nqString2nqKet(self.lstinitState, bitsdim=[AM.Nrlevels for AM in self._AMs])

            self.initnState.dims= [[self.Nrlevels],[1]];
            self.initnState.reshape = (self.Nrlevels, 1);
    
    def buildNHamiltonian(self):
        '''
            Construct the Hamiltonian of the N atomicModel() that constitute the atomicQRegister() and store it in the attribute nHamiltonian.
        [iden((2,2)), ...]
        '''
        _idenLst = [iden(self._AMs[i].Nrlevels) for i in range(len(self._AMs))];
        _buf = [];
        
        for i in range(len(self._AMs)):
            # _bufHamLst = _idenLst.copy(); #copy identity ops list
            _bufHamLst = copy.deepcopy(_idenLst); #copy identity ops list
            _bufHamLst[i] = self.lstHamiltonian[i] #substitute ith iden by ith Hamiltonian
            
            _buf.append(qt.tensor(_bufHamLst)); #save tensor product in buffer list
            
        self.nHamiltonian = sum(_buf);
        self.nHamiltonian.dims= [[self.Nrlevels],[self.Nrlevels]];
        self.nHamiltonian.reshape= (self.Nrlevels, self.Nrlevels);
    
    def buildTNHamiltonian(self):
        '''
            Construct the time-dependt Hamiltonian of the N atomicModel() taht constitute the atomicQRegister() and 
        '''
        self.__mode = 'control';
        
        _bufHQobjEvo = []; #list of storing all the t-dependent Hamiltonian of the register
        _bufnHAQiPTpulses = [AM.Hpulses for AM in self._AMs]; #list of pulses for the atomicModel()
        _bufnHStruct = [AM._lstHamiltonian for AM in self._AMs]; #list of the Hamiltonian's structure of the system
        
        _i=0;    
        for register in range(self.NrQReg):
            
            for H,oft in zip(_bufnHStruct[register],_bufnHAQiPTpulses[register]):

                #list of partial partition for storing as QobjEVO 
                nH = [iden(self._AMs[i].Nrlevels) for i in range(len(self._AMs))];
                nH[_i] = H;
                nH = qt.tensor(nH);
                nH.dims= [[self.Nrlevels],[self.Nrlevels]];

                _bufHQobjEvo+= [[nH, oft]]; #buffer list with struct and pulses of the Hamiltonian
                
                self._lsttHamiltonian+=[H]; #storing all atomicModel's Hamiltonian
            _i+=1;              


        self.tnHamiltonian = _bufHQobjEvo;
        self.Hpulses = _bufnHAQiPTpulses;

    def buildNLindbladians(self):
        '''
            Construct the Lindbladians for the N atomicModel() that constitute the atomicQRegister() and store it in the attribute ncops.
        '''
        _idenLst = [iden(self._AMs[i].Nrlevels) for i in range(len(self._AMs))];
        _buf = [];
        for i in range(len(self.lstcops)):
            _bufLindLst = _idenLst.copy(); #copy identity ops list
            _bufLindLst[i] = sum(self.lstcops[i]); #substitute i-th iden by i-th Lindbladian (sum of all cops of the AM)
            _buf.append(qt.tensor(_bufLindLst)); #save tensor product in buffer
        self.ncops = sum(_buf);
        self.ncops.dims= [[self.Nrlevels],[self.Nrlevels]];
        self.ncops.reshape= (self.Nrlevels, self.Nrlevels);
                    
    def buildNObservables(self, observables=None):
        '''
            Construct the Observables for the N atomicModel() that constitute the atomicQRegister() and store it in the attribute nmops.
        '''
        if observables==None:
            self.nmops, self._levels = obs(1, self.Nrlevels);
        elif isinstance(observables, list):
            self.nmops = observables;
    
    def add2QRegister(self, Nrlevels, psi0, params, name, AM=None):
        '''
            Add a new atomicModel() that constitute the atomicQRegister().
        '''        
        if AM == None:
            _newModel = atomicModel(self.times, Nrlevels, psi0, params, name)
            self._AMs.append(_newModel)
        else:
            self._AMs+=[AM]
    
    def getNHamiltonian(self, Hamiltonian_format='qutip'):
        '''
            Return the Hamiltonian for the N atomicModel() that constitute the atomicQRegister().
        '''

        if Hamiltonian_format=='qutip':
            return self.nHamiltonian;
        if Hamiltonian_format=='numpy':
            return self.nHamiltonian.data.todense();

    def getNLindbladian(self):
        '''
            Return the Lindbladian for the N atomicModel() that constitute the atomicQRegister().
        '''
        return self.ncops
    
    def buildInteractions(self, c6=1, c3=1):

        if len(self._intbasis)==0:


            self._groundstatesAM = [[0] if not sublist else sublist for sublist in self._groundstatesAM]; #replace empty sublists with a single element [0]

            
            _GScombinations = [p for p in itertools.product(*self._groundstatesAM)]; #generate all combinations of GS in NrQReg elements from the sublists
            # _GScombinations = [c for c in _GScombinations if len(set([id(sublist) for sublist in c])) == self.NrQReg]; #filter out combinations with duplicate elements from the same sublist

            _PSI_LST = [[qt.basis(atMod.Nrlevels, gs) for atMod, gs in zip(self._AMs, _GScombinations[comb_idx])] for comb_idx in range(len(_GScombinations))]; #iterate over all possible combinations of GS
             
            for psi_idx in range(len(_PSI_LST)):
                self._buildInteractingBasis(_psi = _PSI_LST[psi_idx]);

                _Vtot=None;

                #for C6 interactions
                for idx_basis in range(len(self._intbasis[0][0])):
                    self._getC6Strength(c6_val=c6, idx=idx_basis%self.NrQReg);
                    if isinstance(_Vtot, qt.Qobj):
                        _Vtot += self.nC6Interaction*self._intbasis[0][0][idx_basis];
                    else:
                        _Vtot = self.nC6Interaction*self._intbasis[0][0][idx_basis];
                try:
                    _intbasis4C3 = [];
                    for ii in self._intbasis[0][1]:
                        append_to_list_if_not_exists(_intbasis4C3, ii+ ii.dag());


                    for idx_basis in range(len(self._C3pairInteraction_idx)):

                        self._getC3Strength(c3_val=c3, idx=idx_basis); #%self.NrQReg

                        if isinstance(_Vtot, qt.Qobj):
                            _Vtot += self.nC3Interaction*_intbasis4C3[idx_basis];
                        else:
                            _Vtot = self.nC3Interaction*_intbasis4C3[idx_basis];
                except:
                    print('Passed C3 interaction. Not found.')

                try:
                    self.tnHamiltonian.append(_Vtot); #add the interaction term as always ON Hamiltonian
                    self.tnHamiltonian = self.tnHamiltonian[-1:] + self.tnHamiltonian[:-1]; #setting the new _Vtot term as first, for qutip solver requirement
                    self._Vint = _Vtot;

                except:
                    self.nHamiltonian+= _Vtot; #add the interaction term as always ON Hamiltonian

    def _getC6Strength(self, c6_val=1, idx=None):

        '''
            Build van der Waals interaction operators

            Assign the interaction operator into the intalInteraction attribute of the atomicModel

            INPUTS:
            -------

            c6_val : value of the C6 coefficient

        '''
        self.nC6Interaction = c6_val/eucdist(self.layout[self._pairInteraction_idx[idx][0]][0], self.layout[self._pairInteraction_idx[idx][0]][1],
                                             self.layout[self._pairInteraction_idx[idx][1]][0], self.layout[self._pairInteraction_idx[idx][1]][1])**6;

    def _getC3Strength(self, c3_val=1, idx=None):

        '''
            Build Dipole-Dipole interactions operators

            Assign the interaction operator into the intalInteraction attribute of the atomicModel

            INPUTS:
            -------

            c3_val : value of the C3 coefficient

        '''
        # print('idx: ', idx)
        # print('xa: ', self.layout[self._C3pairInteraction_idx[idx][0]][0], 
        #     'ya: ', self.layout[self._C3pairInteraction_idx[idx][0]][1],
        #     'xb: ', self.layout[self._C3pairInteraction_idx[idx][1]][0], 
        #     'yb: ', self.layout[self._C3pairInteraction_idx[idx][1]][1])
        self.nC3Interaction = c3_val/eucdist(self.layout[self._C3pairInteraction_idx[idx][0]][0], self.layout[self._C3pairInteraction_idx[idx][0]][1],
                                             self.layout[self._C3pairInteraction_idx[idx][1]][0], self.layout[self._C3pairInteraction_idx[idx][1]][1])**3;

    def getNObservables(self):
        '''
            Return the Observables for the N atomicModel() that constitute the atomicQRegister().
        '''
        return self.nmops
    
    def getResult(self):
        '''
            Return the results for the N atomicModel() that constitute the atomicQRegister() after being simulated with playSim().
        '''
        return self.simRes
    
    def showHamiltonian(self, plot_mode='matplotlib', color_map='viridis'):

        if plot_mode == 'matplotlib':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5), tight_layout=True);

            ax1.imshow(np.real(self.nHamiltonian), cmap=color_map);
            ax1.set_title(r'Re($\mathcal{H}$)');
            
            ax2.imshow(np.imag(self.nHamiltonian), cmap=color_map);
            ax2.set_title(r'Im($\mathcal{H}$)');

        elif plot_mode == 'qutip':
            qt.hinton(self.nHamiltonian);

    def showResults(self, plot_mode='matplotlib', legendON=True, resultTitle=None, resultseq=None, resultlabel=None, figure_size=(10,6), figure=None, axis=None, report=False):
        '''
            Return Results for the N atomicModel() that constitute the atomicQRegister().
        '''

        if plot_mode=='plotly':
            if figure_size==(10,6):
                figure_size=(1650,500);
            resultseq = self.simRes
            if resultlabel == None:
                resultlabel = [lst2str(i) for i in list(itertools.product(*[range(AM.Nrlevels) for AM in self._AMs]))]
                self._basisString = resultlabel

            if figure==None and axis==None:
                fig = go.Figure()

                for i in range(len(resultseq.expect)):
                    if resultlabel != None:
                        fig.add_trace(go.Scatter(x=self.times, y=resultseq.expect[i], name=resultlabel[i]))
                    else:
                        fig.add_trace(go.Scatter(x=self.times, y=resultseq.expect[i], name=str(i)))

                if legendON:
                    fig.update_layout(showlegend=True)
                else:
                    fig.update_layout(showlegend=False)

                fig.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Population',
                    title=resultTitle if resultTitle != None else self._name,
                    width=figure_size[0],
                    height=figure_size[1]
                )
                return fig

            else:
                resultseq = self.simRes

                fig, axs = figure, axis

                colormap = plt.cm.nipy_spectral
                colors = [colormap(i) for i in np.linspace(0, 1, self.Nrlevels)]

                for i in range(len(resultseq.expect)):
                    if resultlabel != None:
                        axs.plot(self.times, resultseq.expect[i], label=resultlabel[i], color=colors[i])
                    else:
                        axs.plot(self.times, resultseq.expect[i], label=i, color=colors[i])

                if legendON:
                    axs.legend()
                axs.set_xlabel('Time', fontsize=18)
                axs.set_ylabel('Population', fontsize=18)

                if resultTitle!=None:
                    axs.set_title(resultTitle)
                else:
                    axs.set_title(self._name)
                
                return fig, axs
            fig.show();
        if plot_mode=='matplotlib':
            resultseq = self.simRes;
            if resultlabel == None:
                resultlabel = [lst2str(i) for i in list(itertools.product(*[range(AM.Nrlevels) for AM in self._AMs]))];
                self._basisString = resultlabel;
                
            if figure==None and axis==None:
                
                fig, axs = plt.subplots(figsize=figure_size);

                for i in range(len(resultseq.expect)):
                    if resultlabel != None:
                        axs.plot(self.times, resultseq.expect[i], label=resultlabel[i]);
                    else:
                        axs.plot(self.times, resultseq.expect[i], label=i);

                if legendON:
                    plt.legend();
                plt.xlabel('Time', fontsize=18);
                plt.ylabel('Population', fontsize=18);
                if resultTitle!=None:
                    plt.title(resultTitle);
                else:
                    plt.title(self._name);

                return fig, axs

            else:
                resultseq = self.simRes
                
                fig, axs = figure, axis;

                colormap = plt.cm.nipy_spectral; #I suggest to use nipy_spectral, Set1,Paired
                axs.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 1, self.Nrlevels)]);

                for i in range(len(resultseq.expect)):
                    if resultlabel != None:
                        axs.plot(self.times, resultseq.expect[i], label=resultlabel[i]);
                    else:
                        axs.plot(self.times, resultseq.expect[i], label=i);

                if legendON:        
                    plt.legend();
                plt.xlabel('Time', fontsize=18);
                plt.ylabel('Population', fontsize=18);

                if resultTitle!=None:
                    plt.title(resultTitle);
                else:
                    plt.title(self._name);


                return fig, axs

    def registerMap(self, plotON=True, figure_size=(8,8)):
        '''
            Return the plot of the map for the N atomicModel() that constitute the atomicQRegister().
        '''
        plt.figure(figsize=figure_size);
        
        self._graphscolors = list(self._AMs[0]._graph['colormap']);
        
        G = self._AMs[0]._graph['graph'];
        
        for i in range(1,len(self._AMs)):
            AM = self._AMs[i];
            self._graphscolors+= AM._graph['colormap'];
            G = nx.disjoint_union(G, AM._graph['graph']);

        if self.connectivity[0]=='All':
            rydberg_edges = list(itertools.product(self._rydbergstates, self._rydbergstates));
            # rydberg_edges.pop() #eliminates the last self-interaction term (n,n) from the connectivity map
        else:
            rydberg_edges = self.connectivity;

        G.add_edges_from(rydberg_edges);

        nx.draw(G, with_labels=True, node_color=self._graphscolors);
        
        self._graphRegister = G;
        self.connectivity = rydberg_edges;

        if plotON==True:
            print('Violet nodes: Rydberg states. Blue nodes: Ground states')
        return self._graphRegister;

#####################################################################################################
#Scans-functions for atomicModels (AQiPT)
#####################################################################################################
class scan(object):

    def __init__(self, 
                 scan_dictionary, 
                 params, 
                 times, 
                 Nrlevels, 
                 psi0, 
                 name='Default scan name', 
                 atomic_model=None, 
                 population_idx='all',
                 simulation_mode='free'):

        self.dictionary = scan_dictionary;
        self.parameters = params;
        self.times = times;
        self.Nrlevels = Nrlevels;
        self.psi0 = psi0;
        self.name = name;
        self.atomicmodel = atomic_model;
        self.population_idx = population_idx;

        self.simulation_mode = simulation_mode;
        self.last_scan = [];
        self.executed_scans = [];


    def _get_scanValues(self, scan, params):    

        _VARs = [];
        for variable in scan['variables']:
            for subvariable in params[variable]:
                if isinstance(params[variable][subvariable][1], np.ndarray):
                    _VARs.append([variable, subvariable, params[variable][subvariable][1]])
                    
        return _VARs

    def _update_params(self, scanNr1, scan_idx, scanNr2, pseudofix_idx, params, scanVariables):
        
        _bufParams = params.copy();
            
        Variable1, Subvariable1, value1 = scanVariables[scanNr1]
        Variable2, Subvariable2, value2 = scanVariables[scanNr2]
        
        _bufParams[Variable1][Subvariable1][1] = value1[scan_idx];
        _bufParams[Variable2][Subvariable2][1] = value2[pseudofix_idx];
            
        return _bufParams

    def _scan_i_disordered(self, scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues, times, Nrlevels, psi0, name, AM, result_idx, results_list, simMode='free'):
        
        if AM is None:

            params = self._update_params(scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues); #updating params

            AM = atomicModel(times, Nrlevels, psi0, params, name = 'Ensemble qubit made of 3-lvl'); #build atomicModel class-object
            if simMode=='free':
                AM.buildHamiltonian(); #building Hamiltonians
            elif simMode=='control':
                AM.buildTHamiltonian(); #building Hamiltonians
            AM.buildLindbladians(); #building Lindbladians
            AM.buildObservables(); #building Observables   
            
        
        else:
            params = self._update_params(scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues); #updating params
            
            AM.dynParams = params; #update new params in atomicModel() object
            if any('couplings' in element for element in scanValues) or any('detunings' in element for element in scanValues):
                if simMode=='free':
                    AM.buildHamiltonian(); #building Hamiltonians
                elif simMode=='control':
                    AM.buildTHamiltonian(); #building Hamiltonians
            if any('dissipators' in element for element in scanValues):
                AM.buildLindbladians(); #building Lindbladians
                
        if simMode=='free':
            AM.playSim(mode=simMode); #playing simulation
        elif simMode=='control':
            AM.playSim(mode=simMode); #playing simulation

        if result_idx==0:
            results_list.append(AM.getResult().expect[0][len(times)-1]); #returning last value of simulation

        elif result_idx==1:
            results_list.append(AM.getResult().expect[1][len(times)-1]); #returning last value of simulation

        elif result_idx==2:
            results_list.append(AM.getResult().expect[2][len(times)-1]); #returning last value of simulation

        elif result_idx==3:
            results_list.append(AM.getResult().expect[3][len(times)-1]); #returning last value of simulation

        else:
            results_list.append([AM.getResult().expect[k] for k in range(Nrlevels)]); #returning last value of simulation
                     
    def RunScans_disordered(self):
        
        _scanValues = self._get_scanValues(self.dictionary, self.parameters);
        
        _jobs=[];
        _jobs_manager = multiprocessing.Manager();
        _results_list = _jobs_manager.list();
        
        for idx1 in range(len(_scanValues[0][2])): 
            for idx2 in range(len(_scanValues[1][2])): 
                if self.atomicmodel==None:
                    _process = multiprocessing.Process(target=self._scan_i_disordered,
                                                       args=(0, idx1, 
                                                             1, idx2, 
                                                             self.parameters, 
                                                             _scanValues, 
                                                             self.times, 
                                                             self.Nrlevels, 
                                                             self.psi0, 
                                                             self.name, 
                                                             self.atomicmodel, 
                                                             self.population_idx, 
                                                             _results_list, 
                                                             self.simulation_mode));
                    _jobs.append(_process);

                else:
                    _process = multiprocessing.Process(target=self._scan_i_disordered,
                                                       args=(0, idx1, 
                                                             1, idx2, 
                                                             self.parameters, 
                                                             _scanValues, 
                                                             self.times, 
                                                             self.Nrlevels, 
                                                             self.psi0, 
                                                             self.name, 
                                                             self.atomicmodel,  
                                                             self.population_idx, 
                                                             _results_list, 
                                                             self.simulation_mode));
                    _jobs.append(_process);

        for _job in _jobs:
            _job.start();

        for _job in _jobs:
            _job.join();
        
        self.last_scan = list(_results_list);
        self.executed_scans.append(self.last_scan);

    def _scan_i(self, scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues, times, Nrlevels, psi0, name, AM, result_idx, results_list, simMode='free'):
        
        if AM is None:

            params = self._update_params(scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues); #updating params

            AM = atomicModel(times, Nrlevels, psi0, params, name = 'Ensemble qubit made of 3-lvl'); #build atomicModel class-object
            if simMode=='free':
                AM.buildHamiltonian(); #building Hamiltonians
            elif simMode=='control':
                AM.buildTHamiltonian(); #building Hamiltonians
            AM.buildLindbladians(); #building Lindbladians
            AM.buildObservables(); #building Observables   
            
        
        else:
            params = self._update_params(scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues); #updating params
            
            AM.dynParams = params; #update new params in atomicModel() object
            if any('couplings' in element for element in scanValues) or any('detunings' in element for element in scanValues):
                if simMode=='free':
                    AM.buildHamiltonian(); #building Hamiltonians
                elif simMode=='control':
                    AM.buildTHamiltonian(); #building Hamiltonians
            if any('dissipators' in element for element in scanValues):
                AM.buildLindbladians(); #building Lindbladians
                
        if simMode=='free':
            AM.playSim(mode=simMode); #playing simulation
        elif simMode=='control':
            AM.playSim(mode=simMode); #playing simulation

        if result_idx==0:
            result = AM.getResult().expect[0][len(times)-1]; #returning last value of simulation

        elif result_idx==1:
            result = AM.getResult().expect[1][len(times)-1]; #returning last value of simulation

        elif result_idx==2:
            result = AM.getResult().expect[2][len(times)-1]; #returning last value of simulation

        elif result_idx==3:
            result = AM.getResult().expect[3][len(times)-1]; #returning last value of simulation

        else:
            result = [AM.getResult().expect[k] for k in range(Nrlevels)]; #returning last value of simulation

        return result

    def _scan_i_parallel(self, args):
        # Unpack the arguments
        scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues, times, Nrlevels, psi0, name, AM, population_idx, results_list, simulation_mode  = args

        # Call the scan_i function
        result = self._scan_i(scanNr1, scan_idx1, scanNr2, scan_idx2, params, scanValues, times, Nrlevels, psi0, name, AM, population_idx, results_list, simulation_mode)

        return result

    def RunScans(self, CPU_cores=12):
        _scanValues = self._get_scanValues(self.dictionary, self.parameters);

        pool = Pool(CPU_cores);        # Create a pool of processes

        jobs = [];
        for idx1 in range(len(_scanValues[0][2])):
            for idx2 in range(len(_scanValues[1][2])):
                args = (0, idx1, 
                        1, idx2, 
                        self.parameters, 
                        _scanValues, 
                        self.times, 
                        self.Nrlevels, 
                        self.psi0, 
                        self.name, 
                        self.atomicmodel, 
                        self.population_idx,
                        [],
                        self.simulation_mode);
                job = pool.apply_async(self._scan_i_parallel, (args,));
                jobs.append(job);

        _results_list = [job.get() for job in jobs];  # Retrieve the results in the order they were submitted

        pool.close();
        pool.join();

        self.last_scan = list(_results_list);
        self.executed_scans.append(self.last_scan);

    def getLastScan(self):
        return self.last_scan

    def plotScanResults(self, time_value, states_of_interest:list, variable:list):
        _x_axis = self.parameters[variable[0]][variable[1]][1];

        plt.figure();
        for _state in range(self.Nrlevels):
            plt.plot(_x_axis, [self.last_scan[i][_state][int(aqipt.time2index(time_value, self.times))] for i in range(len(_x_axis))], label=str(_state));
        plt.legend();

        plt.figure()
        for _state in states_of_interest:
            plt.plot(_x_axis, [self.last_scan[i][_state][int(aqipt.time2index(time_value, self.times))] for i in range(len(_x_axis))], label=str(_state));
        plt.legend();
        plt.show()

        return

#####################################################################################################
#Producer AQiPT class
#####################################################################################################
class producer:
    '''
        A class for creating the map of the director channels+instructions(waveforms) into Hamiltonian
        formalism for implement realistic atomicModels() & atomicQRegister().


        Parameters
        ----------
        _director : aqipt.control.director
            AQiPT director instance containing specificiations
        name : str
            Name of the producer
        _actors_lst : list
            List of actors within the producer


        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.

    '''
    def __init__(self, name, director=None, atom_specs=None, HW_specs=None, SW_specs=None):

        #atributes
        if HW_specs==None or SW_specs==None or atom_specs==None:
            self._director = director;

            self._HW_specs = self._director._HW_specs;
            self._SW_specs = self._director._SW_specs;
            self._atom_specs = self._director._atom_specs;

        elif director==None:
            self._director = None;

            self._HW_specs = HW_specs;
            self._SW_specs = SW_specs;
            self._atom_specs = atom_specs;

        self.name = name;
        self._actors_lst = [];
        self._compiledActors = False;

        try:
            #getting sampling rate
            _sampling_list=[];
            for k_instruction in range(len(self._SW_specs.specifications['instructions'])):
                try:
                    _sampling_list.append(self._SW_specs.specifications['instructions'][k_instruction][list(self._SW_specs.specifications['instructions'][k_instruction].keys())[0]]['SPECS']['args']['sampling']);
                except:
                    pass
            #getting dynamic time
            _max_t = []; _min_t = [];
            for k_instruction in range(len(self._SW_specs.specifications['instructions'])):
                
                try:
                    _max_t.append(max(self._SW_specs.specifications['instructions'][k_instruction][list(self._SW_specs.specifications['instructions'][k_instruction].keys())[0]]['SPECS']['args']['time_range']))
                    _min_t.append(min(self._SW_specs.specifications['instructions'][k_instruction][list(self._SW_specs.specifications['instructions'][k_instruction].keys())[0]]['SPECS']['args']['time_range']))
                except:
                    _max_t.append(self._SW_specs.specifications['instructions'][k_instruction][list(self._SW_specs.specifications['instructions'][k_instruction].keys())[0]]['SPECS']['args']['analog_args']['tp_window'])
            self._general_params = aqipt.general_params({'sampling': max(_max_t)*max(_sampling_list),
                                                         'bitdepth': 16,
                                                         'time_dyn': max(_max_t)});
        except:
            self._general_params=None;

        if isinstance(atom_specs, kernel.atomSpecs):
            self._atom = self._atom_specs.atom;
        else:
            self._atom = atom_specs;
        self.atom_energies = None;

        try:
            self._atomicmodel_args = {'tbase': self._general_params.timebase(),
                                      'Nrlevels': None,
                                      'psi0': None,
                                      'params4AM': None,
                                      };
        except:
            self._atomicmodel_args = None;

        self._atomicmodel = None;
        self._atomicmodelSchedule = None;
        self._atomicqregister = None;
        self._atomicqregisterSchedule = None;
        self._simMode = None;

    def addActors(self, actorName:str, deviceLst:list, roleDict:dict={}, actorSequence:str=None, actorType:str=None, slaveActor:bool=True, actorStatus:bool=False):
        
        _id_mem_alloc_lst = [aqipt.DFsearch(pd.DataFrame(self._HW_specs._IDs_bench), device)['python_memory_alloc'].values[0] for device in deviceLst];
        _dev_in_memory = [ctypes.cast( int(_memory_id, 16), ctypes.py_object).value for _memory_id in _id_mem_alloc_lst];


        _buf_roleDict = copy.copy(roleDict); #copy of dictionary role
        _role = None;
        _lst_model_elements = [None, None, None];


        for key in _buf_roleDict:
           
            if key == 'role':
                _role = _buf_roleDict['role'];

            elif key == 'args':

                #complete args from atom spec
                if len(_buf_roleDict['args'])>1:

                    if _buf_roleDict['args'][2] is None:
                        pass #something happen
                    else:
                        pass

                    if _buf_roleDict['args'][1] is None:
                        pass #add the coupling or detuning from atom spec
                    else:
                        pass

                    if _buf_roleDict['args'][0] is None:
                        pass #operator is added from atom spec

            else:
                _role = None;

                _lst_model_elements[0] = 'operator';
                _lst_model_elements[1] = 'amplitude';
                _lst_model_elements[2] = 'f(t)'; #mixing the signals accordingly with the device nature

                #constructing the time dependency for the simulation
                #build the device-waveform mapper for organizing the waveforms for each sequence corresponding to the different devices that match with the list actor's devices list
                _dev_waveform_map={}
                for _aqipt_sequence in self._SW_specs._python_waveforms:
                    _aqipt_waveform_idx=0;
                    for _aqipt_waveform in _aqipt_sequence._API_sequence:

                        if _aqipt_waveform[1]['identifier'] in deviceLst:


                            _dataframe = pd.DataFrame(self._HW_specs._IDs_bench);

                            _element_hex_id = aqipt.DFsearch(_dataframe, _aqipt_waveform[1]['identifier'])['python_memory_alloc'][aqipt.DFsearch(_dataframe, _aqipt_waveform[1]['identifier'])['python_memory_alloc'].index[0]];
                            _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value;

                            # print(_aqipt_waveform[1]['identifier'], deviceLst, _element_python.__class__.__name__)
                            if _aqipt_waveform_idx==0:
                                _dev_waveform_map[str(_aqipt_sequence.label)] = [{str(_aqipt_waveform[1]['identifier']): _aqipt_sequence.digiWaveformStack[_aqipt_waveform_idx]}, _element_python.__class__.__name__];
                            else:
                                _dev_waveform_map[str(_aqipt_sequence.label)].append([{str(_aqipt_waveform[1]['identifier']): _aqipt_sequence.digiWaveformStack[_aqipt_waveform_idx]},_element_python.__class__.__name__]);
                        _aqipt_waveform_idx+=1;

                roleDict['args']=_lst_model_elements;            

        if _role==None:

            if aqipt.check_elements_equal(_buf_roleDict['args'][0]):
                _role = 'detuning';
            else:
                _role = 'coupling';

            if aqipt.check_dict_template(_buf_roleDict['args'][0], {'atom':None, 'bfield':None, 'state_lst':None}):
                _role = 'zeeman-splitting';

        # _lst_model_elements = [None, None, None];
        # for key in _buf_roleDict:
        #     if key == 'args':

        #         if _buf_roleDict['args'][2] is None:
        #             pass #something happen
        #         else:
        #             pass

        #         if _buf_roleDict['args'][1] is None:
        #             pass #something happen
        #         else:
        #             pass

        #         if _buf_roleDict['args'][0] is None:
        #             pass #something happen
        #         else:
        #             pass
        #     else:

        #         _lst_model_elements[0] = 'operator';
        #         _lst_model_elements[1] = 'amplitude';
        #         _lst_model_elements[2] = 'f(t)';

        #         roleDict['args']=_lst_model_elements;

        _new_actor = actor(name=actorName, kind=actorType, devices=deviceLst, role=_role, role_args=roleDict['args'], slave=slaveActor, statusMode=actorStatus, hw_specs=self._HW_specs);
        self._actors_lst.append(_new_actor);

    def _compileActors(self):
        
        if self._compiledActors == False:
            for _actor in self._actors_lst:
                _actor.compile();
        else:
            pass
        self._atomicmodel_args['Nrlevels']= 1 + max([ max([max(j_args[0]) for j_args in k_actor.role_args]) for k_actor in self._actors_lst if k_actor.kind in ['optical', 'microwave', 'rf']]); #+1 to start in 0

    def _getActors(selfself):
        return self._actors_lst

    def _getActorNames(selfself):
        return [actors_.name for actors_ in self._actors_lst]

    def _getModel(self):
        return self._model

    def getName(self):
        return self.name
    
    def compile(self, psi0=0, t_Hamiltonian=True, simulation_time=None):
        
        if self._compiledActors==False:
            self._compileActors();
            self._compiledActors==True;
        
        #intial state for AM
        self._atomicmodel_args['psi0']=psi0;

        #simulation time
        if isinstance(simulation_time, np.ndarray):
            self._atomicmodel_args['tbase']=simulation_time;

        #parameters for AM
        _AM_couplings = {};
        _AM_detunings = {};
        _AM_dissipators = {};
        
        _AM_dissipators = {'Dissipator0': [[0,13], 6],
                             'Dissipator1': [[0,14], 6],
                             'Dissipator2': [[0,15], 6],
                             'Dissipator3': [[1,12], 6],
                             'Dissipator4': [[1,13], 6],
                             'Dissipator5': [[1,14], 6],
                             'Dissipator6': [[2,11], 6],
                             'Dissipator7': [[2,12], 6],
                             'Dissipator8': [[2,13], 6],
                             'Dissipator9': [[7,8], 6],
                             'Dissipator10': [[6,8], 6],
                             'Dissipator11': [[6,9], 6],
                             'Dissipator12': [[5,8], 6],
                             'Dissipator13': [[5,9], 6],
                             'Dissipator14': [[5,10], 6],
                             'Dissipator15': [[4,9], 6],
                             'Dissipator16': [[4,10], 6],
                             'Dissipator17': [[3,10], 6],
                             'Dissipator18': [[0,8], 6],
                             'Dissipator19': [[1,8], 6],
                             'Dissipator20': [[0,9], 6],
        #                      'Dissipator21': [[1,9], 6],
                             'Dissipator21': [[2,9], 6],
                             'Dissipator22': [[1,10], 6],
                             'Dissipator23': [[2,10], 6],
                             'Dissipator24': [[6,15], 6],
                             'Dissipator25': [[7,15], 6],
                             'Dissipator26': [[5,14], 6],
                             'Dissipator27': [[6,14], 6],
                             'Dissipator28': [[7,14], 6],
                             'Dissipator29': [[4,13], 6],
        #                      'Dissipator31': [[5,13], 6],
                             'Dissipator30': [[6,13], 6],
                             'Dissipator31': [[3,12], 6],
                             'Dissipator32': [[4,12], 6],
                             'Dissipator33': [[5,12], 6],
                             'Dissipator34': [[3,11], 6],
                             'Dissipator35': [[4,11], 6]}; 
        _AM_rydbergs = {'RydbergStates': [], 'l_values':[]};

        if t_Hamiltonian==False:
            self._simMode = 'free';
        else:
            self._simMode = 'control';

        _coupling_idx=0; _detuning_idx=0;
        for k_actor in self._actors_lst:
             if not isinstance(k_actor.field_action, str):
                    
                for k,v in k_actor.field_action.items():
                    
                    if "coupling" in k:
                        _AM_couplings.update({'Coupling'+str(_coupling_idx): v});
                        _coupling_idx+=1;

                    elif "detuning" in k:
                        _AM_detunings.update({'Detuning'+str(_detuning_idx): v});
                        _detuning_idx+=1;

        if 'detuning' not in [_actor.role for _actor in self._actors_lst]:

            if t_Hamiltonian==False:
                for _state_idx  in range(self._atomicmodel_args['Nrlevels']):
                    _AM_detunings.update({'Detuning'+str(_detuning_idx): [[_state_idx, _state_idx], 0, None]});
                    _detuning_idx+=1;

            elif t_Hamiltonian==True:
                for _state_idx  in range(self._atomicmodel_args['Nrlevels']):
                    _AM_detunings.update({'Detuning'+str(_detuning_idx): [[_state_idx, _state_idx], 0, np.ones(len(self._atomicmodel_args['tbase']))]});
                    _detuning_idx+=1;

        self._atomicmodel_args['params4AM'] = {'couplings': _AM_couplings, 
                                               'detunings': _AM_detunings, 
                                               'dissipators': _AM_dissipators,
                                               'rydbergstates': _AM_rydbergs}; 

        self._atomicmodel = atomicModel(self._atomicmodel_args['tbase'], 
                                        self._atomicmodel_args['Nrlevels'], 
                                        self._atomicmodel_args['psi0'], 
                                        self._atomicmodel_args['params4AM'], 
                                        name = 'atomicModel for {ProducerName}'.format(ProducerName=self.name));

        self._atomicmodel.modelMap(plotON=True, figure_size=(3,3));

        for _zeeman_actor in self._actors_lst:  
            if _zeeman_actor.role =='zeeman-splitting':
                self._atomicmodel.add_ZeemanSplitting(atom=self._atom, 
                                                      Bfield=_zeeman_actor.field_action_arguments['bfield'],
                                                      state_lst=_zeeman_actor.field_action_arguments['state_lst'],
                                                      buildTHamiltonian=t_Hamiltonian,
                                                      buildHamiltonian=not(t_Hamiltonian)); 
        self._atomicmodel.buildHamiltonian();       
        self._atomicmodel.buildLindbladians();
        self._atomicmodel.buildObservables();  

    def runSimulation(self, showResults=True, figure_size=(25,5), plot_mode='matplotlib'):

        self._atomicmodel.playSim(mode=self._simMode);     

        if showResults==True:  
            self._atomicmodel.showResults(figure_size=figure_size,  plot_mode=plot_mode);       

#####################################################################################################
#Actor AQiPT class
#####################################################################################################
class actor:
    '''
        A class for creating the map of the director channels+instructions(waveforms) into Hamiltonian
        formalism for implement realistic atomicModels() & atomicQRegister().


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        name : str
            Actor name
        type : str
            Type of actor
        _HW_lst : list
            Hardware list loaded in experiment and produced for actor
        _HWschedule : list
            Real hardware schedule of the actor HW list
        field : aqipt.emulator.field
            Field instance 
        _model : aqipt.emulator.atomicModel or aqipt.emulator.atomicQRegister
            Emulator theoretical model
        _modelSchedule : aqipt.control.sequence
            Waveform schedule for the emulator model

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.

    '''
    
    def __init__(self, name, kind=None, devices=None, role=None, role_args=None, slave=True, statusMode='active', hw_specs=None):

        #atributes
        self.name = name;
        self.kind = kind;
        self.slave_type = slave;
        self.status = statusMode;
        self._HW_specs = hw_specs;
        self._HW_lst = devices;
        self._HWschedule = None;
        self.role = role; #detuning, coupling, shift, zeeman-splitting, geometry

        if self.kind==None and self._HW_lst!=None:

            _kind_lst=[];
            for _identifier in self._HW_lst:
                _dataframe = pd.DataFrame(self._HW_specs._IDs_bench);

                _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
                _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value;

                _kind_lst.append(_element_python.__class__.__name__)

            if any(item in _kind_lst for item in ['Laser', 'AOM', 'LaserAUX', 'Shutter', 'DMD', 'SLM', 'AOD']):
                self.kind = 'optical';
            elif 'AWG' in _kind_lst and 'DDS' in _kind_lst and all(item not in _kind_lst for item in ['Coil', 'Electrode', 'Laser', 'AOM', 'LaserAUX', 'Shutter', 'DMD', 'SLM', 'AOD']):
                self.kind = 'microwave';
            elif 'AWG' in _kind_lst and 'Electrode' in _kind_lst and all(item not in _kind_lst for item in ['Coil', 'Antenna', 'Laser', 'AOM', 'LaserAUX', 'Shutter', 'DMD', 'SLM', 'AOD']):
                self.kind = 'rf';
            elif 'AWG' in _kind_lst and 'Electrode' in _kind_lst and all(item not in _kind_lst for item in ['Coil', 'Antenna', 'Laser', 'AOM', 'LaserAUX', 'Shutter', 'DMD', 'SLM', 'AOD']):
                self.kind = 'electric';
            elif 'AWG' in _kind_lst and 'Coil' in _kind_lst and all(item not in _kind_lst for item in ['Antenna', 'Electrode', 'Laser', 'AOM', 'LaserAUX', 'Shutter', 'DMD', 'SLM', 'AOD']):
                self.kind = 'magnetic';

        try:
            #choosing the right states for the model depending in the format
            self.role_args = [];
            if len(role_args[0])>2 or len(role_args[0])==1:
                for _pair_states in role_args[0]:
                     _copy_role_args = copy.deepcopy(role_args);
                     _copy_role_args[0]=_pair_states;
                     self.role_args.append(_copy_role_args);

            elif isinstance(role_args[0][0],list) and isinstance(role_args[0][1],int):
                _copy_role_args = copy.deepcopy(role_args);
                for state in role_args[0][0]:
                    _copy_role_args[0][0]=state;
                    self.role_args.append(copy.deepcopy(_copy_role_args));

            elif isinstance(role_args[0][1],list) and isinstance(role_args[0][0],int):
                _copy_role_args = copy.deepcopy(role_args);
                for state in role_args[0][1]:
                    _copy_role_args[0][1]=state;
                    self.role_args.append(copy.deepcopy(_copy_role_args));

            elif isinstance(role_args[0][0],list) and isinstance(role_args[0][1],list):
                _copy_role_args = copy.deepcopy(role_args);

                for state_i in role_args[0][0]:
                    _copy_role_args[0][0]=state_i;

                    for state_j in role_args[0][1]:
                        _copy_role_args[0][1]=state_j;

                        self.role_args.append(copy.deepcopy(_copy_role_args));
            else:
                    self.role_args = [role_args];
        except:
            self.role_args = role_args;


        self.field = None;
        self.field_action = {};
        self.field_action_arguments = None;

    def compile(self):
        
        #setting the simulation arguments for the different actors types

        if self.role=='detuning':
            _role_arg_idx=0;
            for _role_arg in self.role_args:
                self.field_action.update({'detuning'+str(_role_arg_idx): [_role_arg[0], _role_arg[1], _role_arg[2]]});
                self.field_action_arguments = {k: self.role_args[k] for k in range(len(self.role_args))};
                _role_arg_idx+=1;
                break
            
        elif self.role=='coupling':
            _role_arg_idx=0;
            for _role_arg in self.role_args:
                self.field_action.update({'coupling'+str(_role_arg_idx): [_role_arg[0], _role_arg[1], _role_arg[2]]});
                self.field_action_arguments = {k: self.role_args[k] for k in range(len(self.role_args))};
                _role_arg_idx+=1;
            
        elif self.role=='shift':
            _role_arg_idx=0;
            for _role_arg in self.role_args:
                self.field_action.update({'detuning'+str(_role_arg_idx): [_role_arg[0], _role_arg[1], _role_arg[2]]});
                self.field_action_arguments = {k: self.role_args[k] for k in range(len(self.role_args))};
                _role_arg_idx+=1;
                print("Not implemented (sum to diagonal terms)")
            
        elif self.role=='zeeman-splitting':
            _role_arg_idx=0;
            for _role_arg in self.role_args:
                self.field_action = 'add_ZeemanSplitting';
                self.field_action_arguments = {'atom':_role_arg['atom'], 
                                               'bfield':_role_arg['bfield'],  
                                               'state_lst':_role_arg['state_lst']};
                _role_arg_idx+=1;
                     
        elif self.role_args=='geometry':
            _role_arg_idx=0;
            for _role_arg in self.role_args:
                self.field_action = 'set-register';
                self.field_action_arguments = _role_arg['coordinates'];
                _role_arg_idx+=1;
                print("Not implemented (define spatial geometry)")
        
        #generating the field accordingly to the actors kind

        if self.kind == 'optical':
            self.field = optical(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);

        elif self.kind == 'acoustic':
            self.field = acoustic(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);

        elif self.kind == 'magnetic':
            self.field = magnetic(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);

        elif self.kind == 'microwave':
            self.field = microwave(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);

        elif self.kind == 'rf':
            self.field = rf(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);

        elif self.kind == 'electric':
            self.field = electric(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False); 

        elif self.kind == 'gravity':
            self.field = gravity(name=self.name, kind=self.kind, devices=self._HW_lst, action=self.field_action, actionArgs=self.field_action_arguments, slave=True, statusMode=False);
          
#####################################################################################################
#field AQiPT class
#####################################################################################################
class field:
    '''
        A class for creating the general instances of fields: optical, microwave, RF, magnetic
        and electric fields.


        Parameters
        ----------
        name : str
            Name of the field
        type : string
            Type of field: optical, acoustic, magnetic, microwave, rf, electric and gravity
        _space : array
            Space dependency
        _time : array
            Time dependency
        _frequency : float
            Frequency value
        _phase : float
            Phase value
        _polarization : array
            Polarization value
        _direction : array
            Propagation k-vector
        _wavelength : float
            Wavelength of the wave field
        _amplitude : array
            Amplitude modulation of the field

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.
        
    '''
    def __init__(self, name, kind, devices, action, slave=True, actionArgs=None, statusMode='active'):

        #atributes
        self.name = name; # name
        self.kind = kind; # type of field
        self._space = None; # space dependency
        self._time = None; # time dependency
        self._transition = action; # atomic transitions involved
        self._frequency = None; # frequency
        self._phase = None; # phase
        self._polarization = None; # polarization
        self._direction = None; # k-vector 
        self._wavelength = None; # wavelength
        self._amplitude = None; # amplitude modulation
        self.units = None; # units of the attributes
        self.states = None;


###===================###
### field sub-classes ###
###===================###

class optical(field):
    '''Optical field'''
    def __init__(self, **kwargs):
        super(optical, self).__init__(**kwargs);
        # self.arg = arg;

class acoustic(field):
    '''Acoustic field'''
    def __init__(self, **kwargs):
        super(acoustic, arg).__init__(**kwargs);
        # self.arg = arg;

class magnetic(field):
    '''Magnetic field'''
    def __init__(self, **kwargs):
        super(magnetic, self).__init__(**kwargs);
        # self.arg = arg;

class microwave(field):
    '''Microwave field'''
    def __init__(self, **kwargs):
        super(microwave, self).__init__(**kwargs);
        # self.arg = arg;

class rf(field):
    '''RF field'''
    def __init__(self, **kwargs):
        super(rf, self).__init__(**kwargs);
        # self.arg = arg;

class electric(field):
    '''Electric field'''
    def __init__(self, **kwargs):
        super(electric, self).__init__(**kwargs);
        # self.arg = arg;

class gravity(field):
    '''Microwave field'''
    def __init__(self, **kwargs):
        super(gravity, self).__init__(**kwargs);
        # self.arg = arg; 


#####################################################################################################
#optElement AQiPT class
#####################################################################################################

class optElement(object):
    
    def __init__(self, args):
        self.args=args
        self.type=None
        self.name=None
        self.label=None
        
    def get_matrix(cls):
        return cls.transferMatrix
    
    def get_name(self):
        return self.name

###========================###
### optElement sub-classes ###
###========================###

class medium(optElement):
    
    def __init__(self, args):
        optElement.__init__(self, args)
        self.type = 'Passive'
        self.name = 'Medium'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , args['distance']],
                                       [0 , 1]])    

class flatInterface(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Flat Interface'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , 0],
                                        [0 , args['n1']/args['n2']]])

class curvedInterface(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Curved Interface'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , 0],
                                        [((args['n1']-args['n2'])/(args['curvature_radius']*args['n2'])) , args['n1']/args['n2']]])    

class flatMirror(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Flat Mirror'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , 0],
                                        [0 , 1]])

class curvedMirror(optElement):
        
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Flat Interface'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , 0],
                                        [-1*(2/args['curvature_radius']) , args['n1']/args['n2']]])

class thinLens(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Flat Interface'
        self.label = args['label']
        self.transferMatrix = np.array([[1 , 0],
                                        [(-1/args['focal_lenght']) , 1]])

class thickLens(optElement):
        
    def __init__(self, args):
        A = np.array([[1 , 0],
                  [((args['n1']-args['n2'])/(args['curvature_radius']*args['n2'])) , args['n1']/args['n2']]])
        B = np.array([[1, args['thickness_center']], [0,1]])
        
        self.type = 'Passive'
        self.name = 'Thick Lens'
        self.label = args['label']
        self.transferMatrix = np.dot(A,np.dot(B,A))        

class prism(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Prism'
        self.label = args['label']
        self.transferMatrix = np.array([[args['beam_expansion'] , args['prism_path_length']/(args['n1']*args['beam_expansion'])],
                                    [0 , 1/args['beam_expansion']]])

class multiplePrism(optElement):
    
    def __init__(self, args):
        self.type = 'Passive'
        self.name = 'Prism'
        self.label = args['label']
        self.transferMatrix = np.array([[args['magnification'],args['total_opt_propagation']],
                                        [0, 1/args['magnification']]])

#####################################################################################################
#beam AQiPT class
#####################################################################################################

class beam(object):
    
    def __init__(self, amplitude, frequency, polarization, direction, phase=0, aligned=True, tSampling=np.linspace(0,1,100)):
        if aligned==True:
            self.tSamp=tSampling;
            self.amp=amplitude;
            self.freq=frequency;
            self.pol=polarization;
            self.dir=direction;
            self.ph=phase;
            self._Efield=self.amp*np.cos(self.freq*self.tSamp + self.ph)*self.pol;
            self._beamVector=[self._Efield, self.dir];
            self.path=np.array([[1,0],[0,1]]);
            self._elements=list();
        else:
            print('Spacial dependency of setup no supported yet. Sorry :(')
    
    @classmethod
    def create(cls, amplitude, frequency, polarization, direction, phase=0, aligned=True, tSampling=np.linspace(0,1,100)):
        return cls(amplitude, frequency, polarization, direction, phase=0, aligned=True, tSampling=np.linspace(0,1,100))
        
    def add2Path(self, optElmnt):
        if isinstance(optElmnt, object):
            self.path=np.dot(self.path,optElmnt.get_matrix())
            self._elements.append(optElmnt)
        if isinstance(optElmnt, list):
            for element in optElmnt:
                self.path=np.dot(self.path,element.get_matrix())
            self._elements+=optElmnt
    
    def get_beamVector(self):
        return self._beamVector
    
    def get_Efield(self):
        return self._Efield
    
    def get_pathMatrix(self):
        return self.path
    
    def get_elements(self):
        return self._elements

#####################################################################################################
#tweezer AQiPT class
#####################################################################################################

class tweezer(object):

    def __init__():
        pass

#####################################################################################################
#OptSetup AQiPT class
#####################################################################################################

class OptSetup(object):
    
    def __init__(self):
        self.name=None
        self._beams=list()
        self._elements=list()
    
    def add2Setup(self, newBeam):
        self._beams.append(newBeam)
        self._elements+= newBeam._elements
    
    def _addElements(self, elements):
        if isinstance(elements, optElement):
            self._elements.append(elements)
        if isinstance(elements, list):
            for element in elements:
                if isinstance(element, optElement):
                    self._elements.append(element)
    
    def getbeams(self):
        return self._beams
    
#     def playStatic():    
#     def playDynamic():