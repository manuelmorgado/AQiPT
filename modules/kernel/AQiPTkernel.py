#Atomic Quantum information Processing Tool (AQIPT) - DAQ module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2021-04-08
# Last update: 2022-10-07


#libs
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from functools import reduce
# import itertools
# from typing import Iterator, List
import copy

from tqdm import tqdm

from IPython.core.display import HTML, display

# from numba import jit
# import numba
import os, time, dataclasses
from datetime import date
from dataclasses import dataclass, field
import json
from typing import List


from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.directory import AQiPTdirectory as dirPath
from AQiPT.modules.control import AQiPTcontrol as control

import arc
from scipy.constants import physical_constants
from scipy.constants import e as C_e

import random

import pandas as pd

'''
	TO DO LIST
	----------

	    -Finish other properties of atoms such C3, Dip.Mat.Element
	    -Include other sourcers like PairStates python package
	    -Include other formats to export the data e.g., '.qg8'
	    -Add extra constrains in the extraParams for including specification of the lasers for the different transitions (at the moment all transitions use same power, polarization and waist.)
'''

# directory = dirPath.directory_tree({'path': 'C:\\Users\\EQM\\Desktop\\AQiPT_vNewPC_20230525\\AQiPT_vLaptop\\AQiPT\\modules\\directory\\',
#                                     'filename': 'directories_windows.json',
#                                     'printON': False})
directory = aqipt.directory;


#####################################################################################################
#atomSpecs AQiPT (data)class
#####################################################################################################

def _getFullStates(states, I):

    try:
        _states_is_dic = aqipt.is_dict_of_dicts(states);
    except:
        _states_is_dic = False;

    if _states_is_dic!=True:

        _full_states_dic={};_idx=0;
        for _state in states:
            
            for F in range(int(abs(_state[2]-I)),int(_state[2]+I+1)):
                
                for mF in range(int(-F),int(F+1)):
                    
                    _full_states_dic[_idx] = [_state[0], _state[1], _state[2], _state[3], F, mF];
                    _idx+=1;
    else:

        states = [list(_state.values()) for _state in states.values()];

        _full_states_dic={};_idx=0;
        for _state in states:
            
            for F in range(int(abs(_state[2]-I)),int(_state[2]+I+1)):
                
                for mF in range(int(-F),int(F+1)):
                    
                    _full_states_dic[_idx] = [_state[0], _state[1], _state[2], _state[3], F, mF];
                    _idx+=1;

    return _full_states_dic

def _getStates(states, I):

    states = _lst2dic(states);

    try:
        _states_is_dic = aqipt.is_dict_of_dicts(states);
    except:
        _states_is_dic = False;

    if _states_is_dic!=True:

        _states_dic={};_idx=0;
        for _state in states:
            
            _states_dic[_idx] = [_state[0], _state[1], _state[2], _state[3]];
            _idx+=1;
    else:

        states = [list(_state.values()) for _state in states.values()];

        _states_dic={};_idx=0;
        for _state in states:
            
            _states_dic[_idx] = [_state[0], _state[1], _state[2], _state[3]];
            _idx+=1;

    return _states_dic

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/');
        except ValueError:
            return None
        try:
            leading, num = num.split(' ');
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1;
        else:
            sign_mult = 1;
        return float(leading) + sign_mult * (float(num) / float(denom));

def orb2Int(orb):
    return [0 if orb =='S' 
            else 1 if orb == 'P' 
            else 2 if orb == 'D' 
            else 3 if orb == 'F' 
            else None][0]

def lst2dic(ket):
    
    if type(ket) is list:
        
        ket_lst={}; nr_state=0;
        
        for state in ket:

            l_val = ''.join([n for n in state if n.isupper()]);
            n_val = state.rpartition(l_val)[0];
            j_val = state[state.find('_')+len('_'):state.rfind(',')];
            s_val = state[state.find(',')+len(',')+1:state.rfind('')];
            try:
                f_val = state[state.find(',')+len(',')+2:state.rfind('')];
                mf_val = state[state.find(',')+len(',')+3:state.rfind('')];
                ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val), 'f':convert_to_float(f_val), 'mf':convert_to_float(mf_val)});
            except:
                ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)});
                nr_state+=1;
        return ket_lst

    else:
        
        l_val = ''.join([n for n in ket if n.isupper()]);
        n_val = ket.rpartition(l_val)[0];
        j_val = ket[ket.find('_')+len('_'):ket.rfind(',')];
        s_val = ket[ket.find(',')+len(',')+1:ket.rfind('')];
        try:
            f_val = state[state.find(',')+len(',')+2:state.rfind('')];
            mf_val = state[state.find(',')+len(',')+3:state.rfind('')];
            return {'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val), 'f':convert_to_float(f_val), 'mf':convert_to_float(mf_val)}
        except:
            return {'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)}
    
def atomicData_fromFile(fname, metadata, engine='pyarrow'):
    '''
        Load atomic data from file (format: .parquet)
        
        
        INPUTS:
        -------
        
            fname : (string) file name
            engine : (string) readout engine of the file format
        
        
        OUTPUTS:
        --------
        
            atomicData (dataclass)
        
    '''
    recovered_DF = pd.read_parquet(fname+'.parquet', engine=engine);
    lenght_DF = len(recovered_DF);
    
    _clabelLst = metadata['cell_labels']
    _cellLst=[];
    for k in range(lenght_DF):
        _recovered2dict = recovered_DF.iloc[k].to_dict()
        _cellLst.append( datacell(_recovered2dict['Energy'],_recovered2dict['Transition_freq'],_recovered2dict['Wavelength'],
                                  _recovered2dict['Rabi_freq'],_recovered2dict['Dip_Mat_element'],_recovered2dict['Lifetime'],
                                  _recovered2dict['Rydberg'],_recovered2dict['BBR'],_recovered2dict['C3'],
                                  _recovered2dict['C6'],_recovered2dict['Rbl'],_recovered2dict['LeRoy_radius'],
                                  _recovered2dict['Polarizability'],_recovered2dict['Stark_Map']) );
    return atomicData(_clabelLst, _cellLst)

def metadata_fromFile(fname, engine='pyarrow'):
    '''
        Load metadata from file (format: .parquet)
        
        INPUTS:
        -------
        
            fname : (string) file name
            engine : (string) readout engine of the file format
        
        
        OUTPUTS:
        --------
        
            metadata (dict)
        
    '''
    recovered_DF = pd.read_parquet('metadata_'+fname+'.parquet', engine=engine);
    return recovered_DF.to_dict(orient="index")[0]

def extract_SMapData(starkMap_obj):
    
    _buflst1=[]; _buflst2=[]; _buflst3=[];
    for i in range(len(starkMap_obj.y)):
        for j in range(len(starkMap_obj.y[int(i)])):
            _buflst1.append(starkMap_obj.eFieldList[int(i)]);
            _buflst2.append(starkMap_obj.y[int(i)][int(j)]);
            _buflst3.append(starkMap_obj.highlight[int(i)][int(j)]);
            
    neFieldList = np.array(_buflst1);
    ny = np.array(_buflst2);
    nyState = np.array(_buflst3);

    sortOrder = nyState.argsort(kind='heapsort');


    return neFieldList[sortOrder], ny[sortOrder], nyState[sortOrder];

def extract_BBRData(BBR_obj):
#     return {'n_values' : BBR_obj['valsn'], 'NoBBR_values' : BBR_obj['valsNoBBR'], 'BBR_values' : BBR_obj['valsBBR']}
    return {i:BBR_obj[i] for i in BBR_obj if i!='figure'}
    
class atomSpecs:
    
    '''
        Special class of AQiPT that pretty much works as a container of Python inmmutable dataclass 
        i.e., ```atomicData()``` built from more specific mutable dataclass ```datacell()``` that contains 
        specific information of the atoms, as well as properties obtained from ab-initio calculations and 
        data measured by the atomic physics community and wrapped within other Python packages.
        
        
        
        ATTRIBUTES:
        -----------
        
            atom :
            _atomName :
            states :
            _metadata :
            atomicdata :
            __atomicdata2save :
            _eParams :
        
        
        METHODS:
        --------
            getTransitionFrequency_fromARC :
            getTransitionWavelength_fromARC :
            getRabifreq_fromARC
            getSMap_fromARC
            getLifetime_fromARC
            getPolarizability_fromARC :           
            getLeRoyRadius_fromARC :
            getC6_fromARC :
            getRb_fromARC :
            getBBR_fromARC :
            
            set_atomicData :
            save :
    '''
    
    def __init__(self, atom=None, states=None, specs=None, metadata=None, extraParams=None, source='ARC'):

        self.atom = atom; 
        self._atomName = self.atom.elementName; 
        self.states = states;
        self.partialstate = _getStates(states, I=1.5);
        self.fullstates = _getFullStates(_lst2dic(self.states), self.atom.I);
        self._eParams = extraParams; 
        self._atomicdata2save = None;
        self._source = source;
        
        #given nothing - requires set_atomicData() for atomicData AQiPT-dataclass
        if specs is None: 
            self._metadata = metadata
            self.atomicdata = specs; #set_atomicData()
        
        #given the (specs,metadata) - requires an atomicData AQiPT-dataclass
        elif dataclasses.is_dataclass(specs):
            self._metadata = metadata;
            self.atomicdata = specs;
        
        #given (specs, metadata) files requires atomicData_fromFile() and metadata_fromFile() to set atomicData AQiPT-dataclass
        elif type(specs) is str and type(metadata) is str: 
            self._metadata = metadata_fromFile(metadata);
            self.atomicdata = atomicData_fromFile(specs);
    
    
    ###############################
    #### METHODS FOR DATACELLS ####
    ###############################
    
    def getTransitionFrequency_fromARC(self, istate, jstate):
        return  self.atom.getTransitionFrequency(istate['n'], istate['l'], istate['j'],
                                                 jstate['n'], jstate['l'], jstate['j'])*1e-12

    def getTransitionWavelength_fromARC(self, istate, jstate):
        return  self.atom.getTransitionWavelength(istate['n'], istate['l'], istate['j'],
                                                  jstate['n'], jstate['l'], jstate['j'])*1e9

    def getRabifreq_fromARC(self, istate, jstate, polarization, power, waist):
        return  self.atom.getRabiFrequency(istate['n'], istate['l'], istate['j'], istate['s'], jstate['n'], jstate['l'], jstate['j'], polarization, power, waist)*1e-6

    def getSMap_fromARC(self, istate, nrange, lmax, Bz, Erange, N, progressBar=True, unit=1, HighlightState=True, HighlightColour='red'):

        #create Stark map object
        sMap = arc.StarkMap(self.atom);

        sMap.defineBasis(istate['n'], istate['l'], istate['j'], istate['s'], nrange[0], nrange[1], lmax, Bz, progressOutput=progressBar, s=0.5); #initialise Basis States for Solver : progressOutput=True gives verbose output

        #generate Stark map
        sMap.diagonalise(np.linspace(Erange[0], Erange[1], N), progressOutput=progressBar);

        #show Stark map
        sMap.plotLevelDiagram(progressOutput=progressBar, units=unit, highlightState=HighlightState, highlightColour=HighlightColour);

        return sMap

    def getLifetime_fromARC(self, istate):
        return  self.atom.getStateLifetime(istate['n'], istate['l'], istate['j'])

    def isRydberg(self, istate):
        if istate['n']>20:
            return True
        else:
            return False

    def getBBR_fromARC(self, nMin, nMax, lj1, lj2, minTemp=0.1, maxTemp=300):

        vals = {'valsn': [], 'valsNoBBR': [], 'valsBBR': [], 'figure': None}

        for n in range(nMin,nMax):

            noBBR = self.atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1], temperature=minTemp)\
                    +self.atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1]+lj2[0], temperature=minTemp);
            withBBR =  self.atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1], temperature=maxTemp)\
                    +self.atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1]+lj2[0], temperature=maxTemp)
            vals['valsn'].append(n);
            vals['valsNoBBR'].append(noBBR);
            vals['valsBBR'].append(withBBR-noBBR);

        width = 0.4;

        figBBR, ax = plt.subplots(figsize=(10,7))
        vals['figure'] = [figBBR, ax];

        ax.bar(np.array(vals['valsn'])-width/2.,np.array(vals['valsNoBBR']),width=width,color="r", label="Spontaneous decays $n=$%.2i" %nMax);
        ax.bar(np.array(vals['valsn'])+width/2.,np.array(vals['valsBBR']),width=width,color="g", label="Black-body induced transitions $n=$%.2i" %nMax);


        ax.set_xlabel("Principal quantum number, $n$");
        ax.set_ylabel(r"Transition rate (s${}^{-1}$)");
        ax.set_title(r"Transition from $\{\ell,m\}$ $S_{1/2}$ to $n$ $P_{1/2,3/2}$");
        ax.legend(fontsize=10);
        ax.set_xlim(4,max(nMax,l));

        return vals

    def getC6_fromARC(self, sMap, phi, theta, dn, dEmax, plotON=False ):
        calc = arc.PairStateInteractions(self.atom, sMap.n, sMap.l, sMap.j, sMap.n, sMap.l, sMap.j, sMap.s, sMap.s)

        c6=[];

        for t in theta:
            C6 = calc.getC6perturbatively(t, phi, dn, dEmax)
            c6.append(abs(C6))

        if plotON==True:
            ax = plt.subplot(111, projection='polar');
            ax.set_theta_zero_location("N");
            line = [];

            # plot results
            lineLegend, = plt.plot(theta,c6,"-",color="r",label=("mj=%d/2"%int(2*sMap.s)) )
            line.append(lineLegend)
            plt.legend(handles=line,fontsize=10)

            return c6, ax

        else:

            return c6

    def getRb_fromARC(self, omega_lst, c6_lst):
        Rb_lst = [];
        for C6 in c6_lst:
            Rb_lst.append([((C6*1e6/omega)**(1/6)) for omega in omega_lst])
            
        return Rb_lst    

    def getLeRoyRadius_fromARC(self, sMap):
        calc = arc.PairStateInteractions(self.atom, sMap.n, sMap.l, sMap.j, sMap.n, sMap.l, sMap.j, sMap.s, sMap.s)
        return calc.getLeRoyRadius()

    def getPolarizability_fromARC(self, sMap, wavelength):

        polarizabilityList = []

        sP = sMap.getPolarizability( minStateContribution=0.9)
    #     print("%s\t%.3e\t\t\t\t%.2f-%.2f" % \
    #         (printStateString(n,0, 0.5), sP, minEfield/100.,maxEfield/100.))
        polarizabilityList.append(sP)

        dP = arc.DynamicPolarizability(sMap.atom, sMap.n, sMap.l, sMap.j, s=sMap.s);
        dP.defineBasis(sMap.nMin, sMap.nMax);
        dynPol = dP.getPolarizability(wavelength, units='SI', accountForStateLifetime=False, mj=None)

        polarizabilityList.append(dynPol)
        return polarizabilityList
  
    def set_atomicData(self, source='ARC', bfield_ON=True):
        '''
            Should go pair by pair of states and generate the datacell() to ultimately create atomicData() dataclass that belongs
        to the atomSpecs class
        '''
        _labels = []; _cells = [];
        _cells2save=[];

        if source!=self._source:
            self._source=source;

        if self._source=='ARC':
            # self.fullstates = _getFullStates(states_dict, self.atom.I);

            if bfield_ON:
                states_dict = _lst2dic([_state_v for _state_k, _state_v in self.fullstates.items()]);
            else:
                states_dict = _lst2dic([_state_v for _state_k, _state_v in self.partialstate.items()]);            

            for i in tqdm(range(len(states_dict)), leave=False):

                if bfield_ON:
                    istate = states_dict[i]; #istate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>}
                else:
                    istate = states_dict[i];  #istate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>, 'F': <val>, 'mF': <val>}

                for j in tqdm(range(len(states_dict))):

                    if bfield_ON:
                        jstate = states_dict[j]; #istate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>}
                    else:
                        jstate = states_dict[j]; #jstate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>, 'F': <val>, 'mF': <val>}

                    _labels.append('|'+str(i)+'X'+str(j)+'|'); #adding labels for atomicData dataclass
                    
                    
                    #gettting data from ARC

                    #transition frequency
                    __TransitionFrequency = self.getTransitionFrequency_fromARC(istate, jstate);
                    

                    #transition wavelength
                    try:
                        __TransitionWavelenght = self.getTransitionWavelength_fromARC(istate, jstate);

                    except:
                        __TransitionWavelenght = None;


                    #rabi frequency
                    __RabiFrequency = [];
                    try:
                        __RabiFrequency = self.getRabifreq_fromARC(istate, 
                                                                   jstate, 
                                                                   self._eParams['4RabiFreq']['polarization'], 
                                                                   self._eParams['4RabiFreq']['power'], 
                                                                   self._eParams['4RabiFreq']['waist']);

                    except:

                        try:
                            __RabiFrequency.append([[self.getRabifreq_fromARC(istate, 
                                                                              jstate, 
                                                                              int(laser.specs['beams'][_beam_idx]['Polarization']), 
                                                                              float(laser.specs['beams'][_beam_idx]['Power']), 
                                                                              float(laser.specs['beams'][_beam_idx]['Waist']))  for _beam_idx in range(len(laser.specs['beams'])) ] for laser in self._eParams['4RabiFreq'] if (__TransitionWavelenght-laser.specs['Wavelength']) < 10]);
                        
                        except:
                            __RabiFrequency.append(0.0);


                    #starkmap
                    __StarkMap = self.getSMap_fromARC(istate, 
                                                      self._eParams['4SMap']['nrange'], 
                                                      self._eParams['4SMap']['lmax'], 
                                                      self._eParams['4SMap']['bz'], 
                                                      self._eParams['4SMap']['erange'], 
                                                      self._eParams['4SMap']['n'], 
                                                      self._eParams['4SMap']['progressbar'], 
                                                      self._eParams['4SMap']['unit'], 
                                                      self._eParams['4SMap']['highlightstate'], 
                                                      self._eParams['4SMap']['highlightcolour']);


                    #spontaneous decay
                    __NaturalLifetime = self.getLifetime_fromARC(istate);
                    

                    #rydberg state
                    __RydbergState = self.isRydberg(istate);
                    

                    #blackbody radiation
                    try:
                        __BlackBodyRadiation = self.getBBR_fromARC(istate['n']-self._eParams['4BBR']['dn'], istate['n']+self._eParams['4BBR']['dn'], self._eParams['4BBR']['lj_vals'][0], self._eParams['4BBR']['lj_vals'][1], self._eParams['4BBR']['mintemp'], self._eParams['4BBR']['maxtemp']);
                    
                    except:
                        __BlackBodyRadiation = None;
                    

                    #c6
                    if __RydbergState is True:
                        __C6 = self.getC6_fromARC(__StarkMap, self._eParams['4C6']['phi'], self._eParams['4C6']['theta'], self._eParams['4C6']['dn'], self._eParams['4C6']['dEmax'], self._eParams['4C6']['ploton'] );

                        __BlockadeRadius = self.getRb_fromARC([__RabiFrequency], __C6[0]);
                    
                    else:
                        __C6 = None;
                        __BlockadeRadius = None;
                        
                    __LeRoyRadius = self.getLeRoyRadius_fromARC( __StarkMap);
                    
                    #polarizability
                    try:
                        __Polarizability = self.getPolarizability_fromARC(__StarkMap, __TransitionWavelenght);
                    except:
                        __Polarizability = None;
                        
                    ##NotImplemented
                    #energy
                    __Energy = None;
                    #c3
                    __C3 = None;
                    #dipole matrix element
                    __Dip_Mat_element = None;
                    
                    #datacell for python object
                    _cell = datacell(Energy=__Energy,
                                     Transition_freq=__TransitionFrequency,
                                     Wavelength=__TransitionWavelenght,
                                     Rabi_freq=__RabiFrequency,
                                     Dip_Mat_element=__Dip_Mat_element,
                                     Lifetime=__NaturalLifetime,
                                     Rydberg=__RydbergState,
                                     BBR=__BlackBodyRadiation,
                                     C3=__C3,
                                     C6=__C6,
                                     Rbl=__BlockadeRadius,
                                     LeRoy_radius=__LeRoyRadius,
                                     Polarizability=__Polarizability,
                                     Stark_Map=__StarkMap,);
                    _cells.append(_cell);
                    
                    #datacell for stored file
                    #adding exceptions that includes plots
                    try:
                        _c6COPY = __C6[0]
                    except:
                        _c6COPY = __C6
                    try:
                        _BBRCOPY = extract_BBRData(__BlackBodyRadiation)
                    except:
                        _BBRCOPY = __BlackBodyRadiation   
                    _cell2save = datacell(Energy=__Energy,
                                     Transition_freq=__TransitionFrequency,
                                     Wavelength=__TransitionWavelenght,
                                     Rabi_freq=__RabiFrequency,
                                     Dip_Mat_element=__Dip_Mat_element,
                                     Lifetime=__NaturalLifetime,
                                     Rydberg=__RydbergState,
                                     BBR=_BBRCOPY,
                                     C3=__C3,
                                     C6=_c6COPY,
                                     Rbl=__BlockadeRadius,
                                     LeRoy_radius=__LeRoyRadius,
                                     Polarizability=__Polarizability,
                                     Stark_Map=extract_SMapData(__StarkMap));
                    _cells2save.append(_cell2save);
                    
            print("There are "+str(len(_cells))+" cells")
            
            self._metadata.update({'cell_labels' : str(_labels)})
            self.atomicdata = atomicData(clabel=_labels, cells=_cells )
            self._atomicdata2save = atomicData(clabel=_labels, cells=_cells2save )
    
    def update(variable, recompilation=True):
    
        if variable=='power':
            pass 
    
    def save(self, fname, cformat='gzip', schema=True):
        '''
            Store the data of the atomic specifications and metadata in .parquet format compressed files.
            
            INPUTS:
            -------
            
                fname :
                cformat :
                schema :
            
            OUTPUTS:
            --------
            
                <atomicspecification>.parquet file
                <metadata>.parquet file
        '''
        fullDataFrame = pd.concat([pd.DataFrame([cell]) for cell in self._atomicdata2save.cells]);
        
        if schema==False:
            _metadaDF = pd.DataFrame(self._metadata, index=[0]);
            _metadaDF.to_parquet('metadata_'+fname+'.parquet', compression=cformat);
            fullDataFrame.to_parquet(fname+'.parquet', compression=cformat);
            
            print('Atomic specifications stored!')
        
        else:
            _metadaDF = pd.DataFrame(self._metadata, index=[0]);
            _metadaDF.to_parquet('metadata_'+fname+'.parquet', compression=cformat);
            paTable = pa.Table.from_pandas(fullDataFrame, schema=pa.Schema.from_pandas(fullDataFrame), nthreads=os.cpu_count());
            pq.write_table(paTable, fname+'.parquet', compression=cformat);
            
            print('Atomic specifications stored!')  
         
@dataclass(frozen=True)
class atomicData:
    '''
        Special dataclass of AQiPT that store the data of many (i,j) tuple of states of single and multiple
        atoms using datacells dataclass.
        
        ATTRIBUTES/FIELDS:
        ------------------

            clabel (list): labels of the cells in atomicData dataclass, usual labels '|iXj|' for the i-th and j-th
                           states, correspondent to the i,j in {0, 1, 2, 3,...} for the elements of the state list
            
            cells  (list): list of datacells containing all physics information related to the state tuple (i,j)
        
    '''
    #fields
    clabel : list;
    cells : list;
    
@dataclass(frozen=False)
class datacell:
    '''
        Special dataclass of AQiPT that store the data of one (i,j) tuple of states of single or multiple
        atoms.
        
        ATTRIBUTES/FIELDS:
        ------------------
        
            Energy :
            Transition_freq :
            Wavelength :
            Rabi_freq :
            Dip_Mat_element :
            Lifetime :
            Rydberg :
            BBR :
            C3 :
            C6 :
            Rbl :
            LeRoy_radius :
            Polarizability :
            Stark_Map :
        
        
        METHODS:
        --------
        
        
    '''
        
    #fields
    Energy : float = field(default=0.0, metadata={'unit': 'Hz', 'value':None})
    Transition_freq : float = field(default=0.0, metadata={'unit': 'Hz'})
    Wavelength : float = field(default=0.0, metadata={'unit': 'nm'})
    Rabi_freq : float = field(default=0.0, metadata={'unit': 'Hz'})
    Dip_Mat_element : float = field(default=0.0, metadata={'unit': 'Hz'})
    Lifetime : float = field(default=0.0, metadata={'unit': 's'})
    Rydberg : bool = field(default=False, metadata={'unit': 'none'})
    BBR : float = field(default=0.0, metadata={'unit': 'none'})
    C3 : float = field(default=0.0, metadata={'unit': 'Hz/um^3'})
    C6 : float = field(default=0.0, metadata={'unit': 'Hz/um^6'})
    Rbl : float = field(default=0.0, metadata={'unit': 'um'})
    LeRoy_radius : float = field(default=0.0, metadata={'unit': 'um'})
    Polarizability : float = field(default=0.0, metadata={'unit': 'none'})
    Stark_Map : list = field(default_factory=list, metadata={'axis1': 'value', 'axis2':'value'})

#####################################################################################################
#atomicKernel AQiPT class
#####################################################################################################
    
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/');
        except ValueError:
            return None
        try:
            leading, num = num.split(' ');
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1;
        else:
            sign_mult = 1;
        return float(leading) + sign_mult * (float(num) / float(denom));

def orb2Int(orb):
    return [0 if orb =='S' 
            else 1 if orb == 'P' 
            else 2 if orb == 'D' 
            else 3 if orb == 'F' 
            else None][0]

def intensity(P, w0):
    return (2*P)/(np.pi*(w0**2));

def lst2dic(ket):
    '''
        Helper function for passing a list of strings or just one string describing the state, returning a dictionary
        with the labeled quantum numbers.

        Example: 

            '4P_1/2, 1/2' -----> {'n': 4, 'l': 1, 'j': 0.5, 's': 0.5}
            '4P_1/2, 1/2, 1, -1' -----> {'n': 4, 'l': 1, 'j': 0.5, 's': 0.5, 'F': 1, 'mF': -1}
    '''
    try:
        if type(ket) is list:
            for element in ket:
                try:
                    ket_lst={}; nr_state=0;
                    
                    for state in element:

                        l_val = ''.join([n for n in state if n.isupper()]);
                        n_val = state.rpartition(l_val)[0];
                        j_val = state[state.find('_')+len('_'):state.rfind(',')];
                        s_val = state[state.find(',')+len(',')+1:state.rfind('')];
                        ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)})
                        nr_state+=1;
                    return ket_lst

                except:
                    ket_lst={}; nr_state=0;
                
                    for state in element:

                        n_val = state[0];
                        l_val = state[1];
                        j_val = state[2];
                        s_val = state[3];
                        ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)})
                        nr_state+=1;
                        print('A')
                    return ket_lst

        else:
            
            l_val = ''.join([n for n in ket if n.isupper()])
            n_val = ket.rpartition(l_val)[0]
            j_val = ket[ket.find('_')+len('_'):ket.rfind(',')]
            s_val = ket[ket.find(',')+len(',')+1:ket.rfind('')]
            print('B:', {'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)})
            return {'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)}
    except:

        if type(ket) is list:
            for element in ket:
                try:
                    ket_lst = {}
                    nr_state = 0

                    for state in element:
                        l_val = ''.join([n for n in state if n.isupper()])
                        n_val = state.rpartition(l_val)[0]
                        j_val = state.split('_')[1].split(',')[0]
                        s_val, F_val, mF_val = state.split(',')[-3:]

                        ket_lst[str(nr_state)] = {
                            'n': int(n_val),
                            'l': orb2Int(l_val),
                            'j': convert_to_float(j_val.strip()),
                            's': convert_to_float(s_val.strip()),
                            'F': int(F_val.strip()),
                            'mF': int(mF_val.strip())
                        }
                        nr_state += 1
                        print('C')
                    return ket_lst

                except:

                    try:
                        ket_lst={}; nr_state=0;
                    
                        for state in element:

                            n_val = state[0];
                            l_val = state[1];
                            j_val = state[2];
                            s_val = state[3];
                            F_val = state[4];
                            mF_val = state[5];
                            ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val), 'F':float(F_val), 'mF':float(mF_val)})
                            nr_state+=1;
                            print('D')
                        return ket_lst
                    except:
                        for state in element:
                            l_val = ''.join([n for n in state if n.isupper()])
                            n_val = state.rpartition(l_val)[0]
                            j_val = state.split('_')[1].split(',')[0]
                            s_val, F_val, mF_val = state.split(',')[-3:]
                            print('E: ', {
                                'n': int(n_val),
                                'l': orb2Int(l_val),
                                'j': convert_to_float(j_val.strip()),
                                's': convert_to_float(s_val.strip()),
                                'F': int(F_val.strip()),
                                'mF': int(mF_val.strip())
                            })
                            return {
                                'n': int(n_val),
                                'l': orb2Int(l_val),
                                'j': convert_to_float(j_val.strip()),
                                's': convert_to_float(s_val.strip()),
                                'F': int(F_val.strip()),
                                'mF': int(mF_val.strip())}

        else:
            l_val = ''.join([n for n in ket if n.isupper()])
            n_val = ket.rpartition(l_val)[0]
            j_val = ket.split('_')[1].split(',')[0]
            s_val, F_val, mF_val = ket.split(',')[-3:]
            print('F: ', {
                'n': int(n_val),
                'l': orb2Int(l_val),
                'j': convert_to_float(j_val.strip()),
                's': convert_to_float(s_val.strip()),
                'F': int(F_val.strip()),
                'mF': int(mF_val.strip())
            })
            return {
                'n': int(n_val),
                'l': orb2Int(l_val),
                'j': convert_to_float(j_val.strip()),
                's': convert_to_float(s_val.strip()),
                'F': int(F_val.strip()),
                'mF': int(mF_val.strip())
            }

def _lst2dic(lst):
    
    '''
    
        Helper function for passing a list of strings or just one string describing the state, returning a dictionary
        with the labeled quantum numbers.

        Example: 

            '4P_1/2, 1/2' -----> {'n': 4, 'l': 1, 'j': 0.5, 's': 0.5}
            '4P_1/2, 1/2, 1, -1' -----> {'n': 4, 'l': 1, 'j': 0.5, 's': 0.5, 'F': 1, 'mF': -1}
        
        CASES:
        -----
        
        A: n,l,j,s
        B: n,l,j,s,F,mF
        
        case1 = [] : lst 
        case2 = '' : str
        case3 = ['',''] : list of str
        case4 = [[],[]] : list of list
        
    '''
    if isinstance(lst, list):#case1
        
        if aqipt.is_list_of_lists(lst):#case4

            _dict2return = {}; _element_idx=0;
            for element in lst:

                try:
                    #case4 A
                    n_val = int(element[0]);
                    l_val = int(element[1]);
                    j_val = float(element[2]);
                    s_val = float(element[3]);
                    F_val = float(element[4]);
                    mF_val = float(element[5]);
                    _dict2return[_element_idx] =  {'n':n_val, 
                                                   'l':l_val, 
                                                   'j':j_val, 
                                                   's':s_val,
                                                   'F':F_val,
                                                   'mF':mF_val};
                except:
                    #case4 B
                    n_val = int(element[0]);
                    l_val = int(element[1]);
                    j_val = float(element[2]);
                    s_val = float(element[3]);
                    _dict2return[_element_idx] = {'n':n_val, 
                                                  'l':l_val, 
                                                  'j':j_val, 
                                                  's':s_val};  

                _element_idx+=1;

            return _dict2return         
            
        elif aqipt.is_list_of_strings(lst):#case3

            _dict2return = {}; _element_idx=0;
            for element in lst:

                try:#case3 A                    
                    l_val = ''.join([n for n in element if n.isupper()]);
                    n_val = element.rpartition(l_val)[0];
                    j_val = element[element.find('_')+len('_'):element.rfind(',')];
                    s_val = element[element.find(',')+len(',')+1:element.rfind('')];
                    _dict2return[_element_idx] =   {'n':int(n_val), 
                                                    'l':orb2Int(l_val), 
                                                    'j':convert_to_float(j_val), 
                                                    's':convert_to_float(s_val)}
                except:#case3 B                    
                    l_val = ''.join([n for n in element if n.isupper()]);
                    n_val = element.rpartition(l_val)[0];
                    j_val = element.split('_')[1].split(',')[0];
                    s_val, F_val, mF_val = element.split(',')[-3:];
                    _dict2return[_element_idx] =   {'n': int(n_val),
                                                    'l': orb2Int(l_val),
                                                    'j': convert_to_float(j_val.strip()),
                                                    's': convert_to_float(s_val.strip()),
                                                    'F': int(F_val.strip()),
                                                    'mF': int(mF_val.strip())}

                _element_idx+=1;

            return _dict2return  

        else:
            try:
                try:
                    #case1 B
                    n_val = int(lst[0]);
                    l_val = int(lst[1]);
                    j_val = float(lst[2]);
                    s_val = float(lst[3]);
                    F_val = float(lst[4]);
                    mF_val = float(lst[5]);
                    return {'n':n_val, 
                            'l':l_val, 
                            'j':j_val, 
                            's':s_val,
                            'F':F_val,
                            'mF':mF_val}
                except:
                    #case1 B
                    n_val = int(lst[0]);
                    l_val = int(lst[1]);
                    j_val = float(lst[2]);
                    s_val = float(lst[3]);
                    return {'n':n_val, 
                            'l':l_val, 
                            'j':j_val, 
                            's':s_val}
                
            except:
                print('Format not recognised.')
     
    elif isinstance(lst, str): #case2
        
        try:#case2 A            
            l_val = ''.join([n for n in lst if n.isupper()]);
            n_val = lst.rpartition(l_val)[0];
            j_val = lst[lst.find('_')+len('_'):lst.rfind(',')];
            s_val = lst[lst.find(',')+len(',')+1:lst.rfind('')];
            return {'n':int(n_val), 
                    'l':orb2Int(l_val), 
                    'j':convert_to_float(j_val), 
                    's':convert_to_float(s_val)}
        except:#case2 B
            l_val = ''.join([n for n in lst if n.isupper()]);
            n_val = lst.rpartition(l_val)[0];
            j_val = lst.split('_')[1].split(',')[0];
            s_val, F_val, mF_val = lst.split(',')[-3:];
            return {'n': int(n_val),
                    'l': orb2Int(l_val),
                    'j': convert_to_float(j_val.strip()),
                    's': convert_to_float(s_val.strip()),
                    'F': int(F_val.strip()),
                    'mF': int(mF_val.strip())}

class atomicKernel:

    '''
        AQiPT class that introduce the notion of the atomic physics into the simulations executed with atomicModel() class.
        The class allows to add all the constrains set by the physics of the system (e.g., neutral and Rydberg atoms). The
        use of the atomicKernel class to generate new atomicModel class-objects open the oportunity to have more realistic 
        and precise simulation of the systems, specially when the system is extended into arrays of qubits. atomicQRegister class.
        
        This class can be feed with precise information given by the atomicSpec class.
        
        
        ATTRIBUTES:
        -----------
            kernelname :
            element :
            states :
            nrState :
            labsetup :
            atomSpecs :
            couplings :
            detunings :
            dissipators :
            params4AMs :
            
        
        METHODS:
        --------
            kernel2Model :
            loadSpec :
            getRabi :
            getGamma :
            __updateParams4AMs :
    
    '''
    def __init__(self, element, statelst, labsetup=None, atomSpec=None, label='default_KernelName'):
        
        self.kernelname = label;
        self.element = [arc.Hydrogen() if element == 'Hydrogen'
                        else arc.Lithium6() if element == 'Lithium6'
                        else arc.Lithium7() if element == 'Lithium7'
                        else arc.Sodium() if element == 'Sodium'
                        else arc.Potassium39() if element == 'Potassium39'
                        else arc.Potassium40() if element == 'Potassium40'
                        else arc.Potassium41() if element == 'Potassium41'
                        else arc.Rubidium85() if element == 'Rubidium85'
                        else arc.Rubidium87() if element == 'Rubidium87'
                        else arc.Caesium() if element == 'Caesium'
                        else None ][0];
        self.states = lst2dic(statelst);
        self._nrStates = len(statelst);
        
        self.labsetup = dict(sorted(labsetup.items()));
        self.atomSpecs = atomSpec;
        self.couplings = None;
        self.detunings = None;
        self.dissipators = None;
        self.params4AMs = {'couplings': self.couplings,
                           'detunings': self.detunings,
                           'dissipators': self.dissipators}; #wrapping dynamic params in dictionary
        
        self._historyAM = [];

    def kernel2Model(self, AM, EXTatomicSpec=None, newName='default_NewModel'):
        '''
            atomicKernel method to generate new atomicModel based in AMO physics estimations e.g., python
            packages or atomic specifications.
            
            INPUTS:
            ------
                AM :
                EXTatomicSpec :
                newName :
                
            OUTPUTS:
            --------
                atomicModel() class
        '''

        if EXTatomicSpec==None:
            
            if self.atomSpecs==None:
                
                print('No atomicSpec found. Embedding kernel in model with ARC \n')

                #### creates COUPLINGS for NEW atomicModel() from atomicKernel() ####

                self.couplings = AM.dynParams['couplings'].copy(); #copy of atomicModel dynparams for protect AM values
                _newCoupling={};
                for label,coupling in self.couplings.items():

                    _istate = self.states[str(coupling[0][0])]; #[list values] [map coupling] [1st edge: state ith]
                    _jstate = self.states[str(coupling[0][1])]; #[list values] [map coupling] [2nd edge: state jth]


                    _OmegaR = self.getRabi(_istate, _jstate); #calculate Rabi freq based in atomicKernel()

                    _newCoupling.update({label: [coupling[0], _OmegaR, coupling[2]]}); #substitute value of the new Rabi freq



                #### creates DISSIPATORS for NEW atomicModel() from atomicKernel() ####

                self.dissipators = AM.dynParams['dissipators'].copy(); #copy of atomicModel dynparams for protect AM values
                _newDissipator={};
                for label,dissipator in self.dissipators.items():

                    _istate = self.states[str(dissipator[0][0])]; #[list values] [map coupling] [1st edge: state ith]
                    _jstate = self.states[str(dissipator[0][1])]; #[list values] [map coupling] [2nd edge: state jth]

                    #check rules
    #                 print(int(abs(_istate['l']-_jstate['l'])), int(abs(_istate['j']-_jstate['j'])), int(abs(_istate['s']-_jstate['s'])))

                    if int(abs(_istate['l']-_jstate['l']))==+1 or int(abs(_istate['l']-_jstate['l']))==-1: #Delta l = +- 1
                        if int(abs(_istate['j']-_jstate['j']))==+1 or int(abs(_istate['j']-_jstate['j']))==-1 or int(abs(_istate['j']-_jstate['j']))==0: #Delta l = 0, +- 1
                            if int(abs(_istate['s']-_jstate['s']))==+1 or int(abs(_istate['s']-_jstate['s']))==-1 or int(abs(_istate['s']-_jstate['s']))==0: #Delta l = 0, +- 1
                                _Gamma = self.getGamma(_istate, _jstate); #calculate Spontaneous decay based in atomicKernel()
                                _newDissipator.update({label: [dissipator[0], _Gamma]}); #substitute value of the new Dissipator
                    else:
                        _gamma = dissipator[1]; #inherits dephasing rate based in atomicKernel()
                        _newDissipator.update({label: [dissipator[0], _gamma]}); #substitute value of the new Dissipator



                #### creates DETUNINGS FOR NEW atomicModel() from atomicKernel() ####
                '''creating list of new values of detuning given by the labsetup dict, where it finds a 'AOM' keyword'''

                _newDetuningslst=[]; _idx_det=0;
                for key, value in self.labsetup.items():
                    if 'LASER' in key:
                        value_atom = value['Frequency']
                    if 'AOM' in key:
                        _newDetuningslst.append(value['Frequency_resonance_scan']);

                    _idx_det+=1;
                    if _idx_det>=len(AM.dynParams['detunings']):
                        break

                self.detunings = AM.dynParams['detunings'].copy(); #copy of atomicModel dynparams for protect AM values

                _newDetuning={}; _idx_det=0;
                for label,detuning in self.detunings.items():
                    _newDetuning.update({label: [detuning[0], _newDetuningslst[_idx_det], detuning[2]]});
                    _idx_det+=1;


                #### update params for atomicModels() ####

                self.couplings = _newCoupling; #substitute of new couplings in atomicKernel() couplings specs
                self.detunings = _newDetuning; #substitute of new detunings in atomicKernel() detunings specs
                self.dissipators = _newDissipator; #substitute of new dissipators in atomicKernel() dissipators specs
                self.__updateParams4AMs();
                
                __NewAM = atomicModel(AM.times, self._nrStates, AM._psi0, self.params4AMs, name = newName);
                self._historyAM.append(__NewAM); #storing new atomic model in kernel history
                
                return __NewAM
            
            else:
                print('atomicSpec found in Kernel. Embedding kernel in model \n')
                ## embeding here from own kernel's atomic spec
        else:
            
            print('atomicSpec provided. Embedding kernel in model \n')
            self.loadSpec(EXTatomicSpec);
            ## embeding here from external specs
            
    def loadSpec(self, atomSpec=None):
        '''
            atomicKernel method for loading atomic specifications (atomicSpecs-class) into kernel to be use for new atomicModel-class. 
            It use the input specs from constructor of atomicKernel-class or a new one as input of the method.
            
            INPUTS:
            -------
            
                atomSpec : 
            
            
            OUTPUTS:
            --------
            
        '''
        if atomSpec==None:
            if self.atomSpecs==None:
                print("ERROR: Atomic specifications not found.")
            else:
                print("Specifications loaded into ", self.kernelname)
        else:
            self.atomSpecs = atomSpec.copy();
        
    def getRabi(self, istate, jstate):
        '''
            atomicKernel method for calculating Rabi Frequency using ARC python library.
            
            INPUTS:
            ------
            
                istate : 
                jstate :
            
            
            OUTPUTS:
            --------
            
            
        '''
        return self.element.getDipoleMatrixElement(int(istate['n']), istate['l'], istate['j'], istate['s'],
                                                   jstate['n'], jstate['l'], jstate['j'], jstate['s'],0)#*(np.sqrt(2*intensity_MW/e0*c )/(2*np.pi*1e9))
    
    def getGamma(self, istate, jstate):
        '''
            atomicKernel method for calculating Natural life-time using ARC python library.
            
            INPUTS:
            ------
            
                istate : 
                jstate :
            
            
            OUTPUTS:
            --------
            
            
        '''
        return 1e-6*(1/self.element.getStateLifetime(int(jstate['n']), int(jstate['l']), jstate['j']))/(2*np.pi)

    def __updateParams4AMs(self):
        '''
            atomicKernel method for updating values of input parameters for atomicModel class
        '''
        self.params4AMs['couplings'] = self.couplings;
        self.params4AMs['detunings'] = self.detunings;
        self.params4AMs['dissipators'] = self.dissipators;


#####################################################################################################
#hardwareSpecs AQiPT class
#####################################################################################################

class hardwareSpecs(object):

    '''
        AQiPT class that use a JSON parser of real setup and instantiate the different optical and
        electronic elements of the setup. The class 
        
        
        ATTRIBUTES:
        -----------
            attribute :
            
        
        METHODS:
        --------
            method :

    '''

    def __init__(self):
        
        self._author = None;
        self._affiliation = None;
        self._website = None;
        self._dateCreation = None;
        self._comments = None; 
        self.specifications = None;
        self.hardwareLSType = [];
        self.hardwareTypes = {"AOM":0xf4240,
                              "laser":0xf4241,
                              "laser-aux":0xf4242,
                              "camera":0xf4243,
                              "awg":0xf4244,
                              "dds":0xf4245,
                              "IQmixer":0xf4246,
                              "dmd":0xf4247,
                              "shutter":0xf4248,
                              "mixer":0xf4249,
                              "electrode":0x0f4250,
                              "coil":0x0f4251,
                              "sensor":0x0f4252,
                              "AOD":0x0f4253,
                              "slm":0x0f4254,
                              "antenna":0x0f4255
                              }; #memory reserved from >hex(1M) 

                              #add switch, AODs, change dds->synthetizer
        self.hardwareLST = {k: [] for k, v in self.hardwareTypes.items()};
        self._IDs_bench = None;

    def loadSpecs(self, path=directory.hardware_specs_dir, printON=False):
        
        #load hardware_specs JSON file
        os.chdir(path)
        with open('hardware_specs_demo.json', 'r') as specs_file:
            hardware_specs = json.load(specs_file);
            if printON:
                print(json.dumps(hardware_specs, indent=1, sort_keys=True))
            self.specifications = hardware_specs;
            self._author = self.specifications['author'];
            self._affiliation = self.specifications['affiliation'];
            self._website = self.specifications['website'];
            self._dateCreation = self.specifications['created'];
            self._comments = self.specifications['comments']; 
        pass

        #load listed devices
        for dev_type in self.hardwareTypes:
            dev_lst=[];
            for element in self.specifications['setup']:
                if dev_type in element.keys():
                    dev_lst.append(element);
                     
            keys = []; vals = [];
            for data in dev_lst:
                val = []
                for k,v in data.items():
                    keys.append(k);
                    val.append(v);
                vals.append(val);

            pd_dev = pd.DataFrame([v for v in vals], columns=list(dict.fromkeys(keys)))
            self.hardwareLSType.append(pd_dev);

            if printON:
                display(HTML(pd_dev.to_html()))

    def printHeader(self):
        print('{0} | {1} | {2} '.format(self._author, self._affiliation, self._website))
        print('Created: {0} \n'.format(self._dateCreation))
        print('Comments: {0}\n'.format(self._comments))

    def createAOM(self, specs=None, driver=None):
        aom_obj = AOM(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['AOM'].append(aom_obj); #added to the list of hardware objects
    
    def createLaser(self, specs=None, driver=None):
        laser_obj = Laser(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['laser'].append(laser_obj); #added to the list of hardware objects

    def createLaserAUX(self, specs=None, driver=None):
        laserAUX_obj = LaserAUX(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['laser-aux'].append(laserAUX_obj); #added to the list of hardware objects

    def createCamera(self, specs=None, driver=None):
        camera_obj = Camera(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['camera'].append(camera_obj); #added to the list of hardware objects

    def createAWG(self, specs=None, driver=None):
        awg_obj = AWG(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['awg'].append(awg_obj); #added to the list of hardware objects
   
    def createDDS(self, specs=None, driver=None):
        dds_obj = DDS(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['dds'].append(dds_obj); #added to the list of hardware objects

    def createIQmixer(self, specs=None, driver=None):
        iqmixer_obj = IQmixer(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['IQmixer'].append(iqmixer_obj); #added to the list of hardware objects

    def createDMD(self, specs=None, driver=None):
        dmd_obj = DMD(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['dmd'].append(dmd_obj); #added to the list of hardware objects

    def createSLM(self, specs=None, driver=None):
        slm_obj = SLM(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['slm'].append(slm_obj); #added to the list of hardware objects

    def createShutter(self, specs=None, driver=None):
        shutter_obj = Shutter(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['shutter'].append(shutter_obj); #added to the list of hardware objects
    
    def createMixer(self, specs=None, driver=None):
        mixer_obj = Mixer(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['mixer'].append(mixer_obj); #added to the list of hardware objects

    def createElectrode(self, specs=None, driver=None):
        electrode_obj = Electrode(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['electrode'].append(electrode_obj); #added to the list of hardware objects

    def createCoil(self, specs=None, driver=None):
        coil_obj = Coil(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['coil'].append(coil_obj); #added to the list of hardware objects

    def createSensor(self, specs=None, driver=None):
        sensor_obj = Sensor(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['sensor'].append(sensor_obj); #added to the list of hardware objects

    def createAOD(self, specs=None, driver=None):
        aod_obj = AOD(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['AOD'].append(aod_obj); #added to the list of hardware objects

    def createElectrode(self, specs=None, driver=None):
        electrode_obj = Electrode(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['electrode'].append(electrode_obj); #added to the list of hardware objects

    def createAntenna(self, specs=None, driver=None):
        antenna_obj = Antenna(specs, driver); #creates sub-instance of the hardwareSpecs class
        self.hardwareLST['antenna'].append(antenna_obj); #added to the list of hardware objects

    def initSetup(self):
        
        for dtype in self.hardwareTypes:

            dtype_idx = list(self.hardwareTypes).index(dtype); #index from hardware LUT
            dtype_specs = self.hardwareLSType[dtype_idx]; #set of specs of the dtype hardware
            dtype_addrs = self.hardwareTypes[dtype]; #LUT address allocation of dtype hardware

            for i in range(len(dtype_specs)):
                dtype_spec_i = dtype_specs.iloc[i]; #iterate over hardware elements

                if dtype_addrs==self.hardwareTypes['AOM']:
                    self.createAOM(specs=dtype_spec_i, driver=None);

                if dtype_addrs==self.hardwareTypes['laser']:
                    self.createLaser(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['laser-aux']:
                    self.createLaserAUX(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['camera']:
                    self.createCamera(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['awg']:
                    self.createAWG(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['dds']:
                    self.createDDS(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['IQmixer']:
                    self.createIQmixer(specs=dtype_spec_i, driver=None);

                if dtype_addrs==self.hardwareTypes['dmd']:
                    self.createDMD(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['shutter']:
                    self.createShutter(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['mixer']:
                    self.createMixer(specs=dtype_spec_i, driver=None);
                
                if dtype_addrs==self.hardwareTypes['electrode']:
                    self.createElectrode(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['coil']:
                    self.createCoil(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['sensor']:
                    self.createSensor(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['AOD']:
                    self.createAOD(specs=dtype_spec_i, driver=None);

                if dtype_addrs==self.hardwareTypes['slm']:
                    self.createSLM(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

                if dtype_addrs==self.hardwareTypes['antenna']:
                    self.createAntenna(specs=dtype_spec_i, driver=dtype_spec_i['driver']);

###===========================###
### hardwareSpecs sub-classes ###
###===========================###

class AOM(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate AOMs on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Laser(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Lasers on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class LaserAUX(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Auxiliar Lasers on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Camera(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Camera on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class AWG(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate AWGs on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);
 
class DDS(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate DDSs on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class IQmixer(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate IQ mixers on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Mixer(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Mixers on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Shutter(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Shutters on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class DMD(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate DMD on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class SLM(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate SLM on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Electrode(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Electrodes on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Coil(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Coil on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Sensor(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Electrodes on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class AOD(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate AODs on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);

class Antenna(hardwareSpecs):

    '''
        AQiPT sub-class that use a JSON parser of real setup and instantiate Antennas on the setup.
        The class 


        ATTRIBUTES:
        -----------
            attribute :


        METHODS:
        --------
            method :

    '''

    def __init__(self, specs, driver, *args, **kwargs):
        self.specs = specs;
        self.driver = driver;
        self.channels = {'IDs':[], 'physical_address':[], 'assigned_track': []};
        super().__init__(*args, **kwargs);
        
#####################################################################################################
#softwareSpecs AQiPT class
#####################################################################################################

class softwareSpecs(object):

    '''
        AQiPT class that use a JSON parser of instructions protocol.
        
        
        ATTRIBUTES:
        -----------
            attribute :
            
        
        METHODS:
        --------
            method :

    '''

    def __init__(self):
        
        self._author = None;
        self._affiliation = None;
        self._website = None;
        self._dateCreation = None;
        self._comments = None; 
        self.specifications = None;
        self._headers = ['variables_UNITS', 'variables', 'tracks', 'sequences', 'instructions'];
        self._IDs_bench = None;
        self._python_waveforms = None;

    
    def loadSpecs(self, path=directory.compiler_dir, printON=False):
        
        #load software_specs JSON file
        os.chdir(path)
        with open('software_specs.json', 'r') as specs_file:
            software_specs = json.load(specs_file);
            if printON:
                print(json.dumps(software_specs, indent=1, sort_keys=True))
            self.specifications = software_specs;
            self._author = self.specifications['author'];
            self._affiliation = self.specifications['affiliation'];
            self._website = self.specifications['website'];
            self._dateCreation = self.specifications['created'];
            self._comments = self.specifications['comments']; 
        pass
        
    def update(self, waveforms=None, variables=None, variables_UNITS=None, instructions=None, configuration=None, save_waveforms=False):
        '''
            Update the software specifications (modify the current .json file of software_specs)
            with the Sequences (AQiPT.control.sequence -object), extracting the _API_sequences 
            attribute and placing it in the specifications for later compilation of experiment
            (AQiPT.control.experiment -object) where schedule is compiled.
        '''

        #updating variables
        if variables!=None:
            self.specifications['variables'] = variables;
        elif variables==None:
            _variables_temp_dict = [];

        #updating waveforms e.g., tracks and sequences
        if waveforms!=None:

            for waveform in waveforms:
                
                #try sequences
                if isinstance(waveform, control.sequence):
                    
                    self._python_waveforms = waveforms;

                    _instruction_name_list_ = [];

                    for _idx_instruction in range(len(waveform._API_sequence)):
                        
                        #storing specs for digital
                        if waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE']=="DIGITAL":
                            
                            _inst_type_ = ' DIGITAL';
                            _instruction_name_i_ = (waveform._API_sequence[_idx_instruction][1]['name']+_inst_type_).replace(" ", "_");
                            
                            _instruction_name_list_.append(_instruction_name_i_);

                            _spec_inst_i_ = {_instruction_name_i_ : {"ID": hex(random.randint(1e4, 1.5e4)),
                                                                     "TYPE": waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE'],
                                                                     "CHANNEL": waveform._API_sequence[_idx_instruction][1]['identifier'],
                                                                     "ACTIVE": "TRUE",
                                                                     "TIME_FLAG": waveform._API_sequence[_idx_instruction][1]['time_range'][0],
                                                                     "COMMAND": waveform._API_sequence[_idx_instruction][0].specs['driver']['digital_cmd'],
                                                                     "SPECS": {"name": waveform._API_sequence[_idx_instruction][1]['name'], "args":aqipt.remove_keywords_from_dict(waveform._API_sequence[_idx_instruction][1], ['API_instruction_TYPE', 'identifier', 'dyn_params', 'name'])},
                                                                     "METADATA": "Digital load from script."}};
                            self.specifications['instructions'].append(_spec_inst_i_); #add instruction
                            if variables == None:
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][0]['identifier']+'_timeflag' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['time_range'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][0]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['timeflags'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented


                        #storing specs for digitals
                        if waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE']=="DIGITALS":
                            
                            _inst_type_ = ' DIGITALS';
                            _instruction_name_i_ = (waveform._API_sequence[_idx_instruction][1]['name']+_inst_type_).replace(" ", "_");
                            
                            _instruction_name_list_.append(_instruction_name_i_);

                            _spec_inst_i_ = {_instruction_name_i_ : {"ID": hex(random.randint(1e4, 1.5e4)),
                                                                     "TYPE": waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE'],
                                                                     "CHANNEL": waveform._API_sequence[_idx_instruction][1]['identifier'],
                                                                     "ACTIVE": "TRUE",
                                                                     "TIME_FLAGS": waveform._API_sequence[_idx_instruction][1]['timeflags'][0],
                                                                     "COMMAND": waveform._API_sequence[_idx_instruction][0].specs['driver']['digital_cmd'],
                                                                     "SPECS": {"name": waveform._API_sequence[_idx_instruction][1]['name'], "args":aqipt.remove_keywords_from_dict(waveform._API_sequence[_idx_instruction][1], ['API_instruction_TYPE', 'identifier', 'dyn_params', 'name'])},
                                                                     "METADATA": "Digital load from script."}};
                            self.specifications['instructions'].append(_spec_inst_i_); #add instruction
                            if variables == None:
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['timeflags'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['timeflags'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented


                        #storing specs for analog
                        elif waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE']=="ANALOG":

                            if save_waveforms==True:
                                _waveform = [_WAVEFORM.tolist() for _WAVEFORM in waveform.digiWaveformStack];
                            else:
                                _waveform = "None"

                            _inst_type_ = ' ANALOG';
                            _instruction_name_i_ = (waveform._API_sequence[_idx_instruction][1]['name']+_inst_type_).replace(" ", "_")
                            
                            _instruction_name_list_.append(_instruction_name_i_);
                            
                            _spec_inst_i_ = {_instruction_name_i_ : {"ID": hex(random.randint(1e4, 1.5e4)),
                                                                     "TYPE": waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE'],
                                                                     "CHANNEL": waveform._API_sequence[_idx_instruction][1]['identifier'],
                                                                     "WAVEFORM": str(_waveform),
                                                                     "TIME_FLAG": 0,
                                                                     "COMMAND": waveform._API_sequence[_idx_instruction][0].specs['driver']['analog_cmd'],
                                                                     "SPECS": {"name": waveform._API_sequence[_idx_instruction][1]['name'], "args":aqipt.remove_keywords_from_dict(waveform._API_sequence[_idx_instruction][1], ['API_instruction_TYPE', 'identifier', 'name'])},
                                                                     "METADATA": "Analog load from script."}};
                            self.specifications['instructions'].append(_spec_inst_i_); #add instruction
                            if variables == None:
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': 0, 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented

                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_amplitude' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['analog_args']['amp'], 'UNIT': "AMPLITUDE_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_initialtime' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['analog_args']['t_o'], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_pulsewidth' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['analog_args']['width'], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_timepulsewindow' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['analog_args']['tp_window'], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented

                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_sampling' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['sampling'], 'UNIT': "OTHER_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_carrierfrequency' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['sampling'], 'UNIT': "FREQUENCY_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[_idx_instruction][1]['identifier']+'_bitdepth' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['sampling'], 'UNIT': "OTHER_UNIT"}}); # random nr can be better implemented

                        #storing specs for analogs
                        elif waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE']=="ANALOGS":

                            if save_waveforms==True:
                                _waveform = [_WAVEFORM.tolist() for _WAVEFORM in waveform.digiWaveformStack];
                            else:
                                _waveform = "None"

                            _inst_type_ = ' ANALOGS';
                            _instruction_name_i_ = (waveform._API_sequence[_idx_instruction][1]['name']+_inst_type_).replace(" ", "_")
                            
                            _instruction_name_list_.append(_instruction_name_i_);
                            
                            _spec_inst_i_ = {_instruction_name_i_ : {"ID": hex(random.randint(1e4, 1.5e4)),
                                                                     "TYPE": waveform._API_sequence[_idx_instruction][1]['API_instruction_TYPE'],
                                                                     "CHANNEL": waveform._API_sequence[_idx_instruction][1]['identifier'],
                                                                     "WAVEFORM": str(_waveform),
                                                                     "TIME_FLAGS": waveform._API_sequence[_idx_instruction][1]['timeflags'][0],
                                                                     "COMMAND": waveform._API_sequence[0][0].specs['driver']['analog_cmd'],
                                                                     "SPECS": {"name": waveform._API_sequence[_idx_instruction][1]['name'], "args":aqipt.remove_keywords_from_dict(waveform._API_sequence[_idx_instruction][1], ['API_instruction_TYPE', 'identifier', 'dyn_params', 'name'])},
                                                                     "METADATA": "Analog load from script."}};
                            self.specifications['instructions'].append(_spec_inst_i_); #add instruction
                            if variables == None:
                                _variables_temp_dict.append({waveform._API_sequence[0][1]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['timeflags'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented
                                _variables_temp_dict.append({waveform._API_sequence[0][1]['identifier']+'_timeflags' : {'ID': hex(random.randint(500e3, 800e3)), 'value': waveform._API_sequence[_idx_instruction][1]['timeflags'][0], 'UNIT': "TIME_UNIT"}}); # random nr can be better implemented

                    _spec_seq_i_ = {waveform.label:{"ID": hex(random.randint(300, 500)),
                                                    "active":"True", 
                                                    "INSTRUCTIONS":_instruction_name_list_}};
                    self.specifications['sequences'].append(_spec_seq_i_); #add sequence


                # elif isinstance(waveform, control.track):

                #     try: #try tracks

                #         _instruction_name_list_ = [];

                #         for _idx_instruction in range(len(waveform._API_sequence)):

                #             if "risetime" in waveform._API_sequence[_idx_instruction][1]:

                #                 _instruction_name_list_.append((waveform._API_sequence[_idx_instruction][1]['name']+' DIGITAL').replace(" ", "_"));
                            
                #             else:

                #                 _instruction_name_list_.append((waveform._API_sequence[_idx_instruction][1]['name']+' ANALOG').replace(" ", "_"));


                #         _spec_seq_i_ = {waveform.label:{"ID": str(random.randint(500, 700)),
                #                                     "active":"True", 
                #                                     "INSTRUCTIONS":_instruction_name_list_}};
                #         self.specifications['tracks'].append(_spec_seq_i_); #add track

                #     except:
                #         pass

            self.specifications['variables'] = _variables_temp_dict;

            with open("software_specs.json", "w") as jsonFile:
                json.dump(self.specifications, jsonFile, indent=4, separators=(',', ': '))

        #updating variables
        if variables!=None:
            pass

        #updating variables units
        if variables_UNITS!=None:
            pass

        #updating instructions
        if instructions!=None:
            pass

        #updating configuration
        if configuration!=None:
            pass

    def printHeader(self):
        print('{0} | {1} | {2} '.format(self._author, self._affiliation, self._website))
        print('Created: {0} \n'.format(self._dateCreation))
        print('Comments: {0}\n'.format(self._comments))


#####################################################################################################
#variables AQiPT class
#####################################################################################################

class variables(object):

    '''
        AQiPT class that use the same JSON parser of instructions protocol to extract variables values
        and create a new object for non-default values during the experiment. Can be resave in the json
        file.
        
        
        ATTRIBUTES:
        -----------
            attribute :
            
        
        METHODS:
        --------
            method :

    '''

    def __init__(self, name='Default', comments='None', author='AQiPT', affiliation='Default affiliation', website='Default website'):
        
        self.name = name;
        self._author = author;
        self._affiliation = affiliation;
        self._website = website;
        self._dateCreation = date.today().isoformat();
        self._comments = comments;
        self.units = None;
        self._RAWvariables = dict();
    
    def add_variable(self, name, value, id_val=None, unit='TIME_UNIT'):
        
        try:
            setattr(self, name, value);
            if id_val=='0x0':
                while True:
                    id_val = aqipt.rand_hex(ndigits=3);
                    # print(id_val)
                    not_present = all(id_val not in elem['ID'] for elem in self._RAWvariables.values());
                    # print(not_present)
                    if not_present:
                        break

            self._RAWvariables.append({name : {'ID': id_val, 'value': value, 'UNIT': unit}});
        except:
            print('Not valid variable')
            pass

    def refresh_VARvalue(self, VARname, VARvalue):

        self._RAWvariables[VARname]['value'] = VARvalue;
        exec('self.'+VARname+' = VARvalue');

    def refresh_VARunit(self, VARname, VARunit):

        self._RAWvariables[VARname]['UNIT'] = VARunit;

    def load_fromSpecs(self, path=directory.compiler_dir, printON=False):
        
        #load variables from software_specs JSON file
        os.chdir(path)
        with open('software_specs.json', 'r') as specs_file:
            software_specs = json.load(specs_file);
            if printON:
                print(json.dumps(software_specs['variables'][0], indent=1, sort_keys=True))

            try:
                self._RAWvariables = software_specs['variables'][0];


                for key, val in zip(software_specs['variables'][0].keys(), software_specs['variables'][0].values()):
                    self.add_variable(key, val)
            except:
                self._RAWvariables = software_specs['variables'];

            self.units = software_specs['variables_UNITS'];

        pass  

    def printHeader(self):
        print('{0} | {1} | {2} '.format(self._author, self._affiliation, self._website))
        print('Created: {0} \n'.format(self._dateCreation))
        print('Comments: {0}\n'.format(self._comments))


#####################################################################################################
#channelsBench AQiPT class
#####################################################################################################

class IDsBench(object):

    '''
       ChannelsBench is an class that contains all the channels IDs plus the commands that can be 
       executed from the device driver of that channel.


        ATTRIBUTES:
        -----------
            _HW_bench : bench for the IDs of the hardware elements
            _SW_bench : bench for the IDs of the software elements
            _CH_bench : bench for the IDS of the channels 
            _AQiPT_bench : bench ofr the IDs reserved for AQiPT
            
        
        METHODS:
        --------
            set_SW_bech : generates the hardware and channels ID benchs
            set_SW_bech : generates the software ID bench
            get_HW_bench : returns the hardware ID benches
            get_CH_bench : returns the channels ID benches
            get_SW_bench : returns the software ID benches

    '''

    def __init__(self):
        
        self._HW_bench: List[IDs] = [];
        self._CH_bench: List[IDs] = [];
        self._SW_bench: List[IDs] = [];

        self._AQiPT_bench: List[IDs] = [];


    def _add_record(self, _name, bench, _hex_ID: str, _python_memory_alloc: str, _subTYPE: str):

        '''
            Add a new record/element of dataclass type (IDs), given the bench, the hexadecimal, python memory allocation and subtype
            of device is or coming from.
        '''
        
        _record = IDs(name=_name, hex_ID=_hex_ID, dec_ID=int(_hex_ID, base=16), python_memory_alloc=_python_memory_alloc, subTYPE=_subTYPE);

        bench.append(_record);

    def set_HW_bench(self, specifications):

        '''
            Set the bench for the HW and at the same time the channels located in the hardware. This is used by the specification 
            parser of the API.
        '''

        for TYPE in specifications.hardwareTypes:

            for k_type in range(len(specifications.hardwareLST[str(TYPE)])):
                
                #hardware bench
                __name = specifications.hardwareLST[str(TYPE)][k_type].specs[str(TYPE)]; #name
                __hex_ID = specifications.hardwareLST[str(TYPE)][k_type].specs['ID']; #id registered in the HW specifications JSON file
                __python_memory_alloc = hex(id(specifications.hardwareLST[str(TYPE)][k_type])); #python memory allocation of hardware in HW_specs
                __subTYPE = str(TYPE); #type of the device it belongs

                self._add_record(_name=__name, bench=self._HW_bench, _hex_ID=__hex_ID, _python_memory_alloc=__python_memory_alloc, _subTYPE=__subTYPE); #creating and adding the new record of the HW dataclass

                #channels bench
                try:

                    i=0;
                    for ch_id in specifications.hardwareLST[str(TYPE)][k_type].specs['properties']['channels_IDs']:
                        
                        __name = str(TYPE) + '_' + str(specifications.hardwareLST[str(TYPE)][k_type].specs[str(TYPE)]) + '_CHANNEL_'+str(i);
                        __CH_hex_ID = str(ch_id); #id registered in the HW specifications JSON file
                        __CH_python_memory_alloc = hex(id(specifications.hardwareLST[str(TYPE)][k_type])); #python memory allocation of hardware in HW_specs
                        __CH_subTYPE = str(TYPE) + ' : ' + str(specifications.hardwareLST[str(TYPE)][k_type].specs[str(TYPE)]); #type of the device it belongs

                        self._add_record(_name=__name, bench=self._CH_bench, _hex_ID= __CH_hex_ID, _python_memory_alloc= __CH_python_memory_alloc, _subTYPE= __CH_subTYPE); #creating and adding the new record of the CH dataclass
                        i+=1;
                except:
                    pass

    def set_SW_bench(self, specifications):
        '''
            Set the bench for the SW. This is used by the specification parser of the API.
        '''
        for HEAD in specifications._headers:

            try:

                for keyword, value in specifications.specifications[str(HEAD)][0].items():

                    #software bench
                    __name = keyword; #name of the variable
                    __hex_ID = value['ID']; #id registered in the HW specifications JSON file
                    __python_memory_alloc = hex(id(specifications)); #python memory allocation of hardware in HW_specs
                    __subTYPE = str(HEAD) + ':' + str(keyword); #type of the device it belongs


                    self._add_record(_name=__name, bench=self._SW_bench, _hex_ID=__hex_ID, _python_memory_alloc=__python_memory_alloc, _subTYPE=__subTYPE); #creating and adding the new record of the HW dataclass

            except:
                pass

            try:

                for k_type in specifications.specifications[str(HEAD)][0]:

                    #software bench
                    __name = 'Default';
                    __hex_ID = k_type['ID']; #id registered in the HW specifications JSON file
                    __python_memory_alloc = hex(id(specifications.specifications[str(HEAD)][0])); #python memory allocation of hardware in SW_specs
                    __subTYPE = str(HEAD) + ':' + str(list(mydict.keys())[1]); #type of the device it belongs

                    self._add_record(_name=__name, bench=self._SW_bench, _hex_ID=__hex_ID, _python_memory_alloc=__python_memory_alloc, _subTYPE=__subTYPE); #creating and adding the new record of the SW dataclass
            except:
                pass
    
    def get_HW_bench(self):
        '''
            Returns the Hardware bench of IDs.
        '''

        return self._HW_bench

    def get_SW_bench(self):
        '''
            Returns the Software bench of IDs.
        '''
        return self._SW_bench

    def get_CH_bench(self):
        '''
            Returns the Channels bench of IDs.
        '''
        return self._CH_bench

@dataclass(frozen=False)
class IDs:
    '''
        Special dataclass of AQiPT that stores the IDs of the channels of software, hardware, atom and other instances.
        
        ATTRIBUTES/FIELDS:
        ------------------
        
            hex_ID : hexadecimal ID value
            dec_ID : decimal ID value
            python_memory_alloc : Python memory allocation
            subTYPE : sub-type of the ID e.g., awg, aom, etc

        METHODS:
        --------

        None
        
        
    '''
        
    #fields
    name : str = field(default="Default")
    hex_ID : str = field(default="0x0")
    dec_ID : str = field(default="0x0")
    python_memory_alloc :str = field(default="0x0")
    subTYPE : str = field(default="Default")




