#Atomic Quantum information Processing Tool (AQIPT) - DAQ module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2021-04-08
# Last update: 2022-01-11


#libs
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# from functools import reduce
# import itertools
# from typing import Iterator, List
import copy

# import warnings
# warnings.filterwarnings('ignore')

from tqdm import tqdm

# from numba import jit
# import numba
import os, time, dataclasses

import AQiPTc as aqipt

import arc
from scipy.constants import physical_constants
from scipy.constants import e as C_e

import pandas as pd

'''
	TO DO LIST
	----------

	    -Finish other properties of atoms such C3, Dip.Mat.Element
	    -Include other sourcers like PairStates python package
	    -Include other formats to export the data e.g., '.qg8'
	    -Add extra constrains in the extraParams for including specification of the lasers for the different transitions (at the moment all transitions use same power, polarization and waist.)
'''

#####################################################################################################
#atomSpecs AQiPT (data)class
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

def lst2dic(ket):
    
    if type(ket) is list:
        
        ket_lst={}; nr_state=0;
        
        for state in ket:

            l_val = ''.join([n for n in state if n.isupper()]);
            n_val = state.rpartition(l_val)[0];
            j_val = state[state.find('_')+len('_'):state.rfind(',')];
            s_val = state[state.find(',')+len(',')+1:state.rfind('')];
            ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)})
            nr_state+=1;
        return ket_lst

    else:
        
        l_val = ''.join([n for n in ket if n.isupper()])
        n_val = ket.rpartition(l_val)[0]
        j_val = ket[ket.find('_')+len('_'):ket.rfind(',')]
        s_val = ket[ket.find(',')+len(',')+1:ket.rfind('')]
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
    
    def __init__(self, Atom=None, States=None, specs=None, metadata=None, extraParams=None):

        self.atom = Atom; 
        self._atomName = self.atom.elementName; 
        self.states = States;
        self._eParams = extraParams;
        self._atomicdata2save = None
        
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
        return  self.atom.getRabiFrequency(n1, l1, j1, mj1, n2, l2, j2, polarization, power, waist)*1e-6

    def getSMap_fromARC(self, istate, nrange, lmax, Bz, Erange, N, progressBar=True, unit=1, HighlightState=True, HighlightColour='red'):

        #create Stark map object
        sMap = arc.StarkMap(atom);

        sMap.defineBasis(istate['n'], istate['l'], istate['j'], istate['s'], nrange[0], nrange[1], lmax, Bz, progressOutput=progressBar, s=0.5); #initialise Basis States for Solver : progressOutput=True gives verbose output

        #generate Stark map
        sMap.diagonalise(np.linspace(Erange[0], Erange[1], N), progressOutput=progressBar);

        #show Stark map
        sMap.plotLevelDiagram(progressOutput=progressBar, units=unit, highlightState=HighlightState, highlightColour=HighlightColour);

        return sMap

    def getLifetime_fromARC(self, istate):
        return  atom.getStateLifetime(istate['n'], istate['l'], istate['j'])

    def isRydberg(self, istate):
        if istate['n']>20:
            return True
        else:
            return False

    def getBBR_fromARC(self, nMin, nMax, lj1, lj2, minTemp=0.1, maxTemp=300):

        vals = {'valsn': [], 'valsNoBBR': [], 'valsBBR': [], 'figure': None}

        for n in range(nMin,nMax):

            noBBR = atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1], temperature=minTemp)\
                    +atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1]+lj2[0], temperature=minTemp);
            withBBR =  atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1], temperature=maxTemp)\
                    +atom.getTransitionRate(nMax, lj1[0], lj1[1], n, lj2[0], lj2[1]+lj2[0], temperature=maxTemp)
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
        calc = arc.PairStateInteractions(atom, sMap.n, sMap.l, sMap.j, sMap.n, sMap.l, sMap.j, sMap.s, sMap.s)

        c6=[];

        for t in theta:
            C6 = calc.getC6perturbatively(t, phi, dn, dEmax)
            c6.append(abs(C6))

        if plotON==True:
            ax = plt.subplot(111, projection='polar');
            ax.set_theta_zero_location("N");
            line = [];

            # plot results
            lineLegend, = plt.plot(theta,c6,"-",color="r",label=("mj=%d/2"%int(2*mj0)) )
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
        calc = arc.PairStateInteractions(atom, sMap.n, sMap.l, sMap.j, sMap.n, sMap.l, sMap.j, sMap.s, sMap.s)
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
  
    def set_atomicData(self, source='ARC'):
        '''
            Should go pair by pair of states and generate the datacell() to ultimately create atomicData() dataclass that belongs
        to the atomSpecs class
        '''
        _labels = []; _cells = [];
        _cells2save=[];
        if source=='ARC':
            states_dict = lst2dic(self.states);
            for i in tqdm(states_dict):
                istate = states_dict[str(i)]; #istate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>}
                for j in tqdm(states_dict, leave=False):
                    jstate = states_dict[str(i)]; #istate as dict like {'n': <val>, 'l': <val>, 'j': <val>, 's': <val>}
                    
                    _labels.append('|'+str(i)+'X'+str(j)+'|'); #adding labels for atomicData dataclass
                    
                    
                    #gettting data from ARC
                    __TransitionFrequency = self.getTransitionFrequency_fromARC(istate, jstate);
                    
                    try:
                        __TransitionWavelenght = self.getTransitionWavelength_fromARC(istate, jstate);
                        
                        if __TransitionWavelenght == inf:
                            __TransitionWavelenght = None
                    except:
                        __TransitionWavelenght = None;
                
                    __RabiFrequency = self.getRabifreq_fromARC(istate, jstate, self._eParams['4RabiFreq']['polarization'], self._eParams['4RabiFreq']['power'], self._eParams['4RabiFreq']['waist']);
                    
                    __StarkMap = self.getSMap_fromARC(istate, self._eParams['4SMap']['nrange'], self._eParams['4SMap']['lmax'], self._eParams['4SMap']['bz'], self._eParams['4SMap']['erange'], self._eParams['4SMap']['n'], self._eParams['4SMap']['progressbar'], self._eParams['4SMap']['unit'], self._eParams['4SMap']['highlightstate'], self._eParams['4SMap']['highlightcolour']);

                    __NaturalLifetime = self.getLifetime_fromARC(istate);
                    
                    __RydbergState = self.isRydberg(istate);
                    
                    try:
                        __BlackBodyRadiation = self.getBBR_fromARC(istate['n']-self._eParams['4BBR']['dn'], istate['n']+self._eParams['4BBR']['dn'], self._eParams['4BBR']['lj_vals'][0], self._eParams['4BBR']['lj_vals'][1], self._eParams['4BBR']['mintemp'], self._eParams['4BBR']['maxtemp']);
                    except:
                        __BlackBodyRadiation = None;
                    
                    if __RydbergState is True:
                        __C6 = self.getC6_fromARC(__StarkMap, self._eParams['4C6']['phi'], self._eParams['4C6']['theta'], self._eParams['4C6']['dn'], self._eParams['4C6']['dEmax'], self._eParams['4C6']['ploton'] );

                        __BlockadeRadius = self.getRb_fromARC([__RabiFrequency], __C6[0]);
                    else:
                        __C6 = None;
                        __BlockadeRadius = None;
                        
                    __LeRoyRadius = self.getLeRoyRadius_fromARC( __StarkMap);
                    
                    try:
                        __Polarizability = self.getPolarizability_fromARC(__StarkMap, __TransitionWavelenght);
                    except:
                        __Polarizability = None;
                        
                    #NotImplemented
                    __Energy = None;
                    __C3 = None;
                    __Dip_Mat_element = None;
                    
                    #datacell for python object
                    _cell = datacell(__Energy,
                                     __TransitionFrequency,
                                     __TransitionWavelenght,
                                     __RabiFrequency,
                                     __Dip_Mat_element,
                                     __NaturalLifetime,
                                     __RydbergState,
                                     __BlackBodyRadiation,
                                     __C3,
                                     __C6,
                                     __BlockadeRadius,
                                     __LeRoyRadius,
                                     __Polarizability,
                                     __StarkMap,);
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
                    _cell2save = datacell(__Energy,
                                     __TransitionFrequency,
                                     __TransitionWavelenght,
                                     __RabiFrequency,
                                     __Dip_Mat_element,
                                     __NaturalLifetime,
                                     __RydbergState,
                                     _BBRCOPY,
                                     __C3,
                                     _c6COPY,
                                     __BlockadeRadius,
                                     __LeRoyRadius,
                                     __Polarizability,
                                     extract_SMapData(__StarkMap));
                    _cells2save.append(_cell2save);
                    
            print("There are "+str(len(_cells))+" cells")
            
            self._metadata.update({'cell_labels' : str(_labels)})
            self.atomicdata = atomicData(clabel=_labels, cells=_cells )
            self._atomicdata2save = atomicData(clabel=_labels, cells=_cells2save )

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
    Energy : float = field(default=0.0, metadata={'unit': 'degrees', 'value':None})
    Transition_freq : float = field(default=0.0, metadata={'unit': 'degrees'})
    Wavelength : float = field(default=0.0, metadata={'unit': 'degrees'})
    Rabi_freq : float = field(default=0.0, metadata={'unit': 'degrees'})
    Dip_Mat_element : float = field(default=0.0, metadata={'unit': 'degrees'})
    Lifetime : float = field(default=0.0, metadata={'unit': 'degrees'})
    Rydberg : bool = field(default=False, metadata={'unit': 'degrees'})
    BBR : float = field(default=0.0, metadata={'unit': 'degrees'})
    C3 : float = field(default=0.0, metadata={'unit': 'degrees'})
    C6 : float = field(default=0.0, metadata={'unit': 'degrees'})
    Rbl : float = field(default=0.0, metadata={'unit': 'degrees'})
    LeRoy_radius : float = field(default=0.0, metadata={'unit': 'degrees'})
    Polarizability : float = field(default=0.0, metadata={'unit': 'degrees'})
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
    
    if type(ket) is list:
        
        ket_lst={}; nr_state=0;
        
        for state in ket:

            l_val = ''.join([n for n in state if n.isupper()]);
            n_val = state.rpartition(l_val)[0];
            j_val = state[state.find('_')+len('_'):state.rfind(',')];
            s_val = state[state.find(',')+len(',')+1:state.rfind('')];
            ket_lst[str(nr_state)]=({'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)})
            nr_state+=1;
        return ket_lst

    else:
        
        l_val = ''.join([n for n in ket if n.isupper()])
        n_val = ket.rpartition(l_val)[0]
        j_val = ket[ket.find('_')+len('_'):ket.rfind(',')]
        s_val = ket[ket.find(',')+len(',')+1:ket.rfind('')]
        return {'n':int(n_val), 'l':orb2Int(l_val), 'j':convert_to_float(j_val), 's':convert_to_float(s_val)}


    
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