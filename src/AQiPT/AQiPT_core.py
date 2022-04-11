#Atomic Quantum information Processing Tool (AQIPT) - General module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): S.Whitlock. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                 F.Rayment.
# Created: 2021-04-08
# Last update: 2022-02-07


#libs
import numpy as np
import qutip as qt

def gen_random():
    return np.random.random()

def QME_scan(H_tot, psi0, times, cops, mops, opts):
    '''
        Quantum master equation scan solver

        Given a list of Hamiltonians and fix parameters for initial state,
        collapse operators, measure operators etc, it can solve and save the 
        simulations of a QME.
    '''

    i=0;
    for H in H_tot:
        result = qt.mesolve(H, psi0, times, cops, mops, options=opts);
#         result_lst.append(result);
        qt.qsave(result,'det-'+str(i)); #storing result
        i+=1;

#function for QME solver   
def QME_sol(H, psi0, times, cops, mops, i, opts):
    result = qt.mesolve(H, psi0, times, cops, mops, options=opts)
    qt.qsave(result,'det-'+str(i)); #storing result
    
#digitize function
def digitize(data, bitdepth, bottom, top):  #Finn & Shannon's code
    d = np.clip(data, bottom, top);
    a = top-bottom;
    return (np.round(((d/a)-bottom)*(2**bitdepth-1))/(2**bitdepth-1)+bottom)*a

#converter from time to index
def time2index(time, times):
    sampling_rate = len(times)
    t_i = times[0]; t_f = times[len(times)-1];
    
    if t_i<t_f:
        try:
            return int(time*sampling_rate/(t_f- t_i))
        except:
            pass
    elif t_f<t_i:
        try:
            return int(time*sampling_rate/abs(t_i-t_f))
        except:
            pass


#####################################################################################################
#general_params AQiPT class
#####################################################################################################
class general_params():
    '''
        general_params class wrap a few of the parameters that can be used in the 'control' and
        'emulator' modules.


        Parameters
        ----------
        args : dict
            Data with necessary arguments linked to the waveform device producers and dynamic simulations

        Attributes
        ----------
        _data : dict
            Data with specifications of waveforms.
        sampling : int
            Sampling rate of waveforms.
        bitdepth : int
            Digitization level for the waveform generation.
        dyn_time : array

        Methods
        -------
        getData()
            Returns full data of the class
        timebase()
            Returns time base (time domain) generated

    '''

    def __init__(self, args):

        '''
            Constructor of the general_params() object of AQiPT.
        '''

        self._data = args;

        #experimental atributes/params
        self.sampling = args['sampling'];
        self.bitdepth = args['bitdepth'];

        #dynamic atributes/params
        self.dyn_time = args['time_dyn'];

    def getData(self):
        '''
            Returns full data in the object
        '''
        return self._data

    def timebase(self):
        '''
            Generates and returns base time for waveforms and simulation.
        '''
        self._tbase = np.linspace(0, self.dyn_time, self.sampling)
        return self._tbase
