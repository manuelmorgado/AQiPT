#Atomic Quantum information Processing Tool (AQIPT) - Control module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): S.Whitlock, S.Bera, S.Yang. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2022-02-07


#libs
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats
import scipy.signal as signal

#####################################################################################################
#function AQiPT class
#####################################################################################################
class function:
    """
        A class for representing easy function objects, such as square pulses, 
        sine, gaussian pulses etc.

        The function class is the AQiPT representation of functions for create waveforms.
        This class also show plots in matplotlib, as well as export the functions to numpy arrays.


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Attributes
        ----------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.

        Methods
        -------
        step()
            Conjugate of quantum object.
    """
    def __init__(self, times, args):
        
        #atributes
        self.tbase = times
        self._res = type
        self.args = args
        
    def step(self, plotON=False):
        '''
            Basic step function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., start time, step width)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                times = np.linspace(0, 1, 100)

                to = 0.2;
                fwidth = 0.1;
                args = {'t_o':to, 'width': fwidth};

                plt.plot(times, step(times, args))
            '''

        function = self.args['amp'] * (abs(self.tbase-self.args['t_o']) < self.args['width']);



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)
    
    def ramp(self, plotON=False):
        '''
            Basic ramp function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., start time, step width)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                    times = np.linspace(0, 1, 100)

                    to = 0.2;
                    fwidth = 0.1;
                    args = {'t_o':to, 'width': fwidth};

                    plt.plot(times, step(times, args))

        '''

        function = self.args['m'] * self.tbase + self.args['b'];



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot
    
        else:

            return function, plt.plot(self.tbase, function)

    def parabola(self, plotON=False):
        '''
            Basic Parabola function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., polynomia coefficients, shift)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                times = np.linspace(-75, 75, 500); #time domain function

                args = {'a':1, 'b': 0, 'c':0, 'shift':0}; #arguments for function

                f, fplot = parabola(times, args);
                plt.show(fplot)

        '''

        function = self.args['a']*(self.tbase-self.args['shift'])**2 + self.args['b']*self.tbase + self.args['c'];



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)
        
    def sinusoidal(self, plotON=False):
        '''
            Basic sinusoidal function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., amplitude, frequency, phase)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                times = np.linspace(0, 1, 500); #time domain function

                args = {'Amp':1, 'freq':5, 'phase':0}; #arguments for function

                f, fplot = sinusoidal(times, args);
                plt.show(fplot)

        '''

        function = self.args['Amp']*np.sin(2 * np.pi * self.args['freq'] * self.tbase + self.args['phase']);



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)
        
    def gaussian(self, plotON=False):
        '''
            Basic Gaussian function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., amplitude, center, standar deviation)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                    times = np.linspace(0, 1, 100)

                    amp = 1.0; #gaussian amplitude
                    center = 0.0; #gaussian center
                    std = 0.1; #standard deviation
                    args = {'g_Amp':amp, 'g_center': center, 'g_std':std}; #arguments for function

                    plt.plot(times, gauss(times, args))

        '''

        function = self.args['g_Amp'] * np.exp( -( (self.tbase - self.args['g_center'])**2 / (4*(self.args['g_std'])**2) ));



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)
        
    def triangular(self, plotON=False):
        '''
        Basic triangular function

        INPUTS
            t (array): time domain of function
            args (dict): width of step function (i.e., amplitude, center, standar deviation)
            plotON (bool): shows (True) or not (False) plot

        OUTPUTS
            function: function ready for aqipt.waveform()
            fig: plot of function

        \Example:

            times = np.linspace(0, 3, 500); #time domain function

            tri_amp = 0; #gaussian amplitude
            tri_freq = 2; #gaussian center
            tri_phase = 0; #standard deviation
            args = {'tri_amp':tri_amp, 'tri_freq':tri_freq, 'tri_phase':tri_phase}; #arguments for function


            f, fplot = triangular(times, args)

        '''

        function = self.args['tri_amp'] *signal.sawtooth(2 * np.pi * self.args['tri_freq'] * self.tbase + self.args['tri_phase'], 0.5);



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)
    
    def sawtooth(self, plotON=False):
        '''
        Basic sawtooth function

        INPUTS
            t (array): time domain of function
            args (dict): width of step function (i.e., amplitude, center, standar deviation)
            plotON (bool): shows (True) or not (False) plot

        OUTPUTS
            function: function ready for aqipt.waveform()
            fig: plot of function

        \Example:

            times = np.linspace(0, 3, 500); #time domain function

            tri_amp = 0; #gaussian amplitude
            tri_freq = 2; #gaussian center
            tri_phase = 0; #standard deviation
            args = {'tri_amp':tri_amp, 'tri_freq':tri_freq, 'tri_phase':tri_phase}; #arguments for function


            f, fplot = sawtooth(times, args)

        '''

        function = self.args['tri_amp'] *signal.sawtooth(2 * np.pi * self.args['tri_freq'] * self.tbase + self.args['tri_phase'], 1);



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            return function, function_plot

        else:

            return function, plt.plot(self.tbase, function)



#####################################################################################################

#Pulse AQiPT class
#####################################################################################################
class pulse: #pulse(function)
    """
        A class for building pulses shapes in time domain starting from AQiPT functions() for later
        build the AQiPT tracks() that can be assigned to AQiPT producers() via AQiPT instructions().

        The pulse class is the AQiPT representation of shaped pulse for create tracks.
        This class also show plots in matplotlib, as well as export the pulses as numpy arrays.


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Attributes
        ----------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.

        Methods
        -------
        step()
            Conjugate of quantum object.
    """
    def __init__(self, pulse_label, times):
        
        #atributes
        self.label = pulse_label
        self.tbase = times
        self._res = type
        # self.args = args
        self.waveform = np.zeros(tbase.shape)
        self.digiWaveform = np.zeros(self.tbase.shape)

    def overwriteFunction(self, tstart, function):
        '''
            Basic step function

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., start time, step width)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                times = np.linspace(0, 1, 100)

                to = 0.2;
                fwidth = 0.1;
                args = {'t_o':to, 'width': fwidth};

                plt.plot(times, step(times, args))
        '''
        result = list(self.waveform);
        func = list(function)
        # result[:function.shape[0]] = function
        replace=lambda result,func,s:result[:s]+ func + result[s+len(func):]; #replace function the waveform of function in the tbase at i-point
        
        self.waveform = np.array(replace(result, func, int(tstart))); #replacing waveform of function
        self.waveform[tstart]= (self.waveform[tstart-1]+self.waveform[tstart+1])/2; #fixing point in 0 to the average of the i-1 th and i+1 th point

        return self.waveform
    
    
    
    def addfunction(self, tstart_index, function, kind=None):
        
        wf = self.waveform;
        if kind == 'Carrier':
        
            for i in range(len(function)):
                wf[i+tstart_index] = function[i];

        else:
            for i in range(len(function)):
                if function[i]>=wf[i+tstart_index]:
                    wf[i+tstart_index] = function[i];
                if function[i]<wf[i+tstart_index]: #
                    wf[i+tstart_index] = wf[i+tstart_index]+function[i]; #
                       
        self.waveform = wf; #/max(wf);
        return self.waveform
    
    def combineFunction(self, tstart, function):

        l = sorted((self.waveform, function), key=len)
        c = l[1].copy()
        c[tstart:tstart+len(l[0])] += l[0]
        self.waveform = c
        return self.waveform

    def mergeFunction(self, tstart, function):

        self.waveform+=function
        return self.waveform
    
    def getPulse(self):
        return self.waveform
    
    def getLabel(self):
        return self.label

    def digitizeWaveform(self, bitdepth, bottom, top):  #Finn & Shannon's code
        data = self.waveform
        d = np.clip(data, bottom, top);
        a = top-bottom;
        self.digiWaveform = (np.round(((d/a)-bottom)*(2**bitdepth-1))/(2**bitdepth-1)+bottom)*a

    def saveWaveform(awg_args:dict, wf_args_lst:list, waveforms_lst:list, fileformat='.csv', fname=None):
        if fileformat == '.csv':

            for i in range(len(waveforms_lst)):
                if fname==None:
                    metadata = ["waveformName," + str(wf_args_lst[i]['name']), "waveformPoints," + str(awg_args['sampling']-2), "waveformType,WAVE_ANALOG_16"]
                    filename = "waveforms_files/ "+ str(wf_args_lst[i]['name']) + fileformat;
                else:
                    metadata = ["waveformName," + fname, "waveformPoints," + str(awg_args['sampling']-2), "waveformType,WAVE_ANALOG_16"]
                    filename = "waveforms_files/ "+ fname + fileformat;
                with open(filename, 'w') as fout:
                    for line in metadata:
                        fout.write(line+'\n')

                    # np.savetxt(filename, (waveforms_lst[i]).astype(np.uint16) , delimiter=",")
                    np.savetxt(filename, waveforms_lst[i] , delimiter=",")
                    print(max(waveforms_lst[i]))
            print('Saved waveforms!')

#####################################################################################################

#Track AQiPT class
#####################################################################################################
class track(pulse):
    def __init__(self, track_label, tTrack, pulse_list=None):
        
        #atributes
        self._res = type
        self.label = track_label
        self.tTrack = tTrack
        self.pulseList = pulse_list
        self.digiWaveform = None
    
    def add2Track(self, pulse_objs):
        if self.digiWaveform is None:
            self.digiWaveform = np.concatenate(pulse_objs, axis=None)
            self.tTrack = np.zeros(len(self.digiWaveform ))
            self.pulseList = pulse_objs
        else:
            for pulse in pulse_objs:
                self.digiWaveform = np.concatenate((self.digiWaveform, pulse), axis=None)
                self.tTrack = np.zeros(len(self.digiWaveform ))
                self.pulseList.append(pulse_objs)

    def getTrack(self):
        return self.digiWaveform

    def getLabel(self):
        return self.label

    def getComposition(self):
        return self.pulseList

    def clearTrack(self):
        self.pulseList = None
        self.digiWaveform = None
        self.tTrack = None

#####################################################################################################
#Sequence AQiPT class
#####################################################################################################
class sequence(pulse):
    def __init__(self, sequence_label, tSequence, pulse_list=None):
        

        #atributes
        self._res = type
        self.label = sequence_label
        self.tSequence = tSequence
        self.pulseList = pulse_list
    
    def add2Sequence(self, pulse_objs):
        if self.pulseList is None:
            self.tSequence = np.zeros( len(pulse_objs[0]) )
            self.pulseList = pulse_objs
        else:
            for pulse in pulse_objs:
                self.pulseList.append(pulse)

    def getSequence(self):
        return self.pulseList

    def getLabel(self):
        return self.label

    def clearSequence(self):
        self.pulseList = None
        self.digiWaveform = None
        self.tSequence = None

#####################################################################################################
#Producer AQiPT class
#####################################################################################################
class producer():

    def __init__(self, name, kind, chnls_map, LANid, slave=True, statusMode=False):

        #atributes
        self._res = type
        self.name = name
        self.kind = kind
        self.slave = slave
        self.statusMode = statusMode
        self._nr_DIO_chnls = chnls_map['DIO']
        self._nr_DAC_chnls = chnls_map['DAC']

        self._DIO_chnl = [self.DIOchnl() for _ in range(self._nr_DIO_chnls)]
        self._DAC_chnl = [self.DACchnl() for _ in range(self._nr_DAC_chnls)]
    
    # def _initChannels():

    # def resetChannel():

    # def triggerChannel():

    # def idleMode():

    # class DIOchnl:
        
    #     def __init__(self):
    #         self._memADDRESS =  
    #         self.sequences =
    #         self.tracks = 
    #         self.IQtype = 
    #         self.status = 

    # class DACchnl:

    #     def __init__(self):
    #         self._memADDRESS =  
    #         self.sequences =
    #         self.tracks = 
    #         self.IQtype = 
    #         self.status = 

