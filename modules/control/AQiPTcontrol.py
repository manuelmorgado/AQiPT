#Atomic Quantum information Processing Tool (AQIPT) - Control module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): S.Whitlock, S.Bera, S.Yang. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2023-06-17


#libs
import warnings#, os
import datetime, time

import ctypes

from sched import scheduler
import inspect, queue

from dataclasses import dataclass, field
 
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as plty

import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import chirp

from AQiPT import AQiPTcore as aqipt
import AQiPT.modules.kernel.AQiPTkernel as kernel

import pandas as pd

from AQiPT.modules.directory import AQiPTdirectory as dirPath

directory = aqipt.directory;

#####################################################################################################
#Function AQiPT class
#####################################################################################################
class function:
    """
        A class for representing easy function objects, such as square pulses, 
        sine, gaussian pulses etc.

        The function class is the AQiPT representation of functions for create waveforms.
        This class also show plots in matplotlib, as well as export the functions to numpy arrays.


        Parameters
        ----------
        times : numpy.array
            Values of the time interval where the function is evaluated.
        args : dict
            Dictionary with all arguments required by the function.
        area : int
            Area under the function i.e., integral of the function.
        

        Attributes
        ----------
        tbase : array_like
            Values of the time interval where the function is evaluated. 
        _res : type
            Resolution of the waveform.
        arg : dict
            Dictionary with all the arguments required.
        waveform : numpy.array
            Values of the signal or waveform of the function object.
        area : int
            Area under the function i.e., integral of the function.

        
        Methods
        -------
        setArea(area)
            Set area of the function. NOT IMPLEMENTED.
        getFunction()
            Returns the value of the waveform of the function object.
        step()
            Define a step() function $f(t) = \text{H}(t+t_{o})-\text{H}(t_{o}-t) $
        ramp()
            Define a ramp function $f(t) = m\dot t + b$
        parabola()
            Define a parabola function $f(t) = a\dot t^{2} + b\dot t + c$
        sinusoidal()
            Define a sinusoidal function $f(t) = A \dot sin(\omega \dot t + \phi)$
        gaussian()
            Define a gaussian function $f(t) = A \dot exp(((t-t_{o})/(\sigma))^2)
        triangular()
            Define a periodical triangular function.
        sawtooth()
            Define a cutted-triangular or sawtooth periodical function.
    """
    def __init__(self, times, args, area=None):
        
        #(times=None, args=None, area=np.pi)
        #atributes
        self.tbase = times
        self._res = type
        self.args = args
        self.waveform = None;
        self.area = None;
        self._plot = None;

    def setArea(self, area=None):
        '''
            Set area of the function (integral) e.g., $\pi$, $\pi/2$ etc

            Input:
            ------

            area : int
                Area of the pulse.
        '''
        pass

    def getFunction(self):
        '''
            Get value of the waveform of the function.
        '''
        return self.waveform
        
    def step(self, plotON=False):
        '''
            Basic step function

            INPUTS:
                args : dict
                    Via function instance. 

                    {'amp': <<float>>, 't_o': <<float>>, 'width': <<float>>, 'tp_window': <<float>>, 'name': <<str>>, 'color': <<str>>, 'type': <<str>>}


            OUTPUTS:
                function : AQiPT.control.function
                    AQiPT function control-class
                fig : matplotlib.figure
                    Plot of function waveform

            \Example:

                times = np.linspace(0, 1, 100)

                to = 0.2;
                fwidth = 0.1;
                args = {'amp': Amp, 't_o':to, 'width': fwidth};

                plt.plot(times, step(times, args))
        '''

        function = self.args['amp'] * (abs(self.tbase-self.args['t_o']) < self.args['width']);

        self.waveform = function;

        if plotON==True:

            
            function_plot = self.plotFunction(Hunits='[$\mu s$]', Vunits='[V]');
            self._plot = function_plot;

            return self.waveform, None
        else:            
            return function, self._plot
    
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

            self.waveform = function;
            return function, function_plot
    
        else:

            self.waveform = function;
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

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
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

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
            return function, plt.plot(self.tbase, function)

    def chirp(self, plotON=False):
        '''
            Frequency chirp function (sinusoid with frequency ramp modulation)

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
        if self.args['method'] == 'linear':
            function = self.args['amp']*np.cos( 2*np.pi*((self.args['f_start']*self.tbase) + ((self.args['f_end']-self.args['f_start'])/(2*self.args['duration']))*self.tbase**2))
        if self.args['method'] == 'quadratic':
            function = self.args['amp']*chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'logarithmic':
            function = self.args['amp']*chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'exponential':
            function =self.args['amp']* chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])



        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
            return function, plt.plot(self.tbase, function)

    def chirplet(self, plotON=False):
        '''
            Frequency chirplet function (sinusoid with double frequency ramp modulation)

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

        if self.args['method'] == 'linear':
            function = self.args['amp']*np.concatenate((np.cos( 2*np.pi*((self.args['f_start']*self.tbase) + ((self.args['f_end']-self.args['f_start'])/(2*self.args['duration']))*self.tbase**2)), np.cos( 2*np.pi*((self.args['f_end']*self.tbase) + ((self.args['f_start']-self.args['f_end'])/(2*self.args['duration']))*self.tbase**2))), axis=None)
        if self.args['method'] == 'quadratic':
            function = self.args['amp']*chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'logarithmic':
            function = self.args['amp']*chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'exponential':
            function = self.args['amp']*chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])

        self.tbase = np.linspace(min(self.tbase), max(self.tbase), len(self.tbase)*2);


        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
            return function, plt.plot(self.tbase, function)

    def gaussian_chirplet(self, plotON=False):
        '''
            Frequency chirp function (sinusoid with double frequency ramp modulation and gaussian amplitude modulation)

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
        if self.args['method'] == 'linear':
            function = np.concatenate((np.cos( 2*np.pi*((self.args['f_start']*self.tbase) + ((self.args['f_end']-self.args['f_start'])/(2*self.args['duration']))*self.tbase**2)), np.cos( 2*np.pi*((self.args['f_end']*self.tbase) + ((self.args['f_start']-self.args['f_end'])/(2*self.args['duration']))*self.tbase**2))), axis=None)
            function =  self.args['amp']*(1/(self.args['sigma']*np.sqrt(2*np.pi)))*np.exp(-(args['t2']-self.args['g_center'])**2/(2*self.args['g_std']**2))*function
        if self.args['method'] == 'quadratic':
            function = chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'logarithmic':
            function = chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])
        if self.args['method'] == 'exponential':
            function = chirp(self.tbase, f0=self.args['f_start'], f1=self.args['f_end'], t1=self.args['duration'], method=self.args['method'])


        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            plt.show(function_plot)

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
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
        self.waveform = function;


        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            self._plot = function_plot;

            return self.waveform, None

        else:

            return function, self._plot
        
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

            self.waveform = function;
            return function, function_plot

        else:

            self.waveform = function;
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

        self.waveform = function;

        if plotON==True:

            
            function_plot = self.plotFunction(Hunits='[$\mu s$]', Vunits='[V]');
            self._plot = function_plot;

            
            return self.waveform, self._plot
        else:

            
            return function, self._plot

    def quadrature(self, plotON=False):
        '''
        Quadrature signal function

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

        function = self.args['amp']*np.exp(1j*(self.args['freq']*self.tbase + self.args['phase']));

        self.waveform = function;

        if plotON==True:

            
            function_plot = self.plotFunction(Hunits='[$\mu s$]', Vunits='[V]');
            self._plot = function_plot;

            
            return self.waveform, self._plot
        else:

            
            return function, self._plot

    def plotFunction(self, Hunits='[$\mu s$]', Vunits='[V]'):
        plt.figure();
        plt.plot(self.tbase, self.waveform);
        plt.xlabel(r'Time' + Hunits )
        plt.ylabel(r'Amplitude ' + Vunits)
        plt.text(0, max(self.waveform), s='function: '+self.args['name'],
                 size=15, rotation=0.,ha="center", va="center", color='white',
                 bbox=dict(boxstyle="round",
                           ec=(.0, 0.4, 0.8),
                           fc=(.0, 0.4, 0.8),) )
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
    def __init__(self, pulse_label, times, function_list=[], timeFlags=[]):
        
        #atributes
        self.tbase = times;
        self._res = type;

        self.label = pulse_label;
        self._compiled = False;
        self._digitized = False;

        self.waveform = np.zeros(self.tbase.shape);
        self.digiWaveform = np.zeros(self.tbase.shape);
        
        self._functionsLST = function_list;
        self._combfunctionLST = {'t_idx':[], 'function':[]};
        self._timeFlags = [aqipt.time2index(time, self.tbase) for time in timeFlags]; #timeFlags;
        # self._timeFlags = timeFlags;
        
    def overwriteFunction(self, tstart_index, function):
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
        
        self.waveform = np.array(replace(result, func, int(tstart_index))); #replacing waveform of function
        self.waveform[tstart_index]= (self.waveform[tstart_index-1]+self.waveform[tstart_index+1])/2; #fixing point in 0 to the average of the i-1 th and i+1 th point
        self._compiled = True;

        return self.waveform
    
    def add2Pulse(self, tstart_index=None, function=None, kind=None, _fromCompiler=False):
        
        if _fromCompiler==True: #TODO: fix the real+imag when run from compiler
            
            self._compiled = True;

            function_real = np.real(function.getFunction());
            function_imag = np.imag(function.getFunction());

            wf_real = np.copy(np.real(self.waveform));
            wf_imag = np.copy(np.imag(self.waveform));

            if kind == 'Carrier':

                for i in range(len(function_real)):
                    wf_real[int(i+tstart_index)] = function_real[i];
                    wf_imag[int(i+tstart_index)] = function_imag[i];

            else:

                for i in range(len(function_real)):

                    if function_real[i]>=wf_real[int(i+tstart_index)]:
                        wf_real[int(i+tstart_index)] = function_real[i];
                        wf_imag[int(i+tstart_index)] = function_imag[i];
                    if function_real[i]<wf_real[int(i+tstart_index)]:
                        wf_real[int(i+tstart_index)] = wf_real[int(i+tstart_index)]+function_real[i];
                        wf_imag[int(i+tstart_index)] = wf_imag[int(i+tstart_index)]+function_imag[i];


        else:
            function_real = np.real(function.waveform);
            function_imag = np.imag(function.waveform);

            wf_real = np.copy(np.real(self.waveform));
            wf_imag = np.copy(np.imag(self.waveform));

            self._functionsLST.append(function);
            self._timeFlags.append(tstart_index);

            if kind == 'Carrier':
            
                for i in range(len(function_real)):
                    wf_real[int(i+tstart_index)] = function_real[i];
                    wf_imag[int(i+tstart_index)] = function_imag[i];

            else:

                for i in range(len(function_real)):
                
                    if function_real[i]>=wf_real[int(i+tstart_index)]:
                        wf_real[int(i+tstart_index)] = function_real[i];
                        wf_imag[int(i+tstart_index)] = function_imag[i];
                    if function_real[i]<wf_real[int(i+tstart_index)]:
                        wf_real[int(i+tstart_index)] = wf_real[int(i+tstart_index)]+function_real[i];
                        wf_imag[int(i+tstart_index)] = wf_imag[int(i+tstart_index)]+function_imag[i];
                       

        self.waveform = wf_real + 1j*wf_imag; #/max(wf);
        self._compiled = True;

        return self.waveform
    
    def removeFunction(self, function):
        f_idx = self._functionsLST.index(function);

        self._functionsLST.remove(function);
        self._timeFlags.pop(f_idx);

        self.waveform = np.zeros(self.tbase.shape);
        print(hex(id(function)),'AQiPT function succesfully removed from pulse.')

        self.compilePulse();

    def sumFunction(self, tstart_index, function):

        l = sorted((self.waveform, function.waveform), key=len);
        c = l[1].copy();
        c[tstart_index:tstart_index+len(l[0])] += l[0];
        
        self.waveform = c;
        self._compiled = True;
        return self.waveform

    def combineFunction(self, tstart_index, function):
 
        l = sorted((self.waveform, function.waveform), key=len);
        c = l[1].copy(); 
        c[tstart_index:tstart_index+len(l[0])] *= l[0];

        self.digiWaveform = c;

        self._combfunctionLST['t_idx'].append(tstart_index);
        self._combfunctionLST['function'].append(function);
        self._compiled = True;

        self.waveform = c;

        return self.digiWaveform

    def mergeFunction(self, tstart_index, function):

        self._functionsLST.append(function);
        self._timeFlags.append(tstart_index);

        function = function.waveform;

        l = sorted((self.waveform, function), key=len);
        c = l[1].copy();
        c[tstart_index:tstart_index+len(l[0])] = l[0];

        self.waveform = c;
        self._compiled = True;
        return self.waveform
    
    def getPulse(self):
        if self._compiled == False:
            warnings.warn('Missing waveform. Please, compile pulse using: PULSENAME.compilePulse()')
        else:
            return self.digiWaveform
    
    def getLabel(self):
        return self.label

    def changeLabel(self, newLabel):
        self.label = newLabel;

    def plotPulse(self, axis=None, xlabel=str(), ylabel=str(), _color=None, figure_size=(15,4)):
        if axis==None:
            plt.figure(figsize=figure_size);
            if _color==None:
                plt.step(self.tbase, np.real(self.digiWaveform), alpha=0.9);
                plt.fill(self.tbase, np.real(self.digiWaveform), alpha=0.3);

                if isinstance(self.digiWaveform[0], np.complex):
                    plt.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    plt.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');

            else:
                plt.step(self.tbase, self.digiWaveform, alpha=0.9, color=_color);
                plt.fill(self.tbase, self.digiWaveform, alpha=0.3, color=_color);

                if isinstance(self.digiWaveform[0], np.complex):
                    plt.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    plt.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');

            plt.xlabel(xlabel);
            plt.ylabel(ylabel);
            plt.text(0, max(self.digiWaveform), s='pulse: '+self.label,
                     size=15, rotation=0.,ha="center", va="center", color='white',
                     bbox=dict(boxstyle="round",
                               ec=(.40, 0.4, 0.8),
                               fc=(.40, 0.4, 0.8),) )

            #TO-DO: add labels with correct units from AQiPTcore
        else:
            if _color==None:
                axis.step(self.tbase, np.real(self.digiWaveform), alpha=0.9);
                axis.fill(self.tbase, np.real(self.digiWaveform), alpha=0.3);

                if isinstance(self.digiWaveform[0], np.complex):
                    axis.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    axis.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');
            else:
                axis.step(self.tbase, np.real(self.digiWaveform), alpha=0.9, color=_color);
                axis.fill(self.tbase, np.real(self.digiWaveform), alpha=0.3, color=_color);

                if isinstance(self.digiWaveform[0], np.complex):
                    axis.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    axis.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');

            axis.set_xlabel(xlabel, fontsize=18);
            axis.set_ylabel(ylabel, fontsize=18);

            plt.text(0, max(self.digiWaveform), s='pulse: '+self.label,
                     size=15, rotation=0.,ha="center", va="center", color='white',
                     bbox=dict(boxstyle="round",
                               ec=(.40, 0.4, 0.8),
                               fc=(.40, 0.4, 0.8),) )
            #TO-DO: add labels with correct units from AQiPTcore
    
    def digitizeWaveform(self, bitdepth, minSignal, maxSignal):

        if self._digitized == True:
            data = self.digiWaveform;
        
        if self._digitized == False:
            data = np.real(self.waveform);

            d = np.clip(data, minSignal, maxSignal);
            a = maxSignal-minSignal;
            self.digiWaveform = (np.round(((d/a)-minSignal)*(2**bitdepth-1))/(2**bitdepth-1)+minSignal)*a;


            data = np.imag(self.waveform);

            d = np.clip(data, minSignal, maxSignal);
            a = maxSignal-minSignal;
            self.digiWaveform = self.digiWaveform + 1j*(np.round(((d/a)-minSignal)*(2**bitdepth-1))/(2**bitdepth-1)+minSignal)*a;

        self._digitized=True;

    def compilePulse(self):
        if self._functionsLST==[] and self._timeFlags==[]:
            warnings.warn('WARNING: Unable to compile. Missing list of functions or time-flags.')
        else:
            # try:
            for _ in range(len(self._functionsLST)):
                if self._functionsLST[_].args['type'] == "quadrature" or self._functionsLST[_].args['type'] == "sinusoidal":
                    self.add2Pulse(self._timeFlags[_], self._functionsLST[_], kind='Carrier', _fromCompiler=True);
                else:
                    self.add2Pulse(self._timeFlags[_], self._functionsLST[_], kind=None, _fromCompiler=True);
            # except:
            #     warnings.warn('WARNING: Missmatch function and time-flag lists\' size')

        self._compiled = True;
    
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
class track:
    """
        A class for building full time-domain track waveforms starting from AQiPT functions() and 
        pulse() for later construction of AQiPT sequence() and or AQiPT experiment(). It can be 
        assigned to AQiPT producers().

        The track class is the AQiPT representation of shaped pulse for create assigned to hardware
        channels.


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.
        step()
            Conjugate of quantum object.
    """
    def __init__(self, track_label, tTrack, pulse_list=None):
        
        #atributes
        self._res = type;
        self.label = track_label;
        self.tTrack = tTrack;
        self.pulseList = pulse_list;
        self.digiWaveform = None;
        self._compiled = False;
    
    def compileTrack(self, new_pulseLST=None):
        if new_pulseLST==None:
            for pulse in self.pulseList:
                if isinstance( self.digiWaveform, np.ndarray):
                    self.digiWaveform = np.concatenate((self.digiWaveform, pulse.getPulse()));
                    # if self.tTrack == None:
                    #     _to=len(self.tTrack)*len(tTrack)
                    #     self.tTrack = np.linspace(0, pulse.tbase[len(pulse.tbase)-1])
                    self._compiled = True;
                else:
                    self.digiWaveform = pulse.getPulse();

    def add2Track(self, pulse_objs):
        if self.digiWaveform is None:
            self.digiWaveform = np.concatenate(pulse_objs, axis=None);
            self._compiled = True;
            self.tTrack = np.zeros(len(self.digiWaveform ));
            self.pulseList = pulse_objs;
        else:
            for pulse in pulse_objs:
                self.digiWaveform = np.concatenate((self.digiWaveform, pulse), axis=None);
                self.tTrack = np.zeros(len(self.digiWaveform ));
                self.pulseList.append(pulse_objs);

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

    def plotTrack(self, axis=None, xlabel=str(), ylabel=str(), pcolor=None, figureSize=(25,3)):
        if axis==None:
            plt.figure(figsize=figureSize);
            plt.step(self.tTrack, self.digiWaveform, alpha=0.9, color= pcolor);
            plt.fill(self.tTrack, self.digiWaveform, alpha=0.3, color= pcolor);

            plt.xlabel(xlabel, fontsize=18);
            plt.ylabel(ylabel, fontsize=18);

            #TO-DO: add labels with correct units from AQiPTcore
        else:
            axis.step(self.tTrack, self.digiWaveform, alpha=0.9, color= pcolor);
            axis.fill(self.tTrack, self.digiWaveform, alpha=0.3, color= pcolor);

            axis.set_xlabel(xlabel, fontsize=18);
            axis.set_ylabel(ylabel, fontsize=18);
            #TO-DO: add labels with correct units from AQiPTcore
    
#####################################################################################################
#Sequence AQiPT class
#####################################################################################################
class sequence:
    '''
        A class for building full time-domain track waveforms starting from AQiPT functions() and 
        pulse() for later construction of AQiPT sequence() and or AQiPT experiment(). It can be 
        assigned to AQiPT producers().

        The track class is the AQiPT representation of shaped pulse for create assigned to hardware
        channels.


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.
        step()
            Conjugate of quantum object.
    '''
    def __init__(self, sequence_label, tSequence, stack=True, variables=None):
        
        #atributes
        self._res = type;
        self.label = sequence_label;
        self.tSequence = tSequence;
        self._Stack = stack;

        self._API_sequence = [];

        if all(isinstance(p, pulse) for p in stack) or all(isinstance(t, track) for t in stack):
            
            try:
                self.tSequence = np.zeros( max([len(w.getPulse()) for w in stack]) );
            except:
                pass
            try:
                self.tSequence = np.zeros( max([len(w.getTrack()) for w in stack]) );
            except:
                pass

            if all(isinstance(p, pulse) for p in self._Stack):
                self.digiWaveformStack = [p.getPulse() for p in self._Stack];

            elif all(isinstance(t, track) for t in self._Stack):
                self.digiWaveformStack = [t.getTrack() for t in self._Stack];

    def stack2Sequence(self, stack, _IDs:bool=False):
        if self._Stack is None:
            
            if _IDs==False:
                if all(isinstance(p, pulse) for p in stack) or all(isinstance(t, track) for t in stack):
                    self._Stack = stack;
                    
                    if all(isinstance(p, pulse) for p in self._Stack):
                        self.tSequence = np.zeros( max([len(w.getPulse()) for w in stack]) );
                        self.digiWaveformStack = [p.getPulse() for p in stack];

                    elif all(isinstance(t, track) for t in self._Stack):
                        self.tSequence = np.zeros( max([len(w.getTrack()) for w in stack]) );
                        self.digiWaveformStack = [t.getTrack() for t in self.stack];
            else:
                if all(isinstance(p[0], pulse) for p in stack) or all(isinstance(t[0], track) for t in stack):
                    self._Stack = stack;
                    
                    if all(isinstance(p[0], pulse) for p in self._Stack):
                        self.tSequence = np.zeros( max([len(w[0].getPulse()) for w in stack]) );
                        self.digiWaveformStack = [p[0].getPulse() for p in stack];

                    elif all(isinstance(t[0], track) for t in self._Stack):
                        self.tSequence = np.zeros( max([len(w[0].getTrack()) for w in stack]) );
                        self.digiWaveformStack = [t[0].getTrack() for t in self.stack];

                    self._Stack = [element[0] for element in stack];

                    try:
                        self._API_sequence += [[p[1], p[2]] for p in stack];
                    except:
                        pass

        else:    
            if _IDs==False:
                if all(isinstance(p, pulse) for p in stack):
                        self.digiWaveformStack += [p.getPulse() for p in stack];
                        self._Stack+=stack;

                elif all(isinstance(t, track) for t in stack):
                    self.digiWaveformStack += [t.getTrack() for t in stack];
                    self._Stack+=stack;
            else:
                if all(isinstance(p[0], pulse) for p in stack):
                        self.digiWaveformStack += [p[0].getPulse() for p in stack];
                        self._Stack+=stack;

                elif all(isinstance(t[0], track) for t in stack):
                    self.digiWaveformStack += [t[0].getTrack() for t in stack];
                    self._Stack+=stack;

                self._Stack = [element[0] for element in stack];

                try:
                    self._API_sequence += [[p[1], p[2]] for p in stack];
                except:
                    pass
                    
    def getSequence(self):
        return self._Stack

    def getLabel(self):
        return self.label

    def clearSequence(self):
        self._Stack = None;
        self.digiWaveformStack = None;
        self.tSequence = None;

    def plotSequence(self, xlabel=str(), ylabel=str(), plotMode='static', figureSize=(18,10), color_sequence='gray', color_text='white', dashboard=False):

        #TO-DO: add labels with correct units from AQiPTcore

        if plotMode == "static":
            plt.rcParams.update({"lines.color": color_text,
                                "patch.edgecolor": color_text,
                                "text.color": "black",
                                "axes.facecolor": color_text,
                                "axes.edgecolor": "lightgray",
                                "axes.labelcolor": color_text,
                                "xtick.color": color_text,
                                "ytick.color": color_text,
                                "grid.color": "lightgray",
                                "figure.facecolor": "black",
                                "figure.edgecolor": "black",
                                "savefig.facecolor": "black",
                                "savefig.edgecolor": "black"})
            
            if len(self.digiWaveformStack)==0:
                fig, axs = plt.subplots( figsize=figureSize, sharex=True, facecolor=color_sequence);

            else:
                fig, axs = plt.subplots(len(self.digiWaveformStack), 1, figsize=figureSize, sharex=True, facecolor=color_sequence);
                fig.subplots_adjust(hspace=0.2);

            _=0;
            for wavef in self.digiWaveformStack:

                color_i = aqipt.color_lst[_%len(aqipt.color_lst)];

                if _==0:
                    fig.suptitle(self.label, fontsize=20, color=color_text);
                if len(self.digiWaveformStack)!=1:
                        

                        if isinstance(self._Stack[_], pulse):
                            axs[_].step(self._Stack[_].tbase, wavef, color=color_i, where='mid');
                            axs[_].fill_between(self._Stack[_].tbase, wavef, color=color_i, step="mid", alpha=0.2);

                        if isinstance(self._Stack[_], track):
                            axs[_].step(self._Stack[_].tTrack, wavef, color=color_i, where='mid');
                            axs[_].fill_between(self._Stack[_].tTrack, wavef, color=color_i, step="mid", alpha=0.2);

                        axs[_].set_facecolor(color_sequence); 
                        axs[_].set_ylabel(self._Stack[_].label);
                        axs[_].minorticks_on();
                        plt.xlabel(xlabel, fontsize=18);
                        axs[_].grid()
                        _+=1;

                else:

                    if isinstance(self._Stack[_], pulse):
                        axs.step(self._Stack[_].tbase, wavef, color=color_i, where='mid');
                        axs.fill_between(self._Stack[_].tbase, wavef, color=color_i, step="mid", alpha=0.2);

                    if isinstance(self._Stack[_], track):
                        axsstep(self._Stack[_].tTrack, wavef, color=color_i, where='mid');
                        axs.fill_between(self._Stack[_].tTrack, wavef, color=color_i, step="mid", alpha=0.2);

                    axs.set_facecolor(color_sequence); 
                    axs.set_ylabel(self._Stack[_].label);
                    axs.minorticks_on();
                    plt.xlabel(xlabel, fontsize=18);
                    axs.grid()

            plt.rcParams.update(plt.rcParamsDefault);
            plt.show()

        elif plotMode=='dynamic':

            # Create figure
            fig = plty.Figure()

            for i in range(len(self._Stack)):
                try:
                    fig.add_trace(
                        plty.Scatter(x=self._Stack[i].tTrack,
                                   y=np.real(self._Stack[i].digiWaveform), name="Re{"+self._Stack[i-1].label+"}"))
                    fig.add_trace(
                        plty.Scatter(x=self._Stack[i].tTrack,
                                   y=np.imag(self._Stack[i].digiWaveform), name="Im{"+self._Stack[i-1].label+"}"))
                except:
                    pass
                try:
                    fig.add_trace(
                        plty.Scatter(x=self._Stack[i].tbase,
                                     y=np.real(self._Stack[i].digiWaveform), name="Re{"+self._Stack[i-1].label+"}"))
                    fig.add_trace(
                        plty.Scatter(x=self._Stack[i].tbase,
                                     y=np.imag(self._Stack[i].digiWaveform), name="Im{"+self._Stack[i-1].label+"}"))
                except:
                    pass

                # Set title
                fig.update_layout(
                    title_text= self.label
                )

                # Add range slider
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([])
                        ),
                        rangeslider=dict(
                            visible=True
                        ),
                #         type="date"
                    )
                )

            fig.update_layout(title_text=self.getLabel())

            if dashboard==True:
                return fig
            else:
                fig.show()

        elif plotMode=='dynamic-subplots':
            
            from plotly.subplots import make_subplots

            # Create figure
            fig = make_subplots(rows=int(len(self._Stack)), cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.5/int(len(self._Stack)))

            for i in range(1, len(self._Stack)+1):

                if isinstance(self._Stack[i-1], track):
                    try:
                        fig.add_trace(
                            plty.Scatter(x=self._Stack[i-1].tTrack,
                                         y=np.real(self._Stack[i-1].digiWaveform), name="Re{"+self._Stack[i-1].label+"}"),
                                         row=i, col=1)
                        fig.add_trace(
                            plty.Scatter(x=self._Stack[i-1].tTrack,
                                         y=np.imag(self._Stack[i-1].digiWaveform), name="Im{"+self._Stack[i-1].label+"}"),
                                         row=i, col=1)
                    except:
                        pass

                    # fig.add_trace(
                    #     plty.Scatter(x=self._Stack[i-1].tTrack,
                    #                  y=self._Stack[i-1].digiWaveform, name=self._Stack[i-1].label),
                    #     row=i, col=1)

                if isinstance(self._Stack[i-1], pulse):
                    try:
                        fig.add_trace(
                            plty.Scatter(x=self._Stack[i-1].tbase,
                                         y=np.real(self._Stack[i-1].digiWaveform), name="Re{"+self._Stack[i-1].label+"}"),
                            row=i, col=1)
                        fig.add_trace(
                            plty.Scatter(x=self._Stack[i-1].tbase,
                                         y=np.imag(self._Stack[i-1].digiWaveform), name="Im{"+self._Stack[i-1].label+"}"),
                            row=i, col=1)
                    except:
                        pass
                
                    # fig.add_trace(
                    #     plty.Scatter(x=self._Stack[i-1].tbase,
                    #                  y=self._Stack[i-1].digiWaveform, name=self._Stack[i-1].label),
                    #     row=i, col=1)

                

                #Add range slider                
                fig.update_xaxes(rangeslider= {'visible':True}, row=int(len(self._Stack)), col=1);
            
            fig.update_layout(title_text=self.getLabel())

            if dashboard==True:
                return fig
            
            else:
                fig.show()

#####################################################################################################
#Experiment AQiPT class
#####################################################################################################

def extract_tracks(waveformLST):
    '''
        INPUT: 
        ------
            waveformLST (list) : list of AQiPT control module objects e.g., functions, pulses, tracks, sequences.

        OUTPUT:
        ------
            list of waveforms (AQiPT control object)
    '''
    if isinstance(waveformLST, sequence):
        return waveformLST._Stack

    elif isinstance(waveformLST, list):
        return [wavef for wavef in waveformLST if isinstance(wavef, track) or isinstance(wavef, pulse)]

    else:
        raise ValueError("Not possible to extract waveforms.")

def extract_sequences(waveformLST):
    '''
        INPUT: 
        ------
            waveformLST (list) : list of AQiPT control module objects e.g., functions, pulses, tracks, sequences.

        OUTPUT:
        ------
            list of waveforms (AQiPT control object)
    '''
    return [wavef for wavef in waveformLST if isinstance(wavef, sequence)]

class experiment:
    '''
        A class for containing the general experiment: hardware, software(protocols), database, models, producers and
        metadata. Requires a pre-loaded director-object with the Hardware and Software specifications to match the
        experimental sequences (via JSON file or AQiPT formalism of waveforms) with physical hardware channels. 


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.
        step()
            Conjugate of quantum object.
    '''
    def __init__(self, atom=None, waveforms=None, hardware=None, director=None, operationMode=None, database=None, producer=None, metadata=None, date=f"{datetime.datetime.now():%Y-%m-%d | %T}"):

        #atributes
        self._metadata = metadata;
        self._creation = date;
        self._operationMode = operationMode;

        self._cmdMapperType = None;
        self._cmdMapper = None;
        self._cmdMapperFile = None;

        self.atom = atom;

        self.director = director;

        if hardware!=None:
            self.hardware = hardware;
        else:
            self.hardware = self.director._HW_specs;

        self.producer = producer;
        self.database = database;



        if waveforms!=None and director==None:
            self.waveforms = waveforms;
            self._cmdMapperType = 'WTYPE';

            if self._operationMode=='sequence':
                self.sequences = extract_sequences(self.waveforms);
                self.tracks = None;

            elif self._operationMode=='track':
                self.tracks = extract_tracks(self.waveforms);
                self.sequences = None;
            else:
                raise ValueError("Incompatible operation mode, please choose: 'sequence' or 'track'.")

        elif waveforms==None and director!=None:
            self.waveforms = waveforms;
            self._cmdMapperType = 'FTYPE';

            if self._operationMode=='sequence':
                self.sequences = self.director._SW_specs.specifications['sequences'];
                self.tracks = None;

            elif self._operationMode=='track':
                self.tracks = self.director._SW_specs.specifications['tracks'];
                self.sequences = None;
            else:
                raise ValueError("Incompatible operation mode, please choose: 'sequence' or 'track'.")


        else:
            raise ValueError("Ambiguity in giving waveforms and director simultaneously.")

    def _executeIFexist(obj, method_name):
        if callable(getattr(obj, method_name, None)):
            getattr(obj, method_name)()
        else:
            print(f"{method_name} is not a valid method of the class.")
    
    def _assembler(self):
        '''
            Assemble the primitive device commands located in the hardware list device with
            the argumentes given in the quantum assembly commands located in the software map.
            Returns a command+argument map with time-flag ready to be executed by execute()
            method.  
        '''
        _scheduler = scheduler(time.time, time.sleep);

        for idx in range(len(self.director._SWmap)):
            for device in self.hardware.hardwareTypes:
                for idx_dv in range(len(self.hardware.hardwareLST[str(device)])):
                    
                    try:
                        _hex_hardware_id = aqipt.DFsearch(pd.DataFrame(self.director._HW_specs._IDs_bench), self.director._SWmap[idx].instruction_realCH)['python_memory_alloc'][aqipt.DFsearch(pd.DataFrame(self.director._HW_specs._IDs_bench), self.director._SWmap[idx].instruction_realCH)['python_memory_alloc'].index[0]] 
                        
                        if ctypes.cast( int(_hex_hardware_id, 16), ctypes.py_object).value.specs['EXT_CHANNEL'] in self.hardware.hardwareLST[str(device)][idx_dv].specs['properties']['channels_IDs']:
                            # print(True, device, idx)

                            if callable(getattr(self.hardware.hardwareLST[str(device)][idx_dv].driver, self.director._SWmap[idx].command, None)):
                                if self.director._SWmap[idx].instruction_type == 'DIGITAL':
                                    _scheduler.enterabs(time= self.director._SWmap[idx].instruction_timeflag, priority=1, action=getattr(self.hardware.hardwareLST[str(device)][idx_dv].driver, self.director._SWmap[idx].command), argument=(), kwargs={});
                                elif self.director._SWmap[idx].instruction_type == 'ANALOG' or self.director._SWmap[idx].instruction_type == 'ANALOGS':
                                    _instruction_name = title = self.director._SWmap[idx].instruction_name;
                                    cmd_waveform = sum(self.director._SW_specs.specifications['instructions'][idx][_instruction_name]['WAVEFORM'], []);
                                    _scheduler.enterabs(time= self.director._SWmap[idx].instruction_timeflag, priority=1, action=getattr(self.hardware.hardwareLST[str(device)][idx_dv].driver, self.director._SWmap[idx].command), argument=(cmd_waveform, self.director._SWmap[idx].instruction_type), kwargs={});
                                
                                # print('time= ',self.director._SWmap[idx].instruction_timeflag, 'priority=',1, 'action=',getattr(self.hardware.hardwareLST[str(device)][idx_dv].driver, self.director._SWmap[idx].command))
                            # _scheduler.enter(_executeIFexist(self.hardware.hardwareLST[str(device)][idx_dv].driver, self.director._SWmap[idx].command)) # maybe too slow by calling the exeIFexist
                            # print('here is the queue',_scheduler.queue)

                    except:
                        pass
        
        return _scheduler

    def compile(self):
        '''
            Compile the quantum assembly primitives into the hardware by creating two mappers, one
            for the primitives that will serve as argument of the command list from the second mapper.
            This retrieve a pair command,argument after the 
        '''
        self.director._mapperHW(hardware=self.hardware);

        if self._operationMode=='sequence':
            self.director._mapperSW(config=self.sequences);
        elif self._operationMode=='track':
            self.director._mapperSW(config=self.tracks);

        #pre-configuring the devices


        #compiling the command mapper as schedule object
        self._cmdMapper = self._assembler();
        return None

    def recompile(self):
        self.director.load_drivers();
        self.director.load_HWspecs();
        self.compile();

    def execute(self):
        self._cmdMapper.run()
        return None

#####################################################################################################
#Director AQiPT class
#####################################################################################################
class director:
    '''
        A class for coordinating the Software specificiations (protocol specifications) that defines
        the waveforms for the experiment from specification file and the Hardware specification (setup 
        specifications) with the devices' drivers.

        At the master level (experiment class) does compile the instructions or waveforms into the
        physical channels in the experiment, setting the channel ready to run the instruction/waveform
        when the experiment class is executed e.g., experiment.execute()


        Parameters
        ----------
        tbase : array_like
            Data for vector/matrix representation of the quantum object.
        res : list
            Dimensions of object used for tensor products.
        args : list
            Shape of underlying data structure (matrix shape).

        Methods
        -------
        function : array_like
            Sparse matrix characterizing the quantum object.
        function_plot : list
            List of dimensions keeping track of the tensor structure.
        step()
            Conjugate of quantum object.
    '''
    def __init__(self, HW_specs=None, SW_specs=None, atom_specs=None, calibration=None):

        #atributes
        self._HW_specs = HW_specs;
        self._SW_specs = SW_specs;
        self._atom_specs = atom_specs;

        self._calibration = calibration;
        
        self._drivers = {}
        self._ID = None;

        self._SWmap = [];
        self._iscompiled = False;

    def load_HWspecs(self, file=None):
        if file!= None:
            _hw_specs = kernel.hardwareSpecs(path=file);
        else:
            _hw_specs = kernel.hardwareSpecs();
        _hw_specs.loadSpecs(printON=False);
        _hw_specs.initSetup();
        self._HW_specs = _hw_specs;

    def load_SWspecs(self, file=None):
        if file!= None:
            _sw_specs = kernel.softwareSpecs(path=file);
        else:
            _sw_specs = kernel.softwareSpecs();
        _sw_specs.loadSpecs(printON=False);
        self._SW_specs = _sw_specs;

    def load_drivers(self):

        #loading virtual devices
        import AQiPT.hardware.drivers.AQiPTvd as __vdrivers
        self._drivers['virtual'] = __vdrivers;

        #loading real devices 
        from AQiPT.hardware.drivers.AQiPTrd import drivers as __rdrivers
        self._drivers['real'] = __rdrivers;

    def _mapperHW(self, hardware):

        _drivables = ['laser', 'laser-aux', 'camera', 'awg', 'dds', 'dmd', 'shutter', 'oscilloscope', 'camera'];

        #check if HW specs in director match with specs in experiment.hardware.specs, show error otherwise
        if self._HW_specs.specifications == hardware.specifications:
            pass
        else:
            raise ValueError("Specifications in director and experiment do not match.")


        for _HWtype in self._HW_specs.hardwareLST:

            if _HWtype in _drivables:
                for element in self._HW_specs.hardwareLST[_HWtype]:

                    if 'virtual' in element.driver:
                        
                        if element.driver['virtual'] == 'DMD':
                            element.driver = self._drivers['virtual'].DMD();

                        elif element.driver['virtual'] == 'DDS':
                            element.driver = self._drivers['virtual'].DDS();

                        elif element.driver['virtual'] == 'camera':
                            element.driver = self._drivers['virtual'].camera();

                        elif element.driver['virtual'] == 'AWG':
                            element.driver = self._drivers['virtual'].AWG();

                        elif element.driver['virtual'] == 'laser':
                            element.driver = self._drivers['virtual'].laser();

                        elif element.driver['virtual'] == 'laser-aux':
                            element.driver = self._drivers['virtual'].laser();
                        
                        elif element.driver['virtual'] == 'shutter':
                            element.driver = self._drivers['virtual'].shutter();
                        
                        elif element.driver['virtual'] == 'oscilloscope':
                            element.driver = self._drivers['virtual'].oscilloscope();

                    elif 'real' in element.driver:
                        pass

    def _mapperSW(self, config):

        if self._iscompiled == False:

            try:
                _variables = pd.DataFrame(self._SW_specs.specifications['variables'][0]);
            except:
                pass
            _assembly = pd.DataFrame(config);
            _instructions = self._SW_specs.specifications['instructions'];

            #for tracks
            #not implemented yet

            #for sequences
            _seq_idx=0;_instruction_idx=0;
            for sequence in _assembly:

                
                for instruction in _assembly[str(sequence)][_seq_idx]['INSTRUCTIONS']:

                    if isinstance(_assembly[str(sequence)][_seq_idx]['INSTRUCTIONS'],list):

                        try:

                            _instructions[_instruction_idx][instruction]['TIME_FLAG']=_variables[str(_instructions[_instruction_idx][instruction]['TIME_FLAG'])]['value'];

                        except:
                            try:
                                _instructions[_instruction_idx][instruction]['TIME_FLAG']=_instructions[_instruction_idx][instruction]['TIME_FLAG'];
                            except:
                                pass

                        try:

                            _instructions[_instruction_idx][instruction]['TIME_FLAGS']=_variables[str(_instructions[_instruction_idx][instruction]['TIME_FLAGS'])]['value'];
                        
                        except:
                            try:
                                _instructions[_instruction_idx][instruction]['TIME_FLAGS']=_instructions[_instruction_idx][instruction]['TIME_FLAGS'];
                        
                            except:
                                pass
                        

                        try:
                            
                            if _instructions[_instruction_idx][instruction]['SPECS']['args'] !='None':
                                
                                for key, value in _instructions[_instruction_idx][instruction]['SPECS']['args'].items():

                                        try:
                                            _instructions[_instruction_idx][instruction]['SPECS']['args'][str(key)]=_variables[str(value)]['value'];
                                        except:
                                            _instructions[_instruction_idx][instruction]['SPECS']['args'][str(key)]=value;

                                                     
                        except:
                            pass

                        #SW map
                        if _instructions[_instruction_idx][instruction]['TYPE']=='DIGITAL' or _instructions[_instruction_idx][instruction]['TYPE']=='ANALOG':
                            self._SWmap.append(mapSW(sequence_name = sequence, 
                                                     sequence_id = _assembly[str(sequence)][_seq_idx]['ID'], 
                                                     instruction_name = instruction, 
                                                     instruction_id = _instructions[_instruction_idx][instruction]['ID'], 
                                                     instruction_type = _instructions[_instruction_idx][instruction]['TYPE'], 
                                                     instruction_specs = _instructions[_instruction_idx][instruction]['SPECS'], 
                                                     instruction_realCH = _instructions[_instruction_idx][instruction]['CHANNEL'], 
                                                     instruction_timeflag = _instructions[_instruction_idx][instruction]['TIME_FLAG'],
                                                     command = _instructions[_instruction_idx][instruction]['COMMAND']));
                        
                        if _instructions[_instruction_idx][instruction]['TYPE']=='DIGITALS' or _instructions[_instruction_idx][instruction]['TYPE']=='ANALOGS':
                            self._SWmap.append(mapSW(sequence_name = sequence, 
                                                     sequence_id = _assembly[str(sequence)][_seq_idx]['ID'], 
                                                     instruction_name = instruction, 
                                                     instruction_id = _instructions[_instruction_idx][instruction]['ID'], 
                                                     instruction_type = _instructions[_instruction_idx][instruction]['TYPE'], 
                                                     instruction_specs = _instructions[_instruction_idx][instruction]['SPECS'], 
                                                     instruction_realCH = _instructions[_instruction_idx][instruction]['CHANNEL'], 
                                                     instruction_timeflags = _instructions[_instruction_idx][instruction]['TIME_FLAGS'],
                                                     command = _instructions[_instruction_idx][instruction]['COMMAND']));

                        _instruction_idx+=1;
                _seq_idx+=1;
                self._iscompiled=True;
        else:
            print('Already compiled.')

    def load_calibration(self):
        return None

@dataclass(frozen=False)
class mapSW:
    '''
        Special dataclass of AQiPT that stores the quantum assembly commands with the low level arguments
        for the classical control.
        
        ATTRIBUTES/FIELDS:
        ------------------
        
            sequence_name : name of the sequence 
            sequence_id : ID of the sequence
            sequence_active : sequence active?
            instruction_name : name of instruction
            instruction_id : ID of the instruction
            instruction_type : type of instruction
            instruction_specs : set of arguments for command
            instruction_realCH : channel's ID of hardware
            instruction_timeflag : value of time when it will be executed
            command : command to device from driver
            
        
        METHODS:
        --------
        
        
    '''
        
    #fields
    sequence_name : str = field(default="Default")
    sequence_id : str = field(default="0x0")
    instruction_name : str = field(default="Default")
    instruction_id : str = field(default="0x0")
    instruction_type : str = field(default="Default")
    instruction_specs : dict = field(default=dict, metadata="Default")
    instruction_realCH : str = field(default="0x0", metadata={'hardware_ID': "0x0"})
    instruction_timeflag : float = field(default=0.0, metadata={'unit': 'TIME_UNIT'})
    instruction_timeflags : float = field(default=0.0, metadata={'unit': 'TIME_UNIT'})
    command : str = field(default=str, metadata="Driver required")
