#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Control module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): S.Whitlock. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2024-12-14


#libs
import warnings#, os
import datetime, time
import copy

import ctypes

from inspect import isclass
from sys import modules, version_info
from os import remove

from sched import scheduler
# import inspect, queue

from dataclasses import dataclass, field

import math as mt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import plotly.graph_objects as plty
from tqdm import tqdm

import imageio
import pandas as pd

from hexalattice.hexalattice import *

import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import chirp, resample
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, linear_sum_assignment
from scipy.spatial.distance import cdist

from itertools import combinations

from AQiPT import AQiPTcore as aqipt
import AQiPT.modules.kernel.AQiPTkernel as kernel

from AQiPT.modules.directory import AQiPTdirectory as dirPath

directory = aqipt.directory;


#for functions
INTERPOLATION_METHODS = [ 'linear', 'nearest', 'nearest-up', 
                          'zero', 'slinear', 'quadratic', 'cubic', 
                          'previous', 'next', 'zero', 'slinear', 'quadratic', 'cubic' ];

#for IAC
TEMP_GIF_STORE_FILES = '/frames/temp'

#for NCOs
INITIAL_AMPLITUDE_NCOS = [0.5]
INITIAL_FREQUENCY_NCOS = [50e6]
INITIAL_PHASE_NCOS = [0]

PYTHON_VERSION = '{VERSION}.{MINOR}'.format(VERSION=version_info[0], MINOR=version_info[1])

#####################################################################################################
#Util functions
#####################################################################################################
def info():
    '''
        Returns a list of all classes defined in the library.
    '''
    
    current_module = modules[__name__] #get the current module (your library)
    classes = []

    for name, obj in vars(current_module).items():
        if isclass(obj) and obj.__module__.startswith(current_module.__name__): #find classes in this module
            classes.append(name)

    return f"<AQiPT control: \n Classes: {', '.join(classes)}> \n Last update: 2024-12-14 \n Author(s): Manuel Morgado S.Whitlock \n Website: https://github.com/AQiPT"

def _resample(tbase, _array, nSamples, interpolation=None):

    if interpolation==None:
        _new_array = resample(_array, nSamples)
        return _new_array

    elif interpolation!=None:

        if interpolation in INTERPOLATION_METHODS:
            _interp = interp1d(tbase, _array, kind=interpolation)
            return _interp

def _area_under_curve_arbitrary(x_data, y_data):
    return simps(y_data, x_data)

def _xquare(var, _coef=[1,1,1]):
    return _coef[0]*(var)**2 + _coef[1]*(var) + _coef[2]

def _constant(var, value):
    return np.ones(len(var))*value

def _noisify(image, noise_level): #background noise

    image_shape = np.shape(image);
    image += np.random.random((image_shape[0],image_shape[1]))*noise_level;

    return image 

###################################################################################################
#######################                 Frontend Control                  #########################
###################################################################################################


###################################################################################################
#######################                 Middleware Control                  #######################
###################################################################################################


#####################################################################################################
#Function AQiPT class
#####################################################################################################
class function:
    '''
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

        resample(nSample)
            Resample the array to nSample number of points.
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
    '''

    def __init__(self, times, args, area=None):
        
        #(times=None, args=None, area=np.pi)
        #atributes
        self.tbase = times
        self._res = type
        self.args = args
        self.waveform = None;
        self.area = None;
        self._plot = None;

    def resample(self, nSamples):

        self.waveform = _resample(self.waveform, nSamples);

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

        try:
            function = self.args['amp'] * (abs(self.tbase-self.args['t_o']) < self.args['width']) + self.args['offset'];
        except:
            function = self.args['amp'] * (abs(self.tbase-self.args['t_o']) < self.args['width']);

        self.waveform = function;

        if plotON==True:

            fig = plt.figure();
            function_plot = self.plotFunction(Hunits='[$\mu s$]', Vunits='[V]');
            self._plot = function_plot;

            return self.waveform, self._plot
        else:            
            return function, None
    
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
            return function

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
            function =  self.args['amp']*(1/(self.args['sigma']*np.sqrt(2*np.pi)))*np.exp(-(self.tbase-self.args['g_center'])**2/(2*self.args['g_std']**2))*function
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

            INPUTS:
            ------
                t (array): time domain of function
                args (dict): width of gaussian function (i.e., amplitude, center, standar deviation)
                plotON (bool): shows (True) or not (False) plot

            OUTPUT:
            ------
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

            return self.waveform,  self._plot

        else:

            return function, None
    
    def blackman(t, v_start, v_end):
        '''

            Basic Blackmand pulse, which reduces spectral leakage..

            INPUTS:
            ------
                t (array): time domain of function
                args (dict): width of blackman function (i.e., amplitude, center, standar deviation)
                plotON (bool): shows (True) or not (False) plot

            OUTPUT:
            ------
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                    
        '''
        _dt = np.max(self.tbase)
        function = (0.5 * (self.args['Ampi'] + self.args['Ampf']) + 0.5 * (self.args['Ampf'] - self.args['Ampi']) * (0.42 - 0.5 * np.cos(2 * np.pi * self.tbase / _dt)+ 0.08 * np.cos(4 * np.pi * self.tbase / _dt)))

        self.waveform = function;


        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(self.tbase, function, figure=fig);
            self._plot = function_plot;

            return self.waveform,  self._plot

        else:

            return function, None

    def supergaussian(self, plotON=False):
        '''
            Supergaussian Gaussian function
    
            f(x, sigma, n) = y0 + A*exp(-(abs(x-x0) / sigma)^n)

            INPUTS
                t (array): time domain of function
                args (dict): width of step function (i.e., amplitude, center, standar deviation)
                plotON (bool): shows (True) or not (False) plot

            OUTPUTS
                function: function ready for aqipt.waveform()
                fig: plot of function

            \Example:

                    

        '''

        function =  self.args['yoffset']+ self.args['amp']*np.exp(-((abs(self.tbase-self.args['xoffset']) / self.args['std']) ** self.args['n_param']));
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

        self.waveform = function;

        if plotON==True:

            function_plot = self.plotFunction(Hunits='[$\mu s$]', Vunits='[V]');
    
            return self.waveform, None

        else:

            return function, self._plot
    
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

    def cspline(self, plotON=False):
        '''
            Cubic spline function

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
        _cs = CubicSpline(self.args['x'], self.args['y']);
        _xs = np.arange(min(self.tbase), max(self.tbase), (max(self.tbase)- min(self.tbase))/len(self.tbase));

        try:
            function = _cs(_xs, self.args['order']);
        except:
            function = _cs(_xs);

        self.tbase = _xs;
        self.waveform = function;

        if plotON==True:

            fig = plt.figure();
            function_plot = plt.plot(_xs, function, figure=fig);
            plt.show(function_plot)

            return function, plt.plot(_xs, function)

        else:

            return function, None

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

    global _opt_hist
    _opt_hist=[]
    def multitone(tones:dict=None, frequencies=[50, 120, 250], amplitudes=[1.0, 1.0, 1.0], phases=[0, 0, 0], sampling_rate=1000, duration = 1):
       '''
         Generate a multi-tone signal with given frequencies, amplitudes, and phases.

         INPUTS:
         -------

         - frequencies: list of frequencies (in Hz) for the tones.
         - amplitudes: list of amplitudes corresponding to each frequency.
         - phases: list of phases (in radians) corresponding to each frequency.
         - sampling_rate: number of samples per second.

         OUTPUTS:
         --------

         - time: time axis for the generated signal.
         - time_signal: multi-tone signal in the time domain.
       '''

       if tones is None:

          #define parameters
          n = int(sampling_rate * duration)  #total number of samples

          #check if the number of amplitudes and phases matches the number of frequencies
          if len(amplitudes) != len(frequencies) or len(phases) != len(frequencies):
             raise ValueError("The number of amplitudes and phases must match the number of frequencies.")

          #generate the frequency domain representation (FFT spectrum)
          spectrum = np.zeros(n, dtype=complex)  #initialize with zeros

          #set the frequency components (positive frequencies) with amplitudes and phases
          for i, f in enumerate(frequencies):
             idx = int(f * duration)  #convert frequency to index in FFT
             #set the magnitude and phase (complex exponential with amplitude and phase)
             spectrum[idx] = amplitudes[i] * np.exp(1j * phases[i])  #amplitude and phase

          #create the conjugate symmetric part for negative frequencies (for real signal)
          spectrum[-(n//2)+1:] = np.conj(spectrum[1:(n//2)][::-1])

          #compute the inverse FFT to get the time-domain signal
          time_signal = np.fft.ifft(spectrum)

          #extract the real part (since we're creating a real signal)
          time_signal = np.real(time_signal)

          #generate the time axis for plotting
          time = np.linspace(0, duration, n, endpoint=False)

          return time, time_signal

       else:

          frequencies = [ tones[_tone_idx]['frequency'] for _tone_idx in range(len(tones)) ]
          amplitudes = [ tones[_tone_idx]['amplitude'] for _tone_idx in range(len(tones)) ]
          phases = [ tones[_tone_idx]['phase'] for _tone_idx in range(len(tones)) ]

          #define parameters
          n = int(sampling_rate * duration)  # Total number of samples

          #check if the number of amplitudes and phases matches the number of frequencies
          if len(amplitudes) != len(frequencies) or len(phases) != len(frequencies):
             raise ValueError("The number of amplitudes and phases must match the number of frequencies.")

          #generate the frequency domain representation (FFT spectrum)
          spectrum = np.zeros(n, dtype=complex)  #initialize with zeros

          #set the frequency components (positive frequencies) with amplitudes and phases
          for i, f in enumerate(frequencies):
             idx = int(f * duration)  #convert frequency to index in FFT
             #set the magnitude and phase (complex exponential with amplitude and phase)
             spectrum[idx] = amplitudes[i] * np.exp(1j * phases[i])  #amplitude and phase

          #create the conjugate symmetric part for negative frequencies (for real signal)
          spectrum[-(n//2)+1:] = np.conj(spectrum[1:(n//2)][::-1])

          #compute the inverse FFT to get the time-domain signal
          time_signal = np.fft.ifft(spectrum)

          #extract the real part (since we're creating a real signal)
          time_signal = np.real(time_signal)

          #generate the time axis for plotting
          time = np.linspace(0, duration, n, endpoint=False)

          return time, time_signal

    def add_tones(tones, nr_tones, sampling_rate, duration, frequency_range=None, frequencies=None, amplitudes=1.0, phases=0.0, plotON=False):

       N = len(tones) #initial number of tones

       #check input for frequencies, if list or range
       if frequency_range is None and isinstance(frequencies, list):
          freq_new_ideal_tones = frequencies

       elif isinstance(frequency_range,list) and frequencies is None:
          freq_new_ideal_tones = np.linspace(frequency_range[0], frequency_range[1], nr_tones, endpoint=True)

       else:

          ValueError('Frequency specification does not match.')


       #check inputs for amplitudes and phases (all combis)
       if isinstance(amplitudes, list) and not isinstance(phases, list):

          for _ in range(nr_tones):

             tones[_+N] = {'frequency': freq_new_ideal_tones[_], 'amplitude': amplitudes[_], 'phase':phases }

       elif isinstance(amplitudes, list) and  isinstance(phases, list):

          for _ in range(nr_tones):

             tones[_+N] = {'frequency': freq_new_ideal_tones[_], 'amplitude': amplitudes[_], 'phase':phases[_] }

       elif not isinstance(amplitudes, list) and  isinstance(phases, list):

          for _ in range(nr_tones):

             tones[_+N] = {'frequency': freq_new_ideal_tones[_], 'amplitude': amplitudes, 'phase':phases[_] }

       elif not isinstance(amplitudes, list) and  not isinstance(phases, list):

          for _ in range(nr_tones):

             tones[_+N] = {'frequency': freq_new_ideal_tones[_], 'amplitude': amplitudes, 'phase':phases }

       else:

          ValueErro(r'Amplitudes or phases specifications does not match.')


       #generates mt signal
       _signal_time, _signal = multitone(tones = tones,
                                         sampling_rate = sampling_rate, 
                                         duration = duration);

       #generates the fft and plot it (if so)
       _domain_fft, _signal_fft = aqipt.get_fft(_signal, duration, sampling_rate, plotON=plotON);

       return tones, _signal_time, _signal, _domain_fft, _signal_fft

    def intermodulate_pairs(tbase, tone_1, tone_2, nonLinear_coef={'a1':1, 'a2':1, 'a3':1}):

       a1 = nonLinear_coef['a1']; a2 = nonLinear_coef['a2']; a3 = nonLinear_coef['a3'];

       freq_1 = tone_1['frequency']; freq_2 = tone_2['frequency'];
       amp_1 = tone_1['amplitude']; amp_2 = tone_2['amplitude'];
       phase_1 = tone_1['phase']; phase_2 = tone_2['phase'];

       _intermodulation_freq_list = [];
       _intermodulation_amp_list = [];

       _intermodulation_freq_list+=[freq_1, freq_2,                                                     #1st order (no intermod)
                                    freq_2-freq_1, freq_1+freq_2,                                       #intermod 2nd order
                                    2*freq_1, 2*freq_2,                                                 #harmonics 2nd order
                                    2*freq_1-freq_2, 2*freq_2-freq_1, 2*freq_1+freq_2, 2*freq_2+freq_1,  #intermod 3rd order
                                    3*freq_1, 3*freq_2                                                  #harmonics 3rd order
                                    ];
       _intermodulation_amp_list+=[a1*amp_1*np.exp(1j*phase_1)+0.75*a3*(amp_1**3)*np.exp(1j*3*phase_1)+1.5*a3*amp_1*(amp_2**2)*np.exp(1j*(phase_1+2*phase_2)),
                                   a1*amp_2*np.exp(1j*phase_2)+0.75*a3*(amp_2**3)*np.exp(1j*3*phase_2)+1.5*a3*amp_2*(amp_1**2)*np.exp(1j*(phase_2+2*phase_1)), #1st order (no intermod)
                                   2*a2*amp_1*amp_2*np.exp(1j*(phase_1-phase_2)),
                                   2*a2*amp_1*amp_2*np.exp(1j*(phase_1+phase_2)), #intermod 2nd order
                                   a2*(amp_1**2)*np.exp(1j*2*phase_1),
                                   a2*(amp_2**2)*np.exp(1j*2*phase_2), #harmonics 2nd order
                                   3*a3*(amp_1**2)*amp_2*np.exp(1j*(2*phase_1-phase_2)),
                                   3*a3*amp_1*(amp_2**2)*np.exp(1j*(2*phase_2-phase_1)),
                                   3*a3*(amp_1**2)*amp_2*np.exp(1j*(2*phase_1+phase_2)),
                                   3*a3*amp_1*(amp_2**2)*np.exp(1j*(2*phase_2+phase_1)), #intermod 3rd order
                                   a3*(amp_1**3)*np.exp(1j*3*phase_1),
                                   a3*(amp_2**3)*np.exp(1j*3*phase_2) #harmonics 3rd order
                                  ];
       _signal=0;
       for _freq, _amp in zip(_intermodulation_freq_list, _intermodulation_amp_list):
          _signal += _amp*np.sin(2*np.pi*_freq*tbase);

       return _signal

    def intermodulate_multitone(time, tones):

       _signal=0;
       num_elements = len(tones)

       for _tone1_idx, _tone2_idx in combinations(tones.keys(), 2):

          if tones[_tone1_idx] == 0 and tones[_tone2_idx] == num_elements - 1:
             continue

          _signal += intermodulate_pairs(time, tones[_tone1_idx], tones[_tone2_idx])

       return _signal

    def multitone_variance(params, *args):

       '''
          Calculates the variance of the multitone frequency tones by pairs, following the expression:
          E^- = Σ_i,j A_ij^- * exp{i(phi_i - phi_j)} * exp{i(omega_i - omega_j)t}

       '''

       frequencies = args[:-1][0]
       time = args[-1]
       _multitone = np.zeros_like(time,dtype=complex)

       amplitudes = params[:int(len(params)/2)]
       phases = params[int(len(params)/2):]

       for idx_freq_i in range(len(frequencies)):

          for idx_freq_j in range(len(frequencies)):

             if idx_freq_i!=idx_freq_j and idx_freq_j>idx_freq_i:
                _multitone += amplitudes[idx_freq_i]*np.exp(1j*(phases[idx_freq_i]-phases[idx_freq_j]))*np.exp(1j*2*np.pi*(frequencies[idx_freq_i]-frequencies[idx_freq_j])*time) #calculation of the sum terms

       _opt_hist.append(np.var(_multitone))
       return np.var(_multitone)

    def equalization(tones, time, min_func=multitone_variance):
       
       '''
          Unpack tones and minimize a given function (multitone_variance as default), yielding the new amplitudes and phases
          of eac hmaitone.

          Tones are given as a dictionary structure, for example:

          tones = { <idx> : {'Frequency': <val>, 'phase': <val>, 'amplitude': <val>} }
       '''

       #unpacking fpa values
       frequencies=[]; phases=[]; amplitudes=[];
       for _idx,tone in tones.items():

          frequencies.append(tone['frequency'])
          phases.append(tone['phase'])
          amplitudes.append(tone['amplitude'])

       if None in phases:
          phases = 2*np.pi*np.random.rand(len(frequencies)) #generating phases if not defined

       if None in amplitudes:
          amplitudes = np.ones_like(frequencies) #generating amplitudes if not defined

       
       opt_params = minimize(min_func, 
                             np.asarray(amplitudes+phases), 
                             args = (frequencies, time), 
                             bounds = np.reshape( (((0.1, 2*np.pi),)*len(frequencies), ((0.8, 1),)*len(frequencies)), (2*len(frequencies),2)), 
                             options={'return_all':True}, 
                             method='COBYLA',
                             tol=1e-2 ) #optimization step

       return opt_params, _opt_hist 

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
    '''
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
    '''

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
    
    def resample(self, nSamples, interpolation=None):
        '''
            Method that resample the waveform reducing the number of points in the array. It can work
            with a basic resampling or using interpolation methods.

            NOTE:
            -----

            Resampling can work with resampling the number of points without considerations of the 
            t-domain, this lead to defects in the waveform, but realistic for undersampled waveforms. 
            Better quality waveforms can be obtained from using the interpolation method of SciPy that
            take into account the t-domain.

            INPUTS
                nSamples (int) : number of samples desired in the new waveform
                interpolation (str) : type of interpolation used to resample the waveform from the list of 
                                      SciPy: [ 'linear', 'nearest', 'nearest-up', 
                                               'zero', 'slinear', 'quadratic', 'cubic', 
                                               'previous', 'next'. 'zero', 'slinear', 'quadratic', 'cubic' ]
        
            OUTPUTS
                self.waveform (array) : resampled waveform of the pulse class
                self.tbase (array) : resampled waveform of the pulse class
                self.digiwaveform (arra) : resampled digitized waveform of the pulse class

        '''

        if interpolation==None:

            self.waveform = _resample(self.tbase, self.waveform, nSamples, interpolation=interpolation);

            if self._digitized:
                self.digiWaveform = _resample(self.tbase, self.digiWaveform, nSamples, interpolation=interpolation); #TODO: check if this works fine. Otherwise re-digitalize the waveform

            self.tbase = np.linspace(np.min(self.tbase), np.max(self.tbase), nSamples)
        else:

            _tbase = copy.deepcopy(self.tbase)
            self.tbase = np.linspace(np.min(self.tbase), np.max(self.tbase), nSamples)
            
            _interpolation = _resample(_tbase, self.waveform, nSamples, interpolation=interpolation);
            self.waveform = _interpolation(self.tbase)

            if self._digitized:
                _dinterpolation = _resample(_tbase, self.digiWaveform, nSamples, interpolation=interpolation); #TODO: check if this works fine. Otherwise re-digitalize the waveform
                self.digiWaveform = _dinterpolation(self.tbase)

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
                    try:
                        if function_real[i]>=wf_real[int(i+tstart_index)]:
                            wf_real[int(i+tstart_index)] = function_real[i];
                            wf_imag[int(i+tstart_index)] = function_imag[i];
                        if function_real[i]<wf_real[int(i+tstart_index)]:
                            wf_real[int(i+tstart_index)] = wf_real[int(i+tstart_index)]+function_real[i];
                            wf_imag[int(i+tstart_index)] = wf_imag[int(i+tstart_index)]+function_imag[i];
                    except:
                        pass


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

    def combinePulses(self, PulseLST):
        
        _nlabel = 1;

        for _newpulse in PulseLST:
            #concatenate waveforms
            self.waveform = np.concatenate((self.waveform, _newpulse.waveform));

            #concatenate digiwaveforms
            self.digiWaveform = np.concatenate((self.digiWaveform, _newpulse.digiWaveform));

            #joint func lists
            self._functionsLST+= _newpulse._functionsLST;

            #joint func dict
            for _idx, _function in zip(_newpulse._combfunctionLST['t_idx'], _newpulse._combfunctionLST['function']):
                self._combfunctionLST['t_idx'].append(_idx + len(self.tbase));
                self._combfunctionLST['function'].append(_function);

            #extend timeflags
            self._timeFlags+=[len(self.tbase)+_flag for _flag in _newpulse._timeFlags]

            #extend tbase
            _newMIN = min(self.tbase);
            _newMAX = max(self.tbase) + max(_newpulse.tbase);
            _newSTEPS = len(self.tbase) + len(_newpulse.tbase);

            self.tbase = np.linspace(_newMIN, _newMAX, _newSTEPS);

            #fix label
            if self.label==_newpulse.label:
                _nlabel+=1;
                self.label+= ' x{rep}'.format(rep=_nlabel);  
            else: 
                self.label+= '+'+_newpulse.label;

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
    
    def getTbase(self):
        return self.tbase

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

                if isinstance(self.digiWaveform[0], np.complex128):
                    plt.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    plt.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');

            else:
                plt.step(self.tbase, self.digiWaveform, alpha=0.9, color=_color);
                plt.fill(self.tbase, self.digiWaveform, alpha=0.3, color=_color);

                if isinstance(self.digiWaveform[0], np.complex128):
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

                if isinstance(self.digiWaveform[0], np.complex128):
                    axis.step(self.tbase, np.imag(self.digiWaveform), alpha=0.9, color='red');
                    axis.fill(self.tbase, np.imag(self.digiWaveform), alpha=0.3, color='red');
            else:
                axis.step(self.tbase, np.real(self.digiWaveform), alpha=0.9, color=_color);
                axis.fill(self.tbase, np.real(self.digiWaveform), alpha=0.3, color=_color);

                if isinstance(self.digiWaveform[0], np.complex128):
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

    def __init__(self, sequence_label, tSequence=None, stack=None, variables=None, nr_cycles=1):
        
        #atributes
        self._res = type;
        self.label = sequence_label;
        self.tSequence = tSequence;
        self.nr_cycles = nr_cycles;

        if self.nr_cycles==1:
            self._Stack = stack;
        else:
            try:
                _stack=[]
                for _stack_element in stack:
                    _stack_element.combinePulses([_stack_element]*(self.nr_cycles-2));
                    _stack.append(_stack_element);

                self._Stack = _stack;
            except:
                pass

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

                try:
                    self._API_sequence += [[p[1], p[2]] for p in stack];
                except:
                    pass

            elif all(isinstance(t, track) for t in self._Stack):
                self.digiWaveformStack = [t.getTrack() for t in self._Stack];

    def stack2Sequence(self, stack, _IDs:bool=False):
        if self._Stack is []:
            
            if _IDs==False:
                if all(isinstance(p, pulse) for p in stack) or all(isinstance(t, track) for t in stack):
                    self._Stack = stack;
                    
                    if all(isinstance(p, pulse) for p in self._Stack):

                        _stack=[];
                        for _stack_element in self._Stack:
                            _stack_element.combinePulses([_stack_element]*(self.nr_cycles-1));
                            _stack.append(_stack_element);
                        self._Stack=_stack;

                        self.tSequence = np.zeros( max([len(w.getPulse()) for w in stack]) );
                        self.digiWaveformStack = [p.getPulse() for p in stack];

                    elif all(isinstance(t, track) for t in self._Stack):
                        self.tSequence = np.zeros( max([len(w.getTrack()) for w in stack]) );
                        self.digiWaveformStack = [t.getTrack() for t in self.stack];
            else:
                if all(isinstance(p[0], pulse) for p in stack) or all(isinstance(t[0], track) for t in stack):
                    self._Stack = stack;
                    
                    if all(isinstance(p[0], pulse) for p in self._Stack):
                        
                        _stack=[];
                        for _stack_element in self._Stack:
                            _stack_element.combinePulses([_stack_element]*(self.nr_cycles-1));
                            _stack.append(_stack_element);
                        self._Stack=_stack;

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

                    _stack=[];
                    for _stack_element in stack:
                        _stack_element[0].combinePulses([_stack_element[0]]*(self.nr_cycles-1));
                        _stack.append(_stack_element[0]);

                    self.digiWaveformStack += [p.getPulse() for p in _stack];
                    self._Stack+=stack;

                elif all(isinstance(t, track) for t in stack):
                    self.digiWaveformStack += [t.getTrack() for t in stack];
                    self._Stack+=stack;
            else:
                if all(isinstance(p[0], pulse) for p in stack):

                    _stack=[];
                    for _stack_element in stack:
                        _buf_stack_element = copy.deepcopy(_stack_element[0]);
                        _stack_element[0].combinePulses([_buf_stack_element]*(self.nr_cycles-1));
                        _stack.append(_stack_element[0]);

                    self.digiWaveformStack += [p.getPulse() for p in _stack];
                    self._Stack+=_stack;

                elif all(isinstance(t[0], track) for t in stack):
                    self.digiWaveformStack += [t[0].getTrack() for t in stack];
                    self._Stack+=stack;

                self._Stack = [element for element in stack];

                try:
                    self._API_sequence += [[p[1], p[2]] for p in stack];
                    self._Stack=[p[0] for p in stack];
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

###################################################################################################
#######################                 Backend Control                  ##########################
###################################################################################################


#####################################################################################################
#IAC (IntegratedAtomicChip) AQiPT class
#####################################################################################################

class IAC:

    def __init__(self, HWgrid, substrate_size=(800,800)):

        self._HWgeometry = HWgrid
        self._filledArray = None
        self._emptyArray = None

        self._HW4dynamics = None
        self._HW4statics = None
        self._HW4gates = None

        self._historical_sites = []
        self._site_params = None
        self._site = None

        self._historical_arrays = []
        self._array_params = None
        self._geometry = None

        self._nr_of_qudits = None
        self._qudit_map = {}
        self._nr_of_registers = 0
        self._qregisters = {}

        self._historical_drags = []
        self._drag_params = []
        self._drags = None

        self._substrate_size = substrate_size
        self._substrate = None

        self._realSubstrate = None

        self._cmap = None

        self._center_frequency = None
        self._center_position = None
        self._resolution = None
        self._resolution_r = None
        self._resolution_f = None


    def _set_site(self, size, bitdepth, profile_type='flat', profile_params=None, output=False):
        '''
            Generates a 2D matrix with values with a given bitdepth, following a gradient profile.

            Parameters:

            size (tuple): A tuple (height, width) specifying the size of the matrix to generate.
            bitdepth (int): The bitdepth of the matrix, i.e., the number of bits used to represent each value.
            profile_type (str): The type of gradient profile to use. Can be one of 'flat' or 'gaussian'.
            profile_params (dict): A dictionary of additional parameters for the gradient profile.
                                   If profile_type is 'gaussian', this dictionary should contain the mean, covariance, and standard deviation of the Gaussian.
                                   It can also optionally contain the maximum number of pixels allowed in the Gaussian region (max_pixels).

            Returns:

            ndarray: A 2D matrix of shape (height, width) with values of the specified bitdepth, following the selected gradient profile.
        
        '''

        #create an array of coordinates for the matrix
        y, x = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij');
        pos = np.empty((size[0], size[1], 2));
        pos[:, :, 0] = y;
        pos[:, :, 1] = x;

        #create the gradient profile
        if profile_type == 'flat':
            max_value = 2**bitdepth - 1;
            
            if 'center' in profile_params and 'radius' in profile_params:
                
                center = profile_params['center'];
                radius = profile_params['radius'];
                dist = np.sqrt((y - center[0])**2 + (x - center[1])**2);
                gradient = np.where(dist <= radius, max_value, 0);
                
            else:
                
                gradient = np.ones(size) * max_value;
                
        elif profile_type == 'gaussian':
            
            center = profile_params.get('center', (size[0]/2, size[1]/2));
            cov = profile_params.get('cov', [[1, 0], [0, 1]]);
            std = profile_params.get('std', [1, 1]);
            cov = np.array(cov) * np.array(std)**2;
            rv = multivariate_normal(center, cov);
            gaussian = rv.pdf(pos);
            max_pixels = profile_params.get('max_pixels', None);
            
            if max_pixels is not None:
                
                num_pixels = np.sum(gaussian > np.max(gaussian) / 2**bitdepth);
                
                if num_pixels > max_pixels:
                    
                    gaussian *= max_pixels / num_pixels;
                    
            max_value = 2**bitdepth - 1;
            max_gaussian = rv.pdf(center);
            scale_factor = max_value / max_gaussian;
            gradient = gaussian * scale_factor;

        __site = gradient.astype(np.uint16)

        if PYTHON_VERSION[0]=='3' and int(PYTHON_VERSION[2])>=9:
            self._site_params = {'size': size, 'bitdepth': bitdepth, 'profile_type':profile_type} | profile_params
        elif PYTHON_VERSION[0]=='3' and int(PYTHON_VERSION[2])>=5:
            self._site_params = {**{'size': size, 'bitdepth': bitdepth, 'profile_type':profile_type}, **profile_params}

        self._site = __site
        self._historical_sites.append(__site)

        if output:
            return __site

    def _rectangular_array(self, m, n, mesh_dim, sep_x, sep_y, origin=(0, 0), output=False):
        '''
            Generates a list of 2D coordinate tuples (x, y) for a rectangular array of sites,
            given the dimensions m by n number of sites, the total dimension size of the hypothetical mesh,
            the constant separation in X and Y between points of the array, and the origin point.

            Parameters:
                - m (int): number of rows in the rectangular array
                - n (int): number of columns in the rectangular array
                - mesh_dim (int): total dimension size of the hypothetical mesh (assumes square mesh)
                - sep_x (float): constant separation in X between points of the array
                - sep_y (float): constant separation in Y between points of the array
                - origin (tuple of floats): starting point (x, y) for the first point in the array (default: (0, 0))

            Returns:
                - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''

        x0, y0 = origin
        coords = []
        for i in range(m):
            for j in range(n):
                x = j * sep_x + x0
                y = i * sep_y + y0
                
                if x < 0 or x > mesh_dim[0] or y < 0 or y > mesh_dim[1]:
                    warnings.warn(f"Generated coordinate ({x}, {y}) is outside the mesh dimensions ({mesh_dim[0]}, {mesh_dim[1]}).")
                
                coords.append((x, y))

        self._array_params = {'dimension m': m, 'dimension n': n, 'full mesh':mesh_dim, 
                              'x resolution': sep_x, 'y resolution': sep_y, 'origin': origin, 
                              'geometry type': 'rectangular'}
        self._geometry = coords
        self._historical_arrays.append(coords)

        if output:
            return coords

    def _triangular_array(self, m, n, mesh_dim, sep, origin=(0, 0), output=False):
        '''
            Generates a list of 2D coordinate tuples (x, y) for a honeycomb lattice,
            given the dimensions m by n number of sites, the total dimension size of the hypothetical mesh,
            the separation between points of the lattice, and the origin point.

            Parameters:
            - m (int): number of rows in the honeycomb lattice
            - n (int): number of columns in the honeycomb lattice
            - mesh_dim (tuple of floats): total dimension size of the hypothetical mesh (mesh_dim_x, mesh_dim_y)
            - sep (float): separation between points of the lattice
            - origin (tuple of floats): starting point (x, y) for the first point in the lattice (default: (0, 0))

            Returns:
            - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''
        n=int(0.5*n);
        mesh_dim_x, mesh_dim_y = mesh_dim
        x0, y0 = origin
        coords = []
        for i in range(m):
            for j in range(n):
                x = j * np.sqrt(3) * sep + (i % 2) * np.sqrt(3) * sep / 2 + x0
                y = i * 3/2 * sep + y0
                if x < 0 or x > mesh_dim_x or y < 0 or y > mesh_dim_y:
                    warnings.warn(f"Generated coordinate ({x}, {y}) is outside the mesh dimensions ({mesh_dim_x}, {mesh_dim_y}).")
                coords.append((x, y))

        self._array_params = {'dimension m': m, 'dimension n': n, 'full mesh':mesh, 
                              'resolution': sep, 'origin': origin, 
                              'geometry type': 'triangular'}
        self._geometry = coords
        self._historical_arrays.append(coords)

        if output:
            return coords

    def _honeycomb_array(self, m, n, r, output=False):

        b=2*r*np.sqrt( 1 - 4*np.sin(np.pi/6)**2)
        a=b/2

        coords = create_hex_grid(nx=m, ny=n, min_diam=r/0.5776, do_plot=False)  # Create 5x5 grid with no gaps

        def generate_hexagon_points(center_x, center_y, radius):
            points = []
            for i in range(6):
                angle_deg = (60) * i +90
                angle_rad = np.pi / 180 * angle_deg
                x = center_x + radius * np.cos(angle_rad)
                y = center_y + radius * np.sin(angle_rad)
                points.append((x, y))
            return points
        
        plt.figure()
        for coord in coords[0]:
            plt.scatter(coord[0], coord[1], color='red', marker='o')
        
        v_values = [];
        for j in range(m):
            for i in range(n):
                plt.scatter(coords[0][i+m*j][0], coords[0][i+n*j][1], color='violet', marker='o')
                vertexes = generate_hexagon_points(center_x=coords[0][i+m*j][0]+(b*i)+(a*j), 
                                                   center_y=coords[0][i+n*j][1]+(a*i)+(b*j), 
                                                   radius=r)
                v_values+=vertexes;
                
                for vertex in vertexes:
                    plt.scatter(vertex[0], vertex[1], color='blue', marker='.')


        self._array_params = {'dimension m': m, 'dimension n': n, 'radius':r, 
                              'geometry type': 'honeycomb'}
        self._geometry = v_values
        self._historical_arrays.append(v_values)
        
        if output:            
            return v_values

    def _ring_array(self, mesh_dim, N, R, alpha_0=0, origin=(0, 0), output=False, plotON=False):
        '''
            Generates a list of 2D coordinate tuples (x, y) for a ring array of N sites,
            over a circumference of radius R, equally spaced over the circunference line 
            (2pi/N) starting at angle alpha_0.

            Parameters:
                - N (float): number of points along the ring
                - R (float): radius of the ring
                - alpha_0 (float): initial angle for placing the points
            Returns:
                - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''

        angles = np.linspace(alpha_0, alpha_0 + 2 * np.pi, N, endpoint=False)
        x_coords = [R * np.cos(angle) + origin[0] for angle in angles]
        y_coords = [R * np.sin(angle) + origin[1] for angle in angles]
        coords = list(zip(x_coords, y_coords))

        self._array_params = {'coordinates':coords,
                              'full mesh': mesh_dim,
                              'nr of sites': N,
                              'radius': R, 
                              'initial angle': alpha_0, 
                              'rotations': [],
                              'origin': origin,
                              'geometry type': 'ring'}
        self._geometry = coords
        self._historical_arrays.append(coords)

        if plotON:
            plt.figure(figsize=(6,6))
            plt.plot(x_coords, y_coords, 'o', label='Original Points')
            plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'k--', lw=0.5)
            for i, (x, y) in enumerate(coords):
                plt.text(x, y, f'{i}', fontsize=9, ha='right')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Ring with {N} Points and Radius {R}')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.legend()
            plt.show()

        if output:
            return coords
    
    def _expand_ring(self, coordinates, d, output=False, plotON=False):
        '''
            Expand the ring coordinates by a distance d.

            Parameters:
                - coordinates (array): initial coordinates
                - d (float): distance displaced summed to the original radius
            Returns:
                - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''

        x, y = coordinates[0]

        R = self._array_params['radius']
        new_radius = R + d
        angles = [np.arctan2(y-self._array_params['origin'][1], x-self._array_params['origin'][0]) for x, y in coordinates]
        expanded_coordinates = [(new_radius * np.cos(angle)+self._array_params['origin'][0], new_radius * np.sin(angle)+self._array_params['origin'][1]) for angle in angles]

        self._array_params['radius'] = new_radius
        self._geometry = expanded_coordinates
        self._historical_arrays.append(expanded_coordinates)

        if plotON:
            original_x, original_y = zip(*coordinates)
            expanded_x, expanded_y = zip(*expanded_coordinates)

            plt.figure(figsize=(6,6))
            plt.plot(original_x, original_y, 'o', label='Original Points')
            plt.plot([original_x[-1], original_x[0]], [original_y[-1], original_y[0]], 'k--', lw=0.5)
            for i, (x, y) in enumerate(coordinates):
                plt.text(x, y, f'{i}', fontsize=9, ha='right')

            plt.plot(expanded_x, expanded_y, 'o', label='Expanded Points', color='green')
            plt.plot([expanded_x[-1], expanded_x[0]], [expanded_y[-1], expanded_y[0]], 'g--', lw=0.5)
            for i, (x, y) in enumerate(expanded_coordinates):
                plt.text(x, y, f'{i}', fontsize=9, ha='left', color='green')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Original and Expanded Points')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.legend()
            plt.show()

        if output:
            return expanded_coordinates

    def _rotate_ring(self, coordinates, beta, output=False, plotON=False):
        
        '''
            Rotate the ring coordinates by an angle beta.

            Parameters:
                - coordinates (array): initial coordinates
                - beta (float): rotation angle performed over the original radius
            Returns:
                - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''



        rotation_matrix = np.array([ [np.cos(beta), -np.sin(beta)],[np.sin(beta), np.cos(beta)] ])
        rotated_coordinates = []
        for (x, y) in coordinates:
            rotated_point = rotation_matrix @ np.array([x-self._array_params['origin'][0], y-self._array_params['origin'][1]])
            rotated_coordinates.append((rotated_point[0]+self._array_params['origin'][0], rotated_point[1]+self._array_params['origin'][1]))

        self._array_params['rotations'].append(beta)
        self._geometry = rotated_coordinates
        self._historical_arrays.append(rotated_coordinates)

        if plotON:
            original_x, original_y = zip(*coordinates)
            rotated_x, rotated_y = zip(*rotated_coordinates)

            plt.figure(figsize=(6,6))
            plt.plot(original_x, original_y, 'o', label='Original Points')
            plt.plot([original_x[-1], original_x[0]], [original_y[-1], original_y[0]], 'k--', lw=0.5)
            for i, (x, y) in enumerate(coordinates):
                plt.text(x, y, f'{i}', fontsize=9, ha='right')

            plt.plot(rotated_x, rotated_y, 'o', label='Rotated Points', color='red')
            plt.plot([rotated_x[-1], rotated_x[0]], [rotated_y[-1], rotated_y[0]], 'r--', lw=0.5)
            for i, (x, y) in enumerate(rotated_coordinates):
                plt.text(x, y, f'{i}', fontsize=9, ha='left', color='red')

            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Original and Rotated Points')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.legend()
            plt.show()

        if output:
            return rotated_coordinates

    def _drag_geometry(self, A, B, M, gtype='ring', gif_filename="sweep.gif", output=False):

        '''
            Rotate the ring coordinates by an angle beta.

            Parameters:
                - coordinates (array): initial coordinates
                - beta (float): rotation angle performed over the original radius
            Returns:
                - coords (list of tuples): list of 2D coordinate tuples (x, y)
        '''

        if gtype=='ring':
            frames = []
            for t in np.linspace(0, 1, M):
                intermediate_coordinates = [( (1-t) * x1 + t * x2, (1-t) * y1 + t * y2 ) for (x1, y1), (x2, y2) in zip(A, B)]
                frames.append(intermediate_coordinates)

            self._drag_params.append({'initial coordinates': A, 'final coordinates': B, 'number of frames': M, 'geometry type': gtype, 'file name GIF': gif_filename})
            self._historical_drags.append(frames)
            self._drags = frames

            if output:
                return frames

            images = []
            for frame in frames:
                fig, ax = plt.subplots(figsize=(6,6))
                A_x, A_y = zip(*A)
                frame_x, frame_y = zip(*frame)

                ax.plot(A_x, A_y, 'o', label='Start Points')
                ax.plot([A_x[-1], A_x[0]], [A_y[-1], A_y[0]], 'k--', lw=0.5)
                ax.plot(frame_x, frame_y, 'o', label='Sweep Points', color='blue')
                ax.plot([frame_x[-1], frame_x[0]], [frame_y[-1], frame_y[0]], 'b--', lw=0.5)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'Sweep Frame')
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True)
                ax.legend()

                plt.tight_layout()
                filename = f"frame_{len(images)}.png"
                plt.savefig(filename)
                images.append(filename)
                plt.close(fig)

            with imageio.get_writer(gif_filename, mode='I', duration=0.1, loop=0) as writer:
                for image in images:
                    writer.append_data(imageio.imread(image))

            for image in images:
                os.remove(image)

            print(f"GIF saved as {gif_filename}")

    def _arbitrary_array(self, coordinates, mesh_dim):

        self._array_params = {'coordinates':coordinates,
                              'full mesh': mesh_dim,
                              'nr of sites': len(coordinates),
                              'geometry type': 'arbitrary'}
        self._geometry = coordinates
        self._historical_arrays.append(coordinates)

    def _geometry_assembler(self, figure_size=(10,10), plotON=True, cmap='jet'):

        '''
            Generates the geometry assemble of the substrate if integrated Atomic Chip (IAC).

            sets the image to give to the wavefront light modulator for generating the patterns for
            the array of optical tweezers.

            plotON (bool) : boolean for plotting or not.
            cmap (str) : matplotlib color map of the showed figure of the substrate
        '''

        _substrate = np.zeros(self._substrate_size)

        site_path = self._site

        #substitute the site generate for the IAC in the respectivatly coordinate of the image.
        for coord in self._geometry:
            if coord[0]>=0 and coord[0]<=self._array_params['full mesh'][1]: 
                if coord[1]>=0 and coord[1]<=self._array_params['full mesh'][0]:
                    
                    _x, _y = (coord[0], coord[1])            
                    cx, cy = site_path.shape
                    cx = 0.5*(cx) ; cy = 0.5*(cy);
                    
                    _substrate[ int(_y+1-cy):int(_y+1+cy), int(_x+1-cx):int(_x+1+cx)]=self._site
                    
        self._substrate = _substrate

        self._set_qudit_map() #set the qudit map with the coordinates of each qubit

        if plotON:
            self.showSubstrate(figure_size= figure_size, color_map=cmap)

    def combine_coordinates(self, coordinates_A, coordinates_B):

        combined_array = coordinates_A+coordinates_B
        self._geometry = combined_array
        self._historical_arrays.append(combined_array)

    def set_qregister(self, qregisters:dict, center_frequency=None, resolution=None, center_position=None):
        '''
            Set the quantum registers with their index and the coordinates from the geometry. It does
            require to have set a qudit map with _set_qudit_map().

            qregisters (dict) : dictionary containing the qregister number and a list of indexes, link
                                to the indexes from the coordinates of the geometry. e.g., {'qr0': [0, 3, 4], ...}

            Returns:

            A dictionary with indexes and coordinates. e.g., {'qr0': [ [index 1, index 2, ...], [coordinate 1, coordinate 2, ...] ], }
        '''


        for key, val in qregisters.items():
            # print(key, val)
            self._qregisters[key] = [val, [self._geometry[idx] for idx in val] ]

        self._nr_of_registers = len(qregisters)

        self._set_qudit_map()

        if center_frequency!=None and resolution!=None and center_position!=None:
            self.set_freq_map( center_frequency, center_position, resolution)

    def _set_qudit_map(self):

        '''
            Set the qudit map with the assigned coordinate to the correspondant qubit label.
        '''

        self._nr_of_qudits = len(self._geometry)        
        
        for _ in range(self._nr_of_qudits):
            self._qudit_map['q_'+str(_)] = self._geometry[_]

    def set_freq_map(self, center_frequency, center_position, resolution):
        '''
            Function to transform from the position coordinates to frequency map coordinates

            center_frequency (tuple) : horizontal (X) and vertical (Y) origin/reference points
            resolution (tuple) : position and frequency resolution if 1 element is given then is given in 
                                units of px/MHz or similar units, if two are given, then the ratio is cal-
                                culated (1px, 0.5MHz)-> 0.5MHz/px resolution
            centerposition (tuple) : 
        '''

        self._center_frequency = center_frequency
        self._center_position = center_position

        if len(resolution)==2:
            self._resolution_f = resolution[1]
            self._resolution_r = resolution[0]
            self._resolution = self._resolution_f/self._resolution_r
            
        else:
            self._resolution = resolution[0]


        if self._nr_of_registers!=0:

            for key, val in self._qudit_map.items():

                _xposition_reg_i = np.rint( (np.abs(self._qudit_map[key][0] - self._center_position[0])*(self._resolution) + (self._center_frequency[0])) / (self._resolution_f) ) * (self._resolution_f)
                _yposition_reg_i = np.rint( (np.abs(self._qudit_map[key][1] - self._center_position[1])*(self._resolution) + (self._center_frequency[1])) / (self._resolution_f) ) * (self._resolution_f)

                self._qudit_map[key] = {'positions [px]': list(self._qudit_map[key]), 'frequencies': [_xposition_reg_i, _yposition_reg_i]}

    def get_freq_qreg_map(self):

        _map ='frequencies'

        return [self._qudit_map[key][_map] for key,vals in self._qudit_map.items()]

    def get_pos_qreg_map(self):

        _map ='positions [px]'
        
        return [self._qudit_map[key][_map] for key,vals in self._qudit_map.items()]

    def showCoordinates(self, figure_size=(10,10)):

        plt.figure(figsize=figure_size, dpi=150)
        for x,y in self._geometry:
            plt.scatter(x,y, s=80, facecolors='none', edgecolors='red',linestyle='--',linewidths=0.4)
        plt.title('Integrated Atomic Circuit (geometry)')
        plt.xlabel('dimension X')
        plt.ylabel('dimension Y')
        plt.show() 

    def showSubstrate(self, figure_size=(10,10), color_map='jet'):

        self._cmap = color_map

        plt.figure(figsize=figure_size, dpi=150)
        plt.imshow(self._substrate, origin='lower', aspect='auto', cmap=color_map)
        plt.colorbar()
        plt.title('Integrated Atomic Circuit (substrate)')
        plt.xlabel('dimension X')
        plt.ylabel('dimension Y')
        plt.show()            

    def showSite(self, figure_size=(10,10), DPI=150, color_map='bone'):

        plt.figure(figsize=figure_size, dpi=DPI)
        plt.imshow(self._site, aspect='auto', cmap=color_map)
        plt.colorbar()
        plt.title('Tweezer site (substrate)')
        plt.xlabel('dimension X')
        plt.ylabel('dimension Y')
        plt.show() 

    def get_GIF(self, path, temp_fname, frames, mesh_dim, DPI=150, figure_size=(10,10), color_map='bone', labels={'xlabel': 'Dimension X [pixel]', 'ylabel': 'Dimension Y [pixel]', 'title': 'Drag movements'}, duration=0.1, noise=False):

        images=[]
        for i in tqdm(range(10)):
            self._arbitrary_array(frames[i], mesh_dim=mesh_dim)
            self._geometry_assembler(plotON=False) 
            
            fig = plt.figure(figsize=figure_size, dpi=DPI)

            if noise:
                plt.imshow(noisify(self._substrate,180), origin='lower', aspect='auto', cmap=color_map)
            else:
                plt.imshow(self._substrate, origin='lower', aspect='auto', cmap=color_map)

                plt.xlabel(labels['xlabel'])
                plt.ylabel(labels['ylabel'])
                plt.title(labels['title'])

                fig.savefig(temp_fname)
                
                image = imageio.imread(temp_fname)
                images.append(image)
                plt.close('all')

        imageio.mimsave(path, images, duration=duration, loop=0)

    #generate a target configuration as a k x j grid
    def generate_grid(self, k, j, spacing=10):
        x_coords = np.linspace(100, IMAGE_SIZE[0]-100, j)
        y_coords = np.linspace(100, IMAGE_SIZE[1]-100, k)
        target_positions = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
        return target_positions

    #function for assigning each agent to the optimal target based on distance distribution
    def assign_targets_optimal(self, initial_positions, target_positions):
        
        #compute distance matrix (agents x target positions)
        distance_matrix = cdist(initial_positions, target_positions)
        
        #solve the assignment problem to minimize the total distance
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        #yields assigned target positions for each agent
        assigned_targets = target_positions[col_ind]
        return assigned_targets

    def optimize_agent_positions(self, initial_positions,assigned_targets,NUM_AGENTS,CONVERGENCE_THRESHOLD,MAX_STEP_SIZE,COLLISION_DISTANCE):
        '''
            Optimizes the positions of agents towards their assigned targets with collision avoidance.

            INPUTS:
            ------
            initial_positions (np.ndarray): Initial positions of agents.
            assigned_targets (np.ndarray): Target positions assigned to agents.
            NUM_AGENTS (int): Number of agents.
            CONVERGENCE_THRESHOLD (float): Distance threshold to consider an agent has reached its target.
            MAX_STEP_SIZE (float): Maximum step size an agent can move in one iteration.
            COLLISION_DISTANCE (float): Minimum allowable distance between agents to avoid collisions.

            OUTPUT:
            ------
            list: A list of position arrays for each generation.
        '''
        #check if each agent has reached its target
        converged_agents = np.zeros(NUM_AGENTS, dtype=bool)

        #initialize arrays to store positions for each generation
        positions_per_generation = [initial_positions.copy()]

        #optimization loop: move agents towards assigned targets with limited step size and collision avoidance
        while not np.all(converged_agents):
            current_positions = positions_per_generation[-1].copy()

            for i in range(NUM_AGENTS):
                if not converged_agents[i]:
                    #compute distance and direction to target
                    distance_to_target = np.linalg.norm(assigned_targets[i] - current_positions[i])

                    #check if the agent has reached its target
                    if distance_to_target < CONVERGENCE_THRESHOLD:
                        converged_agents[i] = True
                        continue

                    #move towards the target with limited step size
                    direction = (assigned_targets[i] - current_positions[i]) / distance_to_target
                    step = min(MAX_STEP_SIZE, distance_to_target)
                    new_position = current_positions[i] + direction * step

                    #check for collisions with other agents
                    for j_idx in range(NUM_AGENTS):
                        if i != j_idx and not converged_agents[j_idx]:
                            #calculate distance to the other agent
                            distance_to_other = np.linalg.norm(new_position - current_positions[j_idx])
                            if distance_to_other < COLLISION_DISTANCE:
                                #adjust movement to avoid collision
                                direction_to_other = (current_positions[j_idx] - current_positions[i]) / distance_to_other
                                avoidance_step = (COLLISION_DISTANCE - distance_to_other) / 2
                                new_position -= direction_to_other * avoidance_step

                    #update the agent's position
                    current_positions[i] = new_position

            #store new positions for this generation
            positions_per_generation.append(current_positions)

        return positions_per_generation

    def update(self, frame_tuple):
        frame_num, frame = frame_tuple
        ax_move.clear()
        ax_dist.clear()

        #tracks 
        for i in range(NUM_AGENTS):
            track = np.array(tracks[i][:frame_num + 1])
            ax_move.plot(track[:, 0], track[:, 1], color='gray', alpha=0.3)

        #initial and target positions
        ax_move.scatter(target_positions[:, 0], target_positions[:, 1], c='green', label='Target Positions', s=4)
        ax_move.scatter(frame[:, 0], frame[:, 1], c='blue', label='Moving Agents', s=4)
        ax_move.set_xlim([0, IMAGE_SIZE[0]])
        ax_move.set_ylim([0, IMAGE_SIZE[1]])
        ax_move.legend(loc='upper right')
        ax_move.set_title(f'Agents Moving with Optimal Target Assignment (Frame {frame_num})')

        #cummulative distances
        for i in range(NUM_AGENTS):
            ax_dist.plot(np.arange(frame_num + 1), cumulative_distances[i, :frame_num + 1],
                         label=f'Agent {i+1}', color=colors(i), alpha=0.5)

        ax_dist.set_title('Distance Traveled vs Frame # for Each Agent')
        ax_dist.set_xlabel('Frame #')
        ax_dist.set_ylabel('Distance Traveled')
        ax_dist.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    #track positions and cumulative distances
    def track_agent_movements(self, positions_per_generation, NUM_AGENTS):
        '''
        Tracks the movements of agents across generations and calculates cumulative distances.

        Parameters:
        positions_per_generation (list): List of position arrays for each generation.
        NUM_AGENTS (int): Number of agents.

        Returns:
        tuple: A list of tracks for each agent and a numpy array of cumulative distances.
        '''
        #initialize tracks and cumulative distances
        tracks = [[] for _ in range(NUM_AGENTS)]
        cumulative_distances = np.zeros((NUM_AGENTS, len(positions_per_generation)))

        #iterate over each generation of positions
        for frame_num, frame in enumerate(positions_per_generation):
            #calculate distances from the previous frame
            if frame_num > 0:
                distances = np.linalg.norm(frame - positions_per_generation[frame_num - 1], axis=1)
                cumulative_distances[:, frame_num] = cumulative_distances[:, frame_num - 1] + distances
            #append positions to tracks
            for i in range(NUM_AGENTS):
                tracks[i].append(frame[i])

        return tracks, cumulative_distances


#####################################################################################################
#Grid AQiPT class
#####################################################################################################
class Grid:
    '''
    A class representing a grid for managing coordinates, amplitudes, frequencies, phases, and configurations.

    Attributes:

        grid_type (str) : 
        _nrLines (int) : The number of lines in the grid.
        _resolution (float) : 
        _dimensions (tuple) : 
        _coordinates (dict) : 
        _amplitudes (None) : The amplitudes of the grid points.
        _frequencies (None) : The frequencies of the grid points.
        _phases (None) : The phases of the grid points.
        _sequenceSweeps (None) : The sequence of sweeps for each grid point.
        initialConfiguration (None) : The initial configuration of the grid.
        finalConfiguration (None) : The final configuration of the grid.
        currentConfiguration (None) : The current configuration of the grid.
    
    Methods:

        _mapper :
        showMap :
        get_amplitudes :
        get_coordinates :
        get_frequencies :
        get_phases :
        get_nrLines :
        get_sequenceSweeps :
        get_initialConfiguration :
        get_currentConfiguration :
        get_finalConfiguration :

    '''

    def __init__(self, grid_type='square', nr_Lines=2, resolution=None):
        '''
            Initialize a Grid object with default attributes.
        '''
        self.grid_type = grid_type
        self._nrLines = nr_Lines

        self._resolution = resolution

        if self.grid_type=='square': 

            if self._nrLines%2==0:
                self._dimensions = (self._nrLines+1, self._nrLines+1)
                self._nrLines = nr_Lines+1
                print('Only odd numbers of devices/lines')

            else:
                self._dimensions = (self._nrLines, self._nrLines)

            self._coordinates = {'frequency_map': np.zeros(self._dimensions, dtype=np.complex128),
                                 'phase_map': np.zeros(self._dimensions),
                                 'amplitude_map': np.zeros(self._dimensions),
                                 'sweep_map': np.zeros(self._dimensions),
                                 'position': np.zeros(self._dimensions)}

            if self._resolution is None:
                self._mapper()
            else:
                self._mapper(self._resolution)


        elif grid_type=='rectangular':
            print('Not implemented, sorry')

        elif grid_type=='Hexagonal':
            print('Not implemented, sorry')

        elif grid_type=='Triangular':
            print('Not implemented, sorry')

        elif grid_type=='Arbitrary':
            print('Not implemented, sorry')

        # deprecated (?)
        self._amplitudes = None
        self._frequencies = None
        self._phases = None
        self._sequenceSweeps = None
        self.initialConfiguration = None
        self.finalConfiguration = None
        self.currentConfiguration = None

    def _mapper(self, resolution=0.5e6):
        '''
            Mapper to define the grid that works as LUT mapping the indexes with the frequencies [MHz]
            for each position with index (x,y).

            resolution (float) : frequency spacing [MHz] between sites 3.5um <--> 0.5MHz

            Note: all the maps are stored in self._coordinates e.g., frequency, phase, amplitude, sweep and positions

            TODO: other maps, only frequency map has been made
        '''

        self.resolution = resolution

        if self.grid_type == 'square':
            for x_idx in range(self._nrLines):
                for y_idx in range(1, int((self._nrLines-1)/2)+1):
                    self._coordinates['frequency_map'][mt.floor(self._nrLines/2) + y_idx, x_idx] = self.resolution*y_idx
                    self._coordinates['frequency_map'][mt.floor(self._nrLines/2) - y_idx, x_idx] = self._coordinates['frequency_map'][mt.floor(self._nrLines/2) + y_idx, x_idx]

    def showMap(self, typeMap='frequency', fig_size=(8,8), fig_dpi=200, labels=True, colomarp='bone'):

        '''
            Maps plotter, it does show the map correspondency used for configuring oscillators (i.e., NCOs)

            typeMap (str) : define which maps are shown
            fig_size (tuple) : size of the figure
            fig_dpi (int) : value of the density of pixel per inch in the figure
            labels (bool) : if sites values are shown
            colormap (str) : colormap for showing the plot

        '''

        if typeMap=='frequency':

            fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
            im = plt.imshow(np.real(self._coordinates['frequency_map']), cmap=colomarp, origin='lower', vmin=np.min(np.real(self._coordinates['frequency_map'])), vmax=np.max(np.real(self._coordinates['frequency_map'])))

            data = np.real(self._coordinates['frequency_map'])

            if labels:
                # Add text annotations on top of each pixel
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        plt.annotate(str([i, j])+"\n"+ '{0:.2f}'.format((data[i,j])*1e-6)+ "[MHz]", xy=(j, i),
                                     color='black', ha='center', va='center',
                                     fontsize=8)
            plt.xlabel('Line index x-axis', fontsize=30)
            plt.ylabel('Site index y-axis', fontsize=30)
            plt.title('Frequency map', fontsize=35)
            cbar = plt.colorbar(im)
            cbar.set_label('Frequency [MHz]', fontsize=30)

        
    @property
    def get_amplitudes(self):
        '''
        Get the amplitudes of the grid points.

        Returns:
            list: The list of amplitudes.
        '''
        return self._amplitudes

    @property
    def get_coordinates(self):
        '''
        Get the coordinates of the grid points.

        Returns:
            list: The list of coordinates.
        '''
        return self._coordinates

    @property
    def get_frequencies(self):
        '''
        Get the frequencies of the grid points.

        Returns:
            list: The list of frequencies.
        '''
        return self._frequencies
    
    @property
    def get_phases(self):
        '''
        Get the phases of the grid points.

        Returns:
            list: The list of phases.
        '''
        return self._phases
    
    @property
    def get_nrLines(self):
        '''
        Get the number of lines in the grid.

        Returns:
            int: The number of lines.
        '''
        return self._nrLines

    @property
    def get_sequenceSweeps(self):
        '''
        Get the sequence of sweeps for each grid point.

        Returns:
            list: The list of sweep sequences.
        '''
        return self._sequenceSweeps
    
    @property
    def get_initialConfiguration(self):
        '''
        Get the initial configuration of the grid.

        Returns:
            dict: The initial configuration.
        '''
        return self._initialConfiguration
    
    @property
    def get_currentConfiguration(self):
        '''
        Get the current configuration of the grid.

        Returns:
            dict: The current configuration.
        '''
        return self.currentConfiguration

    @property
    def get_finalConfiguration(self):
        '''
        Get the final configuration of the grid.

        Returns:
            dict: The final configuration.
        '''
        return self.finalConfiguration


#####################################################################################################
#Sweep AQiPT class
#####################################################################################################

class Sweep:
    '''
    A class representing a sweep for configuring an instrument.

    Attributes:

        _initialConfiguration: The initial configuration before the sweep.
        _finalConfiguration: The final configuration after the sweep.
        _ampModulation: The amplitude modulation for the sweep.
        _freqModulation: The frequency modulation for the sweep.
        _phaseModulation: The phase modulation for the sweep.
        _calibration: The calibration data for the sweep.

    Methods:

        get_initialConfiguration ():
        get_finalConfiguration ():
        get_modulation ():
        get_calibration ():
        calibrate ():

    '''

    def __init__(self, initial_config=None, final_config=None, calibration=3.5e-6):
        '''
        Initialize a Sweep object with default attributes.
        '''
        self._initialConfiguration = initial_config
        self._finalConfiguration = final_config

        self._sweepTime = None
        self._sweepTbase = None

        self._ampModulation = None
        self._freqModulation = None
        self._phaseModulation = None

        self._calibration = calibration

    @property
    def get_initialConfiguration(self):
        '''
        Get the initial configuration before the sweep.

        Returns:
            Any: The initial configuration.
        '''
        return self._initialConfiguration

    @property
    def get_finalConfiguration(self):
        '''
        Get the final configuration after the sweep.

        Returns:
            Any: The final configuration.
        '''
        return self._finalConfiguration

    @property
    def get_modulation(self):
        '''
        Get the modulation parameters for the sweep.

        Returns:
            dict: The modulation parameters (amplitude, frequency, phase).
        '''
        return {'amplitude': self._ampModulation, 'frequency': self._freqModulation, 'phase': self._phaseModulation}

    @property
    def get_calibration(self):
        '''
        Get the calibration data for the sweep.

        Returns:
            Any: The calibration data.
        '''
        return self._calibration

    def calibrate(self):
        '''
        Perform calibration for the sweep.
        '''
        pass

class quadratic_sweep(Sweep):
    '''
        Quadratic sweep subclass

        Attributes:

            _coef (list): List of coefficients for the parabolic sweep
            amp_function (): Amplitude python function for the sweep in the Amplitude parameter
            freq_function (): Frequency python function for the sweep in the Frequency parameter
            phase_function (): Phase python function for the sweep in the Phase parameter
            _sweep (): Full sweep

        Methods:
            
            _set_sweep () :
    '''

    def __init__(self, coef, sweep_steps=1e3, *args, **kwargs):
        super().__init__(*args, **kwargs);

        self._coef = coef;

        self.amp_function = constant;
        self.freq_function = xquare;
        self.phase_function = constant;

        self._sweep = None
        self._sweepSteps = int(sweep_steps)

    def _set_sweep(self):

        self._sweepTime = (-self._coef['frequency'][1] + np.sqrt( self._coef['frequency'][1]**2 - 4*self._coef['frequency'][0]*self._coef['frequency'][2]))/(2*self._coef['frequency'][0])
        self._sweepTbase = np.linspace(0, self._sweepTime, self._sweepSteps)

        self._ampModulation = self.amp_function(self._sweepTbase, self._coef['amplitude'])
        self._freqModulation = self.freq_function(self._sweepTbase, self._coef['frequency'])
        self._phaseModulation = self.phase_function(self._sweepTbase, self._coef['phase'])

        # return _amplitude*np.sin(2*np.pi*_frequency + _phase)

    def showSweep(self, title=None):

        fig, ax = plt.subplots(3,1, figsize=(20,15), dpi=200, sharex=True)
        ax[0].plot(self._ampModulation, color='dodgerblue', alpha=0.8)
        ax[0].set_ylabel('Amplitude [V]', fontsize=14)

        ax[1].plot(self._freqModulation, color='red', alpha=0.8)
        ax[1].set_ylabel('Frequency [MHz]', fontsize=14)

        ax[2].plot(self._phaseModulation, color='orange', alpha=0.8)
        ax[2].set_ylabel('Phase [º]', fontsize=14)

        ax[2].set_xlabel(r'Time [$\mu$s]', fontsize=14)

        if title!=None:
            plt.title(title)

        plt.show()


#####################################################################################################
#NCO AQiPT class
#####################################################################################################
class NCO:
    '''
    A Numerically Controlled Oscillator (NCO) representing a tone generator.

    Attributes:

        number (int): The identifier number of the NCO.
        frequency (float): The frequency of the generated tone.
        phase (float): The phase of the generated tone.
        amplitude (float): The amplitude of the generated tone.
        sweep_type (str): The type of sweep for frequency modulation.
        sweep (tuple): The parameters for frequency sweep (start, end, step).
        busy (bool): Indicates whether the NCO is currently in use.

    Methods:

        set_configuration ():
        update_attribute ():
        release ():

    '''

    def __init__(self, number, frequency, phase, amplitude, sweep_type=None, sweep_params=None):
        '''
        Initialize an NCO object with specified attributes.

        Args:
            number (int): The identifier number of the NCO.
            frequency (float): The frequency of the generated tone.
            phase (float): The phase of the generated tone.
            amplitude (float): The amplitude of the generated tone.
            sweep_type (str, optional): The type of sweep for frequency modulation.
            sweep (tuple, optional): The parameters for frequency sweep (start, end, step).
        '''
        self.number = number
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.sweep_type = sweep_type
        self.sweep_params = sweep_params

        if isinstance(self.sweep_params, list) or isinstance(self.sweep_params, tuple):
            try:
                self.sweep = quadratic_sweep(self.sweep_params) 
            except:
                pass

        self.busy = False

    def set_configuration(self, frequency=None, phase=None, amplitude=None, sweep_type=None, sweep=None):
        '''
        Set the configuration attributes of the NCO.

        Args:
            frequency (float, optional): The frequency of the generated tone.
            phase (float, optional): The phase of the generated tone.
            amplitude (float, optional): The amplitude of the generated tone.
            sweep_type (str, optional): The type of sweep for frequency modulation.
            sweep (tuple, optional): The parameters for frequency sweep (start, end, step).
        '''
        if frequency is not None:
            self.frequency = frequency
        if phase is not None:
            self.phase = phase
        if amplitude is not None:
            self.amplitude = amplitude
        if sweep_type is not None:
            self.sweep_type = sweep_type
        if sweep is not None:
            self.sweep = sweep

    def update_attribute(self, attribute, value):
        '''
        Update a specific attribute of the NCO.

        Args:
            attribute (str): The name of the attribute to update.
            value: The new value of the attribute.
        '''
        if hasattr(self, attribute):
            setattr(self, attribute, value)
        else:
            raise ValueError(f"NCO has no attribute '{attribute}'")

    def release(self):
        '''Release the NCO, marking it as not in use.'''
        self.busy = False


class Device:
    '''
    A device containing multiple Numerically Controlled Oscillators (NCOs) for tone generation.

    Attributes:

        device_id (int) : The identifier number of the device.
        _num_ncos () : Number of NCOs in the device
        nco_pool (list) : The pool of NCOs belonging to the device.
        initial_frequency (): Initial value of the frequency
        initial_amplitude (): Inital value of the Amplitude
        initial_phase (): Initial value of the phase
        initial_configuration (dict) : The initial configuration of NCOs in the device.

    Methods:

        get_available_nco () : Retrieve the fist available NCO following index sorting
        get_available_ncos () : Retrieve a list of available NCOs
        release_nco () : Release NCO. i.e., change busy status from True to False
        flush () : Reset sweep schedule of NCOs
        execute () : Execute NCOs sweeps
        queue () : Add to scheduler NCO's sweeps

    '''

    def __init__(self, device_id, num_nco=64, initial_frequency=None, initial_phase=None, initial_amplitude=None):
        '''
        Initialize a Device object with specified attributes.

        Args:
            device_id (int): The identifier number of the device.
            num_nco (int, optional): The number of NCOs in the device. Default is 64.
            initial_frequency (float, optional): The initial frequency for NCOs. Default is 1000.
            initial_phase (float, optional): The initial phase for NCOs. Default is 0.
            initial_amplitude (float, optional): The initial amplitude for NCOs. Default is 1.
        '''
        self._device_id = device_id
        self._num_ncos = num_nco

        if initial_frequency==None:
            self.initial_frequency = INITIAL_FREQUENCY_NCOS*self._num_ncos
        else:
            self.initial_frequency = initial_frequency

        if initial_amplitude==None:
            self.initial_amplitude = INITIAL_AMPLITUDE_NCOS*self._num_ncos
        else:
            self.initial_amplitude = initial_amplitude

        if initial_phase==None:
            self.initial_phase = INITIAL_PHASE_NCOS*self._num_ncos
        else:
            self.initial_phase = initial_phase

        self.nco_pool = [NCO(i, self.initial_frequency[i], self.initial_phase[i], self.initial_amplitude[i]) for i in range(self._num_ncos)]
        self.initial_configuration = {'frequency': self.initial_frequency, 'phase': self.initial_phase, 'amplitude': self.initial_amplitude}

    def get_available_nco(self):
        '''
        Get an available NCO from the device's pool.

        Returns:
            NCO: An available NCO object, or None if all NCOs are busy.
        '''
        for i, nco in enumerate(self.nco_pool):
            if not nco.busy:
                return nco
        return None

    def get_available_ncos(self):
        '''
        Get an available NCO from the device's pool.

        Returns:
            NCO: An available NCO object, or None if all NCOs are busy.
        '''
        _dNCOs = {}
        for i, nco in enumerate(self.nco_pool):
            if not nco.busy:
                _dNCOs[str(i)] = False
        return _dNCOs

    def release_nco(self, nco_number):
        '''
        Release an NCO back to the pool based on its number.

        Args:
            nco_number (int): The number of the NCO to release.
        '''
        for nco in self.nco_pool:
            if nco.number == nco_number:
                nco.release()

    def flush(self):
        '''Reset all NCOs in the device to their initial configuration.'''
        for nco in self.nco_pool:
            nco.set_configuration(**self.initial_configuration)

    def execute(self):
        pass

    def queue(self):
        pass





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

        if director!=None:
            try:
                self.atom = director.atom_specs;
            except:
                pass
        elif producer!=None:
            try:
                self.atom = producer.atom_specs;
            except:
                pass
        else:
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
