#Atomic Quantum information Processing Tool (AQIPT)

import time, os, sys

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import scipy.signal as signal

"""
Classes:

producer:
function:
pulse:
track:
instruction:
sequence:


waveform:

"""

class producer(object):
	"""docstring for producer

			status:
			connectionStatus:
			addInstruction:
			triggerMode:



	"""
	def __init__(self, arg):
		super(producer, self).__init__()
		self.arg = arg
	def turnON():
	def turn OFF():
	def _error():

	def addInstruction(self, arg):

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
        ramp()
	        Conjugate of quantum object.
        parabola()
	        Conjugate of quantum object.
        sinusoidal()
	        Conjugate of quantum object.
        gaussian()
	        Conjugate of quantum object.
        triangular()
	        Conjugate of quantum object.
        sawtooth()
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

        function = self.args['g_Amp'] * np.exp( -( (self.tbase - self.args['g_center']) / 2*self.args['g_std'])**2 );



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

class pulse:
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
    def __init__(self, times):
        
        #atributes
        self.tbase = times
        self._res = type
        self.waveform = np.zeros(tbase.shape)
    
    def addFunction(self, tstart, function, plotON=False):
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
        replace=lambda result,func,s:result[:s]+ func + result[s+len(func):]
        
        self.waveform = np.array(replace(result, func, int(tstart)))
        return self.waveform
    
    def getPulse(self):
        return self.waveform
        
class track(object):
	"""docstring for tracks"""
	def __init__(self, arg):
		super(tracks, self).__init__()
		self.arg = arg

class instruction(object):
	"""docstring for instruction"""
	def __init__(self, arg):
		super(instruction, self).__init__()
		self.arg = arg
		
class sequence(object):
	"""docstring for sequence"""
	def __init__(self, arg):
		super(sequence, self).__init__()
		self.arg = arg

		


##############
# class waveform(object):
# 	"""docstring for waveforms"""
# 	def __init__(self, arg):
# 		super(waveforms, self).__init__()
# 		self.arg = arg
