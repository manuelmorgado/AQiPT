#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - AQiPT interface module (API)

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                            Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): S.Whitlock. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                 T.Biename. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2023-02-07
# Last update: 2024-12-14

#libs
import ctypes, time, os, sys

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from pandas import DataFrame

from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.control import AQiPTcontrol as control
from AQiPT.modules.kernel import AQiPTkernel as kernel
from AQiPT.modules.analysis import AQiPTanalysis as analysis
from AQiPT.modules.daq import AQiPTdaq as daq

import dash
import dash_core_components as dcc
import dash_html_components as html

###################################################################################################
#######################                 Backend API                  ##############################
###################################################################################################


def constant(to:float=0, value:float=1, duration:float=0, name='Default constant', dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), tpw:float=None, color:str=aqipt.color_lst[0], ftype='square', show_plot=False):

    ''' 
        API wrapper of AQiPT-function for creating a constant DC signal at to time,
        DC value and a given duration time .
        
        INPUTS
        ------

        to (float): start time for the constant waveform
        value (float): amplitude value 
        duration (float): duration of the constant value
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time 
        tpw (float): time window of pulse
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        ftype (str): function type (from AQiPTcontrol module function)
        show_plot (bool): shows plot or not


        OUTPUTS
        -------
        
        aqipt_function, plot (tuple): AQiPT function object and generated plot
    
    '''

    #check if dynamic params are given
    if dyn_params==None:
        raise TypeError("Remember adding the dynamic parameters.")


    else:

        if tpw==None:
            tpw = duration;

        tbase = np.linspace(0, tpw, int(tpw*dyn_params.sampling));

        arguments = {'amp': value, 't_o': to+(duration/2), 'width': (duration/2), 
                    'tp_window':tpw, 
                    'name': name, 'color': color, 
                    'type': ftype};

        #function creation
        aqipt_function = control.function(tbase, arguments);
        function, plot = aqipt_function.step(plotON=show_plot)

        if show_plot==True:
            #plot
            plt.xlabel(r'Time [$\mu s$]')
            plt.ylabel(r'Amplitude [V]')
            plt.text(0, max(function), s='function: '+arguments['name'],
                    size=15, rotation=0.,ha="center", va="center", color='white',
                    bbox=dict(boxstyle="round",
                                ec=(.0, 0.4, 0.8),
                                fc=(.0, 0.4, 0.8),) )
            plt.show()

    return aqipt_function, plot

def wait(to:float=0, duration=0, name='Default wait', dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), tpw:float=None, color:str=aqipt.color_lst[0], ftype='square', show_plot=False):

    ''' 
        API wrapper of AQiPT-function for creating a constant zero DC signal at to time,
        DC=0 value and a given duration time.
        
        INPUTS
        ------

        to (float): start time for the constant waveform
        duration (float):  duration of the constant value
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time 
        tpw (float): time window of pulse
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        ftype (str): function type (from AQiPTcontrol module function)
        show_plot (bool): shows plot or not


        OUTPUTS
        -------

        aqipt_function, plot (tuple): AQiPT function object and generated plot
        
    '''

    #check if dynamic params are given
    if dyn_params==None:
        raise TypeError("Remember adding the dynamic parameters.")
        

    else:
       
        if tpw==None:
            tpw = duration;

        tbase = np.linspace(0, tpw, int(tpw*dyn_params.sampling));

        arguments = {'amp': 0, 't_o': to+(duration/2), 'width': (duration/2), 
                     'tp_window':tpw, 
                     'name': name, 'color': color, 
                     'type': ftype};

        #function creation
        aqipt_function = control.function(tbase, arguments);
        function, plot = aqipt_function.step(plotON=show_plot);

        if show_plot==True:
            #plot
            plt.xlabel(r'Time [$\mu s$]')
            plt.ylabel(r'Amplitude [V]')
            plt.text(0, max(function), s='function: '+arguments['name'],
                    size=15, rotation=0.,ha="center", va="center", color='white',
                    bbox=dict(boxstyle="round",
                                ec=(.0, 0.4, 0.8),
                                fc=(.0, 0.4, 0.8),) );
            plt.show()

    return aqipt_function, plot

def digital(ch_id:str, time_range:list, ttl_duration:float, name='Default digital', risetime:float=0.1, falltime:float=0.1, invertion:bool=False, dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), tpw:float=None, color:str=aqipt.color_lst[0], show_plot=False):

    ''' 
        API wrapper of AQiPT-function + AQiPT-pulse for creating a square pulse signal
        at to time, with DC value and a given duration time. Also consider as TTL signal.
        
        INPUTS
        ------

        ch_id (str): ID of physical channel in hardware
        time_range (list): full time of the digital waveform including rise and fall time
        ttl_duration (float): time duration of the digital square waveform
        risetime (float): time for the signal to raise (e.g., response time)
        falltime (float): time for the signal to fall (e.g., relaxation time)
        invertion (bool): definition of the HIGH and LOW states
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time 
        tpw (float): time window of pulse
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        show_plot (bool): shows plot or not

        OUTPUTS
        -------

        _AQiPT_waveform, _hw_python, arguments (tuple): AQiPT pulse object, hardware python instance and arguments of the pulse

    '''

    rising, rplot = wait(duration=risetime, dyn_params=dyn_params, name='ttl_rise', tpw=risetime);
    fall, fplot = wait(duration=falltime, dyn_params=dyn_params, name='ttl_fall', tpw=falltime);

    if invertion==False:
        ttl, tplot = constant(duration=ttl_duration, dyn_params=dyn_params, name='ttl', tpw=ttl_duration);
    elif invertion==True:
        ttl, tplot = constant(duration=ttl_duration, value=-1, dyn_params=dyn_params, name='ttl', tpw=ttl_duration);

    _AQiPT_waveform = compose(time_range=time_range, signals=[rising, ttl, fall], timeflags=[0, risetime, risetime+ttl_duration], dyn_params=dyn_params, name=name, show_plot=show_plot, plot_color=aqipt.color_lst[6]);

    
    #locating the python hardware object for _API_sequence
    if ch_id!='0x0':

        _dataframe = DataFrame(_hwspecs._IDs_bench); _identifier = ch_id;
        
        _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
        _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value


        _dataframe = DataFrame(_chspecs); _hardware_hex_id = _element_python.specs['EXT_CHANNEL'];

        _python_memory_alloc_ = aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'].index[0]]
        _hw_python = ctypes.cast( int(_python_memory_alloc_, 16), ctypes.py_object).value


    else:
        _hw_python = None;


    #locating the arguments for _API_sequence
    arguments = {'API_instruction_TYPE':'DIGITAL',
                 'identifier':ch_id,
                 'time_range' : time_range, 
                 'duration' : ttl_duration, 
                 'risetime' : risetime, 
                 'falltime' : falltime, 
                 'invertion' : invertion, 
                 'dyn_params': dyn_params, 
                 'time_pulse_window' : tpw, 
                 'name' : name};
    

    return _AQiPT_waveform, _hw_python, arguments

def digitals(ch_id:str, time_range:list, timeflags:list, ttl_durations:float, name='Default digitals', risetimes:float=0.0, falltimes:float=0.0, invertions:bool=False, dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), tpw:float=None, color:str=aqipt.color_lst[0], show_plot=False, bitlimits=[0,1]):

    ''' 
        API wrapper of AQiPT-function + AQiPT-pulse for creating a sequence of digital (square) signals
        at different to times (timeflags), with DC value and a given duration time. Also consider as TTL signal.
        
        INPUTS
        ------

        ch_id (str): ID of physical channel in hardware
        time_range (list): full time of the digital waveform including rise and fall time
        timeflags (list): list of time values when each digital waveform starts
        ttl_durations (float): time durations of each digital square waveforms
        risetimes (float): time for the signal to raise (e.g., response time)
        falltimes (float): time for the signal to fall (e.g., relaxation time)
        invertions (bool): definition of the HIGH and LOW states
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time
        tpw (float): time window of pulse
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        show_plot (bool): shows plot or not
        bitlimits (list): values for limiting the vertical resolution

        OUTPUTS
        -------
        
        _AQiPT_waveform, _hw_python, arguments (tuple): AQiPT pulse object, hardware python instance and arguments of the pulse

    '''

    if isinstance(invertions, bool):
        invertions = [False]*len(ttl_durations);

    if isinstance(risetimes, float):
        risetimes = [0]*len(ttl_durations);

    if isinstance(falltimes, float):
        falltimes = [0]*len(ttl_durations);

    _signals = [];
    _timeflags = [];
    _tbase_ttl = [];

    i=0;
    for _timeflag, _ttl_duration, _risetime, _falltime, _invertion in zip(timeflags, ttl_durations, risetimes, falltimes, invertions):

        _tbase_ttl_ = _timeflag+_risetime + _ttl_duration + _falltime;
        _tbase_ttl.append(_tbase_ttl_);

        if _timeflag < _tbase_ttl[i-1] and i>0:
            print('Warning: conflict between Signal {ttl1} and Signal {ttl2}'.format(ttl1=i, ttl2=i-1))

        rising, rplot = wait(duration=_risetime, dyn_params=dyn_params, name='ttl_rise', tpw=_risetime);
        fall, fplot = wait(duration=_falltime, dyn_params=dyn_params, name='ttl_fall', tpw=_falltime);

        if _invertion==False:
            ttl, tplot = constant(duration=_ttl_duration, dyn_params=dyn_params, name='ttl', tpw=_ttl_duration);
        elif _invertion==True:
            bitlimits=[-1,1]
            ttl, tplot = constant(duration=_ttl_duration, value=-1, dyn_params=dyn_params, name='ttl', tpw=_ttl_duration);

        _signals+=[rising,ttl,fall];
        _timeflags+=[0+_timeflag, _risetime+_timeflag, (_risetime+_ttl_duration)+_timeflag];

        i+=1;

    _AQiPT_waveform = compose(time_range=time_range, signals=_signals, timeflags=_timeflags, dyn_params=dyn_params, name=name, show_plot=show_plot, bitlimits=bitlimits, plot_color=aqipt.color_lst[8]);

    #locating the python hardware object for _API_sequence
    if ch_id!='0x0':

        _dataframe = DataFrame(_hwspecs._IDs_bench); _identifier = ch_id;
        
        _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
        _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value


        _dataframe = DataFrame(_chspecs); _hardware_hex_id = _element_python.specs['EXT_CHANNEL'];

        _python_memory_alloc_ = aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'].index[0]]
        _hw_python = ctypes.cast( int(_python_memory_alloc_, 16), ctypes.py_object).value


    else:
        _hw_python = None;


    #locating the arguments for _API_sequence
    arguments = {'API_instruction_TYPE':'DIGITALS',
                 'identifier':ch_id,
                 'time_range' : time_range,
                 'timeflags' : timeflags, 
                 'durations' : ttl_durations, 
                 'risetimes' : risetimes, 
                 'falltimes' : falltimes, 
                 'invertions' : invertions, 
                 'dyn_params': dyn_params, 
                 'time_pulse_window' : tpw,
                 'bitlimits' : bitlimits, 
                 'name' : name};
    

    return _AQiPT_waveform, _hw_python, arguments

def analog(ch_id:str, analog_args:dict, sampling:int, carrier_frequency:float=None, bitdepth:int=16, name:str='Default analog', color:str=aqipt.color_lst[0], show_plot=False):

    ''' 
        API wrapper of AQiPT-function for creating any arbitrary analog signal with or
        without any carrier frequency.
        

        INPUTS
        ------

        ch_id (str): ID of physical channel in hardware
        analog_args (dict): argument to be use in the generation of an analog waveforms
        sampling (int): sampling rate for the analog waveforms
        carrier_frequency (float): frequency of the carrier, later mixed with envelop analog signal
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        show_plot (bool): shows plot or not


        OUTPUTS
        -------

        _AQiPT_waveform, _hw_python, arguments (tuple): AQiPT pulse object, hardware python instance and arguments of the pulse


    '''
    dyn_params = aqipt.general_params({'sampling':int(sampling),
                                       'bitdepth':int(bitdepth),
                                       'time_dyn':analog_args['tp_window']});

    tbase = np.linspace(0, analog_args['tp_window'], int(analog_args['tp_window']*sampling));

    #function creation
    aqipt_function = control.function(tbase, analog_args);

    if analog_args['type']=='square':
        
        analog_args['width']= analog_args['width']*0.5; #fixing width/2 
        analog_args['t_o']= analog_args['t_o']+analog_args['width']; #fixing offset due to width/2

        func, plot = aqipt_function.step(plotON=False)

    elif analog_args['type']=='sine':
        func, plot = aqipt_function.sinusoidal(plotON=False)

    elif analog_args['type']=='ramp':
        func, plot = aqipt_function.ramp(plotON=False)

    elif analog_args['type']=='chirp':
        func, plot = aqipt_function.chirp(plotON=False)

    elif analog_args['type']=='chirplet':
        func, plot = aqipt_function.chirplet(plotON=False)

    elif analog_args['type']=='gaussian_chirplet':
        func, plot = aqipt_function.gaussian_chirplet(plotON=False)

    elif analog_args['type']=='gaussian':
        func, plot = aqipt_function.gaussian(plotON=False)
    
    elif analog_args['type']=='supergaussian':
        func, plot = aqipt_function.supergaussian(plotON=False)
    
    elif analog_args['type']=='triangular':
        func, plot = aqipt_function.triangular(plotON=False)

    elif analog_args['type']=='cspline':
        func, plot = aqipt_function.cspline(plotON=False)

    if carrier_frequency!=None:
        carrier_args = {'Amp':1, 'freq':carrier_frequency, 'phase':0, 'tp_window':analog_args['tp_window'], 'name':'carrier', color:'gray', 'type':'sine'};
        carrier_aqipt_function = control.function(tbase, carrier_args);
        
        if show_plot:
            carrier, carrier_plot = carrier_aqipt_function.sinusoidal(plotON=show_plot)
        else:
            carrier = carrier_aqipt_function.sinusoidal(plotON=show_plot)
        
        aqipt_function.waveform = aqipt_function.waveform * carrier_aqipt_function.waveform;

    if show_plot==True:
        #plot
        plt.xlabel(r'Time [$\mu s$]')
        plt.ylabel(r'Amplitude [V]')
        plt.text(0, max(func), s='function: '+analog_args['name'],
                size=15, rotation=0.,ha="center", va="center", color='white',
                bbox=dict(boxstyle="round",
                            ec=(.0, 0.4, 0.8),
                            fc=(.0, 0.4, 0.8),) )
        plt.show()

    _AQiPT_waveform = compose(time_range=[0,analog_args['tp_window']], signals=[aqipt_function], timeflags=[0],  dyn_params=dyn_params, bitdepth=dyn_params.bitdepth, bitlimits=[min(aqipt_function.waveform), max(aqipt_function.waveform)], name=name, show_plot=show_plot);

    #locating the python hardware object for _API_sequence
    if ch_id!='0x0':

        _dataframe = DataFrame(_hwspecs._IDs_bench); _identifier = ch_id;
        
        _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
        _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value


        _dataframe = DataFrame(_chspecs); _hardware_hex_id = _element_python.specs['EXT_CHANNEL'];

        _python_memory_alloc_ = aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'].index[0]]
        _hw_python = ctypes.cast( int(_python_memory_alloc_, 16), ctypes.py_object).value


    else:
        _hw_python = None;


    #locating the arguments for _API_sequence
    arguments = {'API_instruction_TYPE':'ANALOG',
                 'identifier':ch_id,
                 'analog_args' : analog_args,
                 'sampling' : sampling, 
                 'carrier_frequency' : carrier_frequency, 
                 'bitdepth' : bitdepth, 
                 'name' : name};
    

    return _AQiPT_waveform, _hw_python, arguments

def analogs(ch_id:str, time_range:list, timeflags:list, analog_args:list, samplings:list, carrier_frequencies:list, bitdepths:list, name:str='Default analogs', dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), analogs_duration:float=None, color:str=aqipt.color_lst[0], show_plot=False, bitlimits:list=[[-1,1]]):

    ''' 
        API wrapper of AQiPT-function + AQiPT-pulse for creating a sequence of analogs signals
        at different to times (timeflags), with carrier frequencies and bitdepths.
        

        INPUTS
        ------

        ch_id (str): ID of physical channel in hardware
        time_range (list): full time of the analogs waveform including rise and fall time
        timeflags (list): list of time values when each analogs waveform starts
        analog_args (float): list of arguments for the generation of each analog waveform
        samplings (int): list of sampling rates for the analog waveforms
        carrier_frequencies (float): list of carrier frequencies for each waveform
        bitdepths (list): list of bitdepths for each analog waveform
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time
        tpw (float): time window of pulse
        color (str): color of the ploted waveform
        name (str): assigned name od the AQiPT object
        show_plot (bool): shows plot or not
        bitlimits (list): values for limiting the vertical resolution


        OUTPUTS
        -------

        _AQiPT_waveform, _hw_python, arguments (tuple): AQiPT pulse object, hardware python instance and arguments of the pulse


    '''

    if analogs_duration==None:
        analogs_duration = [_args['tp_window'] for _args in analog_args];

    _signals = [];

    i=0;
    for _timeflag, _analog_args, _sampling, _carrier_frequency, _bitdepth, _tpw, _bitlimits in zip(timeflags, analog_args, samplings, carrier_frequencies, bitdepths, analogs_duration, bitlimits):

        if _timeflag < timeflags[i-1]+_tpw and i>0:
            print('Warning: conflict between Signal {signal1} and Signal {signal2}. \n Check time-flags and signal durations.'.format(signal1=i, signal2=i-1))

        _dyn_params = aqipt.general_params({'sampling':int(_sampling),
                                            'bitdepth':int(_bitdepth),
                                            'time_dyn':_tpw});

        tbase = np.linspace(0, _tpw, int(_tpw*_sampling));

        #function creation
        aqipt_function = control.function(tbase, _analog_args);
        print(_analog_args)
        if _analog_args['type']=='square':
            func, plot = aqipt_function.step(plotON=False)

        elif _analog_args['type']=='sine':
            func, plot = aqipt_function.sinusoidal(plotON=False)

        elif _analog_args['type']=='ramp':
            func, plot = aqipt_function.ramp(plotON=False)

        elif _analog_args['type']=='chirp':
            func, plot = aqipt_function.chirp(plotON=False)

        elif _analog_args['type']=='chirplet':
            func, plot = aqipt_function.chirplet(plotON=False)

        elif _analog_args['type']=='gaussian':
            func, plot = aqipt_function.gaussian(plotON=False)

        if _carrier_frequency!=None:
            carrier_args = {'Amp':1, 'freq':_carrier_frequency, 'phase':0, 'tp_window':_tpw, 'name':'carrier', 'color':'gray', 'type':'sine'};
            carrier_aqipt_function = control.function(tbase, carrier_args);
            
            if show_plot:
                carrier, carrier_plot = carrier_aqipt_function.sinusoidal(plotON=show_plot);
            else:
                carrier = carrier_aqipt_function.sinusoidal(plotON=show_plot);
            
            aqipt_function.waveform = np.real(aqipt_function.waveform) * np.real(carrier_aqipt_function.waveform);

        if show_plot==True:
            #plot
            plt.xlabel(r'Time [$\mu s$]')
            plt.ylabel(r'Amplitude [V]')
            plt.text(0, max(func), s='function: '+_analog_args['name'],
                    size=15, rotation=0.,ha="center", va="center", color='white',
                    bbox=dict(boxstyle="round",
                                ec=(.0, 0.4, 0.8),
                                fc=(.0, 0.4, 0.8),) )
            plt.show()


        # _AQiPT_waveform_i = compose(time_range=[0,_tpw], 
        #                             signals=[aqipt_function], 
        #                             timeflags=[0],  
        #                             dyn_params=_dyn_params, 
        #                             bitdepth=_bitdepth, 
        #                             bitlimits=[min(aqipt_function.waveform), max(aqipt_function.waveform)], 
        #                             name=name, 
        #                             show_plot=show_plot);

        _signals.append(aqipt_function);

        i+=1;


    _minbitlimit = min([min(signal.waveform) for signal in _signals]);
    _maxbitlimit = max([max(signal.waveform) for signal in _signals]);
    _maxbitdepth = max(bitdepths);

    _AQiPT_waveform = compose(time_range=time_range, 
                              signals=_signals, 
                              timeflags=timeflags, 
                              dyn_params=_dyn_params, 
                              name=name, 
                              show_plot=show_plot, 
                              plot_color=aqipt.color_lst[8],
                              bitdepth=_maxbitdepth,
                              bitlimits=[_minbitlimit, _maxbitlimit]); #TODO: add concatenate pulses in AQiPT.control.pulses and deprecate this line.

    #locating the python hardware object for _API_sequence
    if ch_id!='0x0':

        _dataframe = DataFrame(_hwspecs._IDs_bench); _identifier = ch_id;
        
        _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
        _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value;


        _dataframe = DataFrame(_chspecs); _hardware_hex_id = _element_python.specs['EXT_CHANNEL'];

        _python_memory_alloc_ = aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'].index[0]]
        _hw_python = ctypes.cast( int(_python_memory_alloc_, 16), ctypes.py_object).value;


    else:
        _hw_python = None;


    #locating the arguments for _API_sequence
    arguments = {'API_instruction_TYPE':'ANALOGS',
                 'identifier':ch_id,
                 'time_range' : time_range,
                 'timeflags' : timeflags, 
                 'analog_args' : analog_args, 
                 'samplings' : samplings, 
                 'carrier_frequencies' : carrier_frequencies, 
                 'bitdepths' : bitdepths, 
                 'dyn_params': dyn_params, 
                 'time_pulse_window' : analogs_duration,
                 'bitlimits' : bitlimits, 
                 'name' : name};
    

    return _AQiPT_waveform, _hw_python, arguments

def compose(time_range:list, signals:list, timeflags:list, name:str='Default composed', dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':1}), bitdepth:float=8, bitlimits:float=[0,1], show_plot=False, plot_color=None):
    ''' 
        API wrapper of AQiPT-pulse for creating a pulse instance composed by functions to be stack in a AQiPT-sequence. 
        Can be used to compose digital() signals and or analog() signals from the API.
        

        INPUTS
        ------

        time_range (list): full time of the digital waveform including rise and fall time
        signals (list): list of waveforms that should be composed
        timeflags (list): list of time values when each digital waveform starts
        dyn_params (aqipt.general_params): AQiPT general parameters object with sampling, bithdepth and dynamic time
        bitdepth (int): list of bitdepths for each analog waveform
        bitlimits (list): values for limiting the vertical resolution
        name (str): assigned name od the AQiPT object
        show_plot (bool): shows plot or not




        OUTPUTS
        -------

        pulse (aqipt.pulse): pulse AQiPT object

    '''

    pulse_tbase = np.linspace(time_range[0], time_range[1], ceil(abs(time_range[0]-time_range[1]))*dyn_params.sampling);
    pulse = control.pulse(pulse_label=name, 
                         times=pulse_tbase, 
                         function_list=signals, 
                         timeFlags=timeflags);

    pulse.compilePulse();

    pulse.digitizeWaveform(bitdepth, bitlimits[0],bitlimits[1]);

    if show_plot==True:
      pulse.plotPulse(xlabel=r'Time [$\mu s$]' , ylabel=r'Amplitude [V]', _color=plot_color);

    return pulse

def data(ch_id, data:analysis.ImageData, timeflag:list, name='Default data', show_data=False):

    '''
        API wrapper for passing data of data i.e., images to devices such as SLMs and DMDs
    '''

    

        #locating the python hardware object for _API_sequence
    if ch_id!='0x0':

        _dataframe = DataFrame(_hwspecs._IDs_bench); _identifier = ch_id;
        
        _element_hex_id = aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _identifier)['python_memory_alloc'].index[0]];
        _element_python = ctypes.cast( int(_element_hex_id, 16), ctypes.py_object).value


        _dataframe = DataFrame(_chspecs); _hardware_hex_id = _element_python.specs['EXT_CHANNEL'];

        _python_memory_alloc_ = aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'][aqipt.DFsearch(_dataframe, _hardware_hex_id)['python_memory_alloc'].index[0]]
        _hw_python = ctypes.cast( int(_python_memory_alloc_, 16), ctypes.py_object).value


    else:
        _hw_python = None;




def showDAQ(waveforms:list):

    '''
        API wrapper for visualizing sequences and other acquisition data
    '''

    daq.plotSequences(waveforms);
    
    print('running server...')

def generateSpecifications(mode='JSON', hwPATH=None, swPATH=None):

    ''' 
        API wrapper loading Hardware and Software specifications.
        
        INPUTS
        ------

        mode (str): mode if use json files
        hwPATH (str): path directory of hardware specification file
        swPATH (str): path directory of software specification file

        OUTPUTS
        -------

        (dictionary): hardware, software and channel specifications

    '''
    global _hwspecs
    global _swspecs
    global _chspecs

    if mode=='JSON':
        _IDsBench = kernel.IDsBench();

        print('Loading hardware specification')

        _hwspecs = kernel.hardwareSpecs();
        _hwspecs.loadSpecs(path=hwPATH,printON=False);
        _hwspecs.printHeader();
        _hwspecs.initSetup();

        _IDsBench.set_HW_bench(_hwspecs);
        _hwspecs._IDs_bench = _IDsBench.get_HW_bench();

        print('Loading software specification')

        _swspecs = kernel.softwareSpecs();
        _swspecs.loadSpecs(path=swPATH,printON=False);
        _swspecs.printHeader();

        _IDsBench.set_SW_bench(_swspecs);
        _swspecs._IDs_bench = _IDsBench.get_SW_bench();

        return {'Hardware': _hwspecs, 'Software': _swspecs}

    elif mode=='waveforms':

        _IDsBench = kernel.IDsBench();


        print('Loading hardware specification')

        _hwspecs = kernel.hardwareSpecs();
        _hwspecs.loadSpecs(path=hwPATH,printON=False);
        _hwspecs.printHeader();
        _hwspecs.initSetup();

        _IDsBench.set_HW_bench(_hwspecs);
        _hwspecs._IDs_bench = _IDsBench.get_HW_bench();

        print('Initializing software specification. ')

        _swspecs = kernel.softwareSpecs();
        _swspecs.loadSpecs(path=swPATH,printON=False);

        _chspecs = _IDsBench.get_CH_bench();

        # global _ID_BENCH
        # _ID_BENCH =_hwspecs._IDs_bench;


        return {'Hardware': _hwspecs, 'Software': _swspecs, 'HWChannels': _chspecs}

def generateDirector(atom, HW, SW, calibrations=None):
    
    '''
        API wrapper for generating the director for handling the backend resources
        

        INPUTS
        ------

        atom (): atom object for extracting properties
        HW (): hardware specification python abstraction
        SW (): software specification python abstraction
        calibration ():
        
        OUTPUTS
        -------

        _director (aqipt.director): director object from control module

    '''

    _director = control.director(atom_specs=atom, HW_specs = HW, SW_specs = SW, calibration=calibrations);

    _director.load_drivers();

    return _director

def generateProducer(atom, HW, SW, initial_state=None, time_Hamiltonian=True, time_simulation=None):

    '''
        API wrapper for generating the director for handling the backend resources
        

        INPUTS
        ------
        
        atom (): atom object for extracting properties
        HW (): hardware specification python abstraction
        SW (): software specification python abstraction
        initial_state (): initial state of the simulation
        time_Hamiltonian (bool): boolean for time simulation
        time_simulation (array): array of times
        
        OUTPUTS
        -------

        _producer (aqipt.producer): producer object from emulator module 

    '''

    _producer = emulator.producer(atom_specs=atom, HW_specs = HW, SW_specs = SW, simulation_time=time_simulation);

    if initial_state==None:
        print('Invalid initial state.')
    else:
        _producer.compile(psi0=initial_state, t_Hamiltonian=time_Hamiltonian);

    return _producer

def runSimulation(producer):

    '''
        API wrapper for runing simulation within producer
        

        INPUTS
        ------
        
        producer (aqipt.producer): producer with loaded Hamiltonian and model
        
        OUTPUTS
        -------

        producer (aqipt.producer): producer object from emulator module 

    '''
    producer.runSimulation();
    return producer

def machine(director, producer=None, database=None, mode='sequence', metadata='Default metadata', compile=True, execute=True, verbose=True, nshots=1):
    
    _machine = control.experiment(director=director, 
                                  producer=producer, 
                                  database=database,
                                  operationMode=mode,
                                  metadata=metadata)

    if nshots==1:
        _machine.compile()

        if verbose:
            print('Compilation completed.')

        _machine.execute()

        if verbose:
            print('Execution completed.')

            return _machine

    elif nshots>=1:
        _machine.compile()

        if verbose:
            print('Compilation completed.')

        _machine.execute()

        if verbose:
            print('Execution completed.')

            return _machine
            
def prepareAcquisition():

    '''
        API wrapper for showing live acquisition
    '''
    
    ixon897.Configure(args={'FanMode': 2, #0: full, 1: low, 2: off
                            'AcquisitionMode': 3, #1:single scan, #2:accumulate, 3: kinetics, 4: fast kinetics, 5: run till abort
                            'TriggerMode': 0, #0: internal, 1: external, 6: external start, 10: software trigger
                            'ReadMode': 4, #0: full vertical binning, 1:multi-track, 2: random track, 3: sinlge track, 4: image
                            'ExposureTime': 0.01784,
                            'NumberAccumulations': 1,
                            'NumberKinetics': 1,
                            'KineticCycleTime': 0.02460,
                            'VSSpeed': 4,
                            'VSAmplitude': 0,
                            'HSSpeed': [0,0],
                            'PreAmpGain': 2,
                            'ImageParams': {'hbin':1, 
                                            'vbin':1, 
                                            'hstart':1, 
                                            'hend':512, 
                                            'vstart':1,
                                            'vend':512}});


###################################################################################################
#######################               Middleware API                 ##############################
###################################################################################################



###################################################################################################
#######################                 Frontend API                 ##############################
###################################################################################################