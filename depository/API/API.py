#Atomic Quantum information Processing Tool (AQIPT) - Control module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): S.Whitlock. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                 T.Biename. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2023-02-07
# Last update: 2023-05-26

#libs
import ctypes, time, os, sys

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from pandas import DataFrame

from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.control import AQiPTcontrol as control
from AQiPT.modules.kernel import AQiPTkernel as kernel
from AQiPT.modules.daq import AQiPTdaq as daq

import dash
import dash_core_components as dcc
import dash_html_components as html

###################################################################################################
#######################                 AQiPT API                    ##############################
###################################################################################################


def constant(to:float=0, value:float=1, duration:float=0, name='Default constant', dyn_params:aqipt.general_params=aqipt.general_params({'sampling':int(1e3), 'bitdepth':16, 'time_dyn':0}), tpw:float=None, color:str=aqipt.color_lst[0], ftype='square', show_plot=False):

    ''' 
        EQM wrapper of AQiPT-function for creating a constant DC signal at to time,
        DC value and a given duration time .

        to (float):
        value (float):
        duration (float): 
        dyn_params (aqipt.general_params):
        tpw ():
        color (str):
        name (str):
        ftype (str):
        show_plot (bool):
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
        EQM wrapper of AQiPT-function for creating a constant zero DC signal at to time,
        DC=0 value and a given duration time.

        to (float):
        duration (float): 
        dyn_params (aqipt.general_params):
        tpw ():
        color (str):
        name (str):
        ftype (str):
        show_plot (bool):
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
        EQM wrapper of AQiPT-function + AQiPT-pulse for creating a square pulse signal
        at to time, with DC value and a given duration time. Also consider as TTL signal.

        ch_id (str):
        time_range (list):
        ttl_duration (float):
        risetime (float): 
        falltime (float):
        invertion (bool):
        dyn_params (aqipt.general_params):
        tpw ():
        color (str):
        name (str):
        show_plot (bool):
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
        EQM wrapper of AQiPT-function + AQiPT-pulse for creating a sequence of digital (square) signals
        at different to times (timeflags), with DC value and a given duration time. Also consider as TTL signal.

        ch_id (str):
        time_range (list):
        timeflags (list):
        ttl_durations (float): 
        risetimes (float):
        falltimes (float):
        invertions (bool):
        dyn_params (aqipt.general_params):
        tpw ():
        color (str):
        name (str):
        show_plot (bool):
        bitlimits (list):
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
        EQM wrapper of AQiPT-function for creating any arbitrary analog signal with or
        without any carrier frequency.

        ch_id (str)
        analog_args (dict):
        sampling (int):
        carrier_frequency (float):
        color (str):
        name (str):
        show_plot (bool):
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

    if carrier_frequency!=None:
        carrier_args = {'Amp':1, 'freq':carrier_frequency, 'phase':0, 'tp_window':analog_args['tp_window'], 'name':'carrier', color:'gray', 'type':'sine'};
        carrier_aqipt_function = control.function(tbase, carrier_args);
        carrier, carrier_plot = carrier_aqipt_function.sinusoidal(plotON=False)

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
        EQM wrapper of AQiPT-function + AQiPT-pulse for creating a sequence of analogs signals
        at different to times (timeflags), with carrier frequencies and bitdepths.

        ch_id (str):
        time_range (list):
        timeflags (list):
        analog_args (float): 
        samplings (float):
        carrier_frequencies (float):
        bitdepths (bool):
        dyn_params (aqipt.general_params):
        tpw ():
        color (str):
        name (str):
        show_plot (bool):
        bitlimits (list):
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
            carrier, carrier_plot = carrier_aqipt_function.sinusoidal(plotON=False);

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


        _AQiPT_waveform_i = compose(time_range=[0,_tpw], 
                                    signals=[aqipt_function], 
                                    timeflags=[0],  
                                    dyn_params=_dyn_params, 
                                    bitdepth=_bitdepth, 
                                    bitlimits=[min(aqipt_function.waveform), max(aqipt_function.waveform)], 
                                    name=name, 
                                    show_plot=show_plot);

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
      EQM wrapper of AQiPT-pulse for creating a pulse instance composed by functions to be stack in a AQiPT-sequence. 
      Can be used to compose digital() signals and or analog() signals from the API.

      time_range (list):
      signals (list): 
      timeflags (list):
      dyn_params (list):
      bitdepth (int):
      bitlimits (list):
      name (str):
      show_plot (bool):
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

def data(ch_id, data, timeflag:list, name='Default data', show_data=False):
    
    pass

def showDAQ(waveforms:list):
    daq.plotSequences(waveforms);
    
    print('running server...')

def generateSpecifications(mode='JSON', hwPATH=None, swPATH=None):

    ''' 
        EQM wrapper loading Hardware and Software specifications.
    
        mode (str):
        hwPATH (str):
        swPATH (str):
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

def generateDirector( HW, SW, calibrations=None):

    _director = control.director(HW_specs = HW, SW_specs = SW, calibration=calibrations);

    _director.load_drivers();

    return _director

def prepareAcquisition():


    
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