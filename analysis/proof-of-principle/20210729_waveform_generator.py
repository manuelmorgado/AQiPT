# Author: Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2021-07-28
'''
	Short description of this pilot script for AQiPT: 

	Generate multiple waveforms within a pulse sequence, so we do not made use of pulse-object 
	of AQiPT, however that can be done. It also store the waveforms as csv files.


'''

#libs
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import csv
import AQiPTbeta as aqipt


color_lst = ['purple', 'green', 'orange', 'blue', 'red',
             'black', 'turquoise', 'magenta', 'yellow', 'maroon', 
             'lime', 'sienna', 'limegreen', 'violet', 'dodgerblue']; #list of colors


nr_physical_ops = 3; #number of waveforms to be construct

#PULSE SPECIFICATIONS

args = {'sampling':int(1e3), 'bitdepth':16, 'time_dyn':10}; #dictionary of general parameters introduced by user
params = aqipt.general_params(args); #get params AQiPT object from dictionary
tbase = params.timebase(); #obtain time-base from params object

#FUNCTION FOR PULSES IN 1 SEQUENCE

#function parameter arguments
args_lst=[{'amp': 1, 't_o':0.5, 'width': 0.25, 'tp_window': 1, 'name': 'func_1', 'color': color_lst[0]},
#          {'amp': 1, 't_o':2, 'width': 2,'tp_window': 5, 'name': 'func_2', 'color': color_lst[1]},
#          {'amp': 1, 't_o':4, 'width': 2,'tp_window': 5, 'name': 'func_3', 'color': color_lst[2]},
#          {'amp': 1, 't_o':2, 'width': 0.5,'tp_window': 5, 'name': 'func_4', 'color': color_lst[3]},
#          {'amp': 1, 't_o':0.5, 'width': 0.25,'tp_window': 5, 'name': 'func_5', 'color': color_lst[4]},
#          {'amp': 1, 't_o':2, 'width': 3.1,'tp_window': 5, 'name': 'func_6', 'color': color_lst[5]},
#          {'amp': 1, 't_o':5, 'width': 1,'tp_window': 5, 'name': 'func_7', 'color': color_lst[6]},
#          {'amp': 1, 't_o':4.75, 'width': 3.32,'tp_window': params.dyn_time, 'name': 'func_8', 'color': color_lst[7]},
         {'Amp':1, 'freq':20/(2*np.pi), 'phase':0,'tp_window': params.dyn_time, 'name': 'carrier', 'color': color_lst[8]},
         {'g_Amp':1, 'g_center': 0.8, 'g_std':0.2,'tp_window': 1.5, 'name': 'gaussian', 'color': color_lst[9]}];



#generation and ploting functions
fig, axs = plt.subplots(nr_physical_ops+1, 1, figsize=(18,10), sharex=True);
fig.subplots_adjust(hspace=0);

funcs_lst=[];
for _ in range(nr_physical_ops):
    #function creation
    tp_window = args_lst[_]['tp_window']; 

    if args_lst[_]['name'] == 'carrier':
        tp = np.linspace(0, tp_window, params.sampling);
        func, plot = aqipt.function(tp, args_lst[_]).sinusoidal();
        funcs_lst.append(func);

    elif args_lst[_]['name'] == 'gaussian':
        tp = np.linspace(0, tp_window, int((tp_window-0)*params.sampling/params.dyn_time));
        func, plot = aqipt.function(tp, args_lst[_]).gaussian();
        funcs_lst.append(func);

    else:
        tp = np.linspace(0, tp_window, int((tp_window-0)*params.sampling/params.dyn_time)); #time domain function
        func, plot = aqipt.function(tp, args_lst[_]).step(plotON=False);
        funcs_lst.append(func);


    #ploting
    axs[_].step(tp, func, color=args_lst[_]['color'], where='mid');
    axs[_].fill_between(tp, func, color=args_lst[_]['color'], step="mid", alpha=0.2);
    axs[_].set_ylabel(args_lst[_]['name']);
    axs[_].minorticks_on();
    axs[_].grid(b=True, which='both', color='gray', linestyle='--', alpha=0.25);
    plt.xlabel(r'Time $[\mu s]$', fontsize=18);

    
#storing waveforms
metadata = ["waveformName,"+str(), #waveformName
            "waveformPoints,1000", #waveform nr of points
            "waveformType,WAVE_ANALOG_16"] #waveform type
aqipt.saveWaveform(args,
                   args_lst, funcs_lst,
                   fileformat='.csv')