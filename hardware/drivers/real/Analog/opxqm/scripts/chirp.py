
#Atomic Quantum information Processing Tool (AQIPT) - Example OPX chirp

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2023-01-08

import os, time, sys

original_path = os.getcwd();




##############################################################################################################################################################################
aqipt_newPC = '/home/mmorgado/Desktop/AQiPT_vNewPC_20230617/';
os.chdir(aqipt_newPC);
sys.path.append(aqipt_newPC);
print('Changing directory to: ', os.getcwd())
from AQiPT import AQiPTcore as aqipt
from AQiPT.depository.APIs.API import * 

general_params = aqipt.general_params({'sampling':int(1e2), 'bitdepth':16, 'time_dyn':1} );


a1_omega_params = {'amp':1, 't_o':0,  'width': 8, 'tp_window':8, 'name': 'Omega a1',  'color': aqipt.color_lst[3],  'type': "square"};
a1_detuning_params = {'amp':-1.5, 't_o':3.9,  'width': 0.2, 'offset': 0.4, 'tp_window':8, 'name': 'Delta a1',  'color': aqipt.color_lst[3],  'type': "square"};

a2_omega_params = {'amp':1, 't_o':0,  'width': 8, 'tp_window':8, 'name': 'Omega a2',  'color': aqipt.color_lst[3],  'type': "square"};
a2_detuning_params = {'tri_amp':1, 'tri_freq':1/8,  'tri_phase': np.pi, 'tp_window':8, 'name': 'Delta a2',  'color': aqipt.color_lst[3],  'type': "triangular"};

a3_omega_params = {'amp':1, 't_o':0,  'width': 8, 'tp_window':8, 'name': 'Omega a3',  'color': aqipt.color_lst[3],  'type': "square"};
a3_detuning_params = {'g_Amp':-1, 'g_center':4,  'g_std': 1.5, 'tp_window':8, 'name': 'Delta a3',  'color': aqipt.color_lst[3],  'type': "gaussian"};


a4_omega_params = {'amp':1, 'xoffset':4, 'yoffset':0, 'n_param': 15, 'std':3, 'tp_window':8, 'name': 'Omega a4',  'color': aqipt.color_lst[3],  'type': "supergaussian"};
a4_detuning_params = {'g_Amp':-1, 'g_center':4,  'g_std': 0.5, 'tp_window':8, 'name': 'Delta a4',  'color': aqipt.color_lst[3],  'type': "gaussian"};

a5_omega_params = {'amp':1, 't_o':0,  'width': 8, 'tp_window':8, 'name': 'Omega a5',  'color': aqipt.color_lst[3],  'type': "square"};
a5_detuning_params = {'x':[0, 1, 2, 3, 4, 5, 6, 7, 8], 'y':[0.5, 0.55, 0.0, -0.5, -0.48, -0.5, 0.0, 0.55, 0.5], 'tp_window':8, 'name': 'Delta a5',  'color': aqipt.color_lst[3],  'type': "cspline"};


a1_omega = analog('0x0', a1_omega_params, general_params.sampling, None, 16, r'$\Omega$ A.1')
a1_detuning = analog('0x0', a1_detuning_params, general_params.sampling, None, 16, r'$\Delta$ A.1')

a2_omega = analog('0x0', a2_omega_params, general_params.sampling, None, 16,  r'$\Omega$ A.2')
a2_detuning = analog('0x0', a2_detuning_params, general_params.sampling, None, 16, r'$\Delta$ A.2')

a3_omega = analog('0x0', a3_omega_params, general_params.sampling, None, 16,  r'$\Omega$ A.3')
a3_detuning = analog('0x0', a3_detuning_params, general_params.sampling, None, 16, r'$\Delta$ A.3')

a4_omega = analog('0x0', a4_omega_params, general_params.sampling, None, 16,  r'$\Omega$ A.4')
a4_detuning = analog('0x0', a4_detuning_params, general_params.sampling, None, 16, r'$\Delta$ A.4')

a5_omega = analog('0x0', a5_omega_params, general_params.sampling, None, 16,  r'$\Omega$ A.5')
a5_detuning = analog('0x0', a5_detuning_params, general_params.sampling, None, 16,  r'$\Delta$ A.5')

##############################################################################################################################################################################
os.chdir(original_path)

import numpy as np
import opxqm as op
import matplotlib.pyplot as plt

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate.credentials import create_credentials
from qm import SimulationConfig


ip_address = '130.79.148.122';
A1 = 0.5 #envelope amplitude in Volt
T = 50000 #envelope period in ns and pulse duration
phi = np.pi*0.1322 #carrier phase
f = 1e6 #carrier frequency in Hz 
amp_scaling = 1.0 #scaling amplitude factor of the carrier - can be changed in real time
Constant_tone_Amp = 0.4


##############################################################################################################################################################################

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0}, # DC offset of the waveform
            },
        }
    },
    'elements': {
        'AOM1': {
            'singleInput': {
                'port': ('con1', 1),
            },
            'intermediate_frequency': f,
            'operations': {
                'Sin_Envelope': 'Sin_Envelope_pulse', #AOM1 has a single pulse: 'Sin_Envelope'
                'Constant': 'Constant',
            },
        },
    },
    'pulses': {
        'Sin_Envelope_pulse': {
            'operation': 'control',
            'length': T, #duration of the pulse in ns
            'waveforms': {
                'single': 'sin_wf', #sinusoid pulse uses a sin waveform
            }
        },
        "Constant": {
            'operation': 'control',
            'length': T, #ns
            'waveforms': {
                'single': 'const_wf',
            }
        },
    },
    'waveforms': {
        'sin_wf': { #definition of the sinusoid envelope
            'type': 'arbitrary',
            'samples': A1*np.sin([2*np.pi*t/T for t in range(T)]), #setting up the envelope function, you can play with other functions
        },
        'const_wf': { #definition of constant envelope
            'type': 'constant',
            'sample': Constant_tone_Amp,
        },
    },

}

##############################################################################################################################################################################
#simple example
with program() as program1:
    frame_rotation(phi, 'AOM1') #setting up the phase of the carrier to phi
    play('Sin_Envelope', 'AOM1') #playing the desired pulse: A1*sin(2*pi*t/T)*cos(2*pi*f*t + phi) for a duration T (the duration of the pulse)


#complex example changing in real time:
with program() as program2:
    phi_QUA = declare(fixed) #phi
    carrier_frequency = declare(int) #the carrier frequency
    i = declare(int) #index in a real time for loop

    with for_(i,0, i<4, i+1): #for loop of pulses with different frequencies
        assign(carrier_frequency, 1e6 + 50e6*i) #change the frequency of the carrier by 20MHz for each iteration startig from 80MHz
        frame_rotation(phi_QUA, 'AOM1')  #setting up the phase of the carrier
        update_frequency('AOM1', carrier_frequency) #setting up the frequency of the carrier
        play('Sin_Envelope', 'AOM1')


#simple linear chirp example
with program() as program3:
    chirp_ratep = declare(int, value=25000)
    frame_rotation(phi, 'AOM1') #setting up the phase of the carrier to phi
    play('Constant', 'AOM1', chirp=(chirp_ratep, 'Hz/nsec')) #playing the Constant pulse with chirp

#simple linear V chirp example
with program() as program4:
    chirp_ratep = declare(int, value=25000)
    chirp_ratem = declare(int, value=-25000)
    frame_rotation(phi, 'AOM1') #setting up the phase of the carrier to phi
    play('Constant', 'AOM1', chirp=(chirp_ratep, 'Hz/nsec')) #playing the Constant pulse with chirp up
    play("Constant", "AOM1", chirp=(chirp_ratem, 'Hz/nsec')) #playing the Constant pulse with chirp down

#simple non-linear chirp example
with program() as program5:
    chirp_rate = declare(int, value=[25000, 0, 10, -30000, 80000])

    frame_rotation(phi, 'AOM1') #setting up the phase of the carrier to phi
    play('Constant', 'AOM1', chirp=(chirp_rate, 'Hz/nsec'))

# rates_v = [int("{:.0f}".format(value)) for value in np.real(800+1000*a5_detuning[0].getPulse()).tolist()];
# print(rates_v)
rates_v = [200, 500, -100, 1500]; 
# times =[0, 15196, 25397, 56925];
#simple non-linear chirp example with times
with program() as program6:
    # rates = declare(int, value=rates_v) #[2000, 5000, -1000, 15000]
    rates = declare(int, value=rates_v)
    times = declare(int, value=[0, 300, 600, 900])

    frame_rotation(phi, 'AOM1') #setting up the phase of the carrier to phi
    play('Constant', 'AOM1', chirp=(rates,times, 'Hz/nsec')) 


##############################################################################################################################################################################
qmm = QuantumMachinesManager(host=ip_address) #open a quantum machines manager instance (running on a remote machine)
qm = qmm.open_qm(config) #open a quantum machine with the current config file

simulation_duration = 80000 #units of 4ns
job = qm.simulate(program1, SimulationConfig(simulation_duration)) #simulate the program
job.get_simulated_samples().con1.plot()


plt.show()