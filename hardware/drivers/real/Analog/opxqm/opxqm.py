#Lab hardware drivers | OPX+ Quantum Machines Waveform generator

#Author(s): Manuel Morgado, Universite de Strasbourg.
#                           Laboratory of Exotic Quantum Matter | Centre Europeen de Sciences Quantiques (CESQ)
#							Universitaet Stuttgart
# 							Physikalische Institute 5, QRydDemo 
#Contributor(s):
#Created: 2023-03-10
#Last update: 2024-12-14

import os

from qm import QuantumMachinesManager
from qm.qua import *
from qm.simulate.credentials import create_credentials
from qm import SimulationConfig, generate_qua_script

import numpy as np
# import matplotlib.pyplot as plt
# import asyncio

#TODO: change2config(), reconnect(), simulate get results, save_results(), write stream_processing

#types
DATA_DEV = 0x2DC6C0;
DIGITAL_DEV = 0x2DC6C1;
ANALOG_DEV = 0x2DC6C2;

#status
STATUS_INACTIVE = 0x2F4D60;
STATUS_ACTIVE = 0x2F4D61;
STATUS_BUSY = 0x2F4D62;

#units
#frequency
mHz = 0x87CDA0;
Hz = 0x87CDA1;
MHz = 0x87CDA2;
GHz = 0x87CDA3;
THz = 0x87CDA4;

#time
ns = 0x87CDA5;
mus = 0x87CDA6;
ms = 0x87CDA7;
s = 0x87CDA8;

#voltage
muV = 0x87CDA9;
mV = 0x87CDAA;
V = 0x87CDAB;

#current
muA = 0x87CDAC;
mA = 0x87CDAD;
A = 0x87CDAE;

#power
dBm = 0x87CDAF;
dB = 0x87CDB0;
dBc = 0x87CDB1;

#angle
rad = 0x87CDB2;
deg = 0x87CDB3;

#sampling
Ss = 0x87CDB4;
KSs = 0x87CDB5;
MSs = 0x87CDB6;
GSs = 0x87CDB7;

DEFAULT_CONFIG = {'version': 1,
                  'controllers': {'con1': {'type': 'opx1',
                                           'analog_outputs': {1: {'offset': 0.0},
                                                              2: {'offset': 0.0},
                                                              3: {'offset': 0.0},
                                                              4: {'offset': 0.0},
                                                              5: {'offset': 0.0},
                                                              6: {'offset': 0.0},
                                                              7: {'offset': 0.0},
                                                              8: {'offset': 0.0},
                                                              9: {'offset': 0.0},
                                                              10: {'offset': 0.0},},
                                           'digital_outputs': {1: {},
                                                               2: {},
                                                               3: {},
                                                               4: {},
                                                               5: {},
                                                               6: {},
                                                               7: {},
                                                               8: {},
                                                               9: {},
                                                               10: {},},
                                           'analog_inputs': {1: {'offset': 0.0},
                                                             2: {'offset': 0.0},},},
                                  'con2': {'type': 'opx1',
                                           'analog_outputs': {1: {'offset': 0.0},
                                                              2: {'offset': 0.0},
                                                              3: {'offset': 0.0},
                                                              4: {'offset': 0.0},
                                                              5: {'offset': 0.0},
                                                              6: {'offset': 0.0},
                                                              7: {'offset': 0.0},
                                                              8: {'offset': 0.0},
                                                              9: {'offset': 0.0},
                                                              10: {'offset': 0.0},},
                                           'digital_outputs': {1: {},
                                                               2: {},
                                                               3: {},
                                                               4: {},
                                                               5: {},
                                                               6: {},
                                                               7: {},
                                                               8: {},
                                                               9: {},
                                                               10: {},},
                                           'analog_inputs': {1: {'offset': 0.0},
                                                             2: {'offset': 0.0},},
                                          }
                                 },
                      'elements': { },
                      'pulses': { },
                      'waveforms': { },
                      'digital_waveforms': { },
                      'integration_weights': { },
                      };
                      
class opxqm:

	'''
	    Python class of the Keysight N5173B Microwave generator.

	    ATTRIBUTES:
	    -----------
			_ID (str) : ID of the device
			_type (str) : type of device: analog, digital and data
			_status (str) : status of the device: active, busy, inactive
			_controller (python-api) : python-wrapper controller object

			__IPaddress (str) : IP address of the device
			__nrChannels (int) : number of channels of the device
			__channelsID (str) : channels ID
			__badwidth (dict) : bandwidth of the device
			__sampling (dict) : sampling rate of the device

			triggerConfig (dict) : trigger configuration dictionary
			triggerLevel (dict) : trigger level value
			acquisitionMode (str) : acquisiton mode: normal, average
			saveMode (str) : storage mode waveform, csv, png
			clock (dict) : clock value
			channelsConfig (dict) : channels configuration dictionary
			channelsData (dict) : channels data values
			horizontalConfig (dict) : horizontal configuration dictionary

	    METHODS:
	    --------
	    
			configure(config) : set configuration of the OPX
			add_param(name, value, unit, label) : add param to the configuration
			update_param(name, value, unit) : update param already within the configuration
			generate_program(name) : creates the program instance within the abstraction of the OPX
			write(command, *args) : add instruction/command in the program of the OPX
			init_manager(ip_address) : initialize the QUA manager for the quantum machine from IP-address
			connect(config) : generate the instance of the quantum machine hardware with configuration for executing jobs
			execute_job(program) : execute program as job in real hardware
			simulate(config, program, simulation duration) : simulates a program and queue as a job
			get_job_results() : get results either fom job executions or simulations
			save_results() : save results from executions of simulations
			disconnect() : close all connections
	      

	'''


	def __init__(self, ADDRESS, ID='0x0', controller='qua', channels_info=None, bandwidth=None, sampling=None, configuration=None, program=None, triggerConfig= {'mode': 'edge', 'source': 1}, triggerLevel= {'values': 0, 'unit': mV}, *args):

		self._ID = ID;
		self._type = DATA_DEV
		self._status = STATUS_INACTIVE;
		self._controller = controller;

		self.__IPaddress = ADDRESS;
		self.__nrChannels = {'digital_outputs': 10, 'analog_outputs': 10, 'analog_inputs': 2};
		self.__channelsID = channels_info;
		self.__bandwidth = bandwidth;
		self.__sampling = sampling;

		self._configs = [];
		self._programs = [];
		self._jobs = [];
		self._simulations = [];
		self._results = [];

		self.config = configuration;
		self.script = [];
		self.program = program;
		self.job = None;
		self.simulation = None;
		self.result = None;
		self.result_signals = {};
		self._manager = None;
		self._hardware = None;

		self.triggerConfig = triggerConfig;
		self.triggerLevel = triggerLevel;


	def configure(self, config):

		'''
			Set configure for QM OPX+. Either from Python
			dictionary or json file. Will load it into
			config attribute.
		'''

		if isinstance(config, dict):
			self.config = config;

		elif isinstance(config, str):
			_file = open(config);

			import json
			self.config = json.load(_file);

		self._configs.append(self.config);
			
	def add_element(self, name:str, info:dict):
		'''
			Add element in config with the corresponding 
			information, following QM structure.
		'''

		self.config['elements'][name] = info;

	def add_pulse(self, name:str, info:dict):
		'''
			Add pulse in config with the corresponding 
			information, following QM structure.
		'''

		self.config['pulses'][name] = info;

	def add_waveforms(self, name:str, info:dict):
		'''
			Add waveform in config with the corresponding 
			information, following QM structure.
		'''
		
		self.config['waveforms'][name] = info;

	def add_digitalwaveforms(self, name:str, info:dict):
		'''
			Add digital waveform in config with the corresponding 
			information, following QM structure.
		'''
		
		self.config['digital_waveforms'][name] = info;

	def add_integration_weights(self, name:str, info:dict):
		'''
			Add integration waight in config with the corresponding 
			information, following QM structure.
		'''
		
		self.config['integration_weights'][name] = info;

	def change2config(self, parameter:str, change):
		'''
			Change specific parameter in configuration, given the 
			full position/direction within config, following QM
			structure format.
		'''
		#TODO: separate string and reach in dictionary and subsitute
		
	def generate_program(self):
		'''
			Instatiate class of program where instructions will be added
		'''
		self.program = program();

	def write(self, command:str, *args):
		'''
			Write full instructions: command+arguments as string and store
			it into script attribute list, used later to execute job within context.
		'''
		if command=='play':
			self.script.append("play({ARG1}, {ARG2})".format(ARG1=args[0], ARG2=args[1]));
			return

		if command=='align':
			self.script.append("align({ARG1}, {ARG2})".format(ARG1=args[0], ARG2=args[1]));
			return

		if command=='wait':
			self.script.append("wait({ARG1})".format(ARG1=args[0]));
			return

		if command=='measure':
			self.script.append("measure({ARG1}, {ARG2}, {ARG3})".format(ARG1=args[0], ARG2=args[1], ARG3=args[2]));
			return

		if command=='wait_for_triggrer':
			self.script.append("wait_for_trigger({ARG1})".format(ARG1=args[0]));
			return

		if command=='reset_phase':
			self.script.append("reset_phase({ARG1})".format(ARG1=args[0]));
			return

		if command=='assign':
			self.script.append("assign({ARG1}, {ARG2})".format(ARG1=args[0], ARG2=args[1]));
			return

		if command=='declare':
			self.script.append("{ARG1} = declare({ARG2}, {ARG3})".format(ARG1=args[0], ARG2=args[1], ARG3=args[2]));
			return

		if command=='save':
			self.script.append("save({ARG1})".format(ARG1=args[0]));
			return

		if command=='declare_stream':
			self.script.append("{ARG1} = declare_stream()".format(ARG1=args[0]));
			return

		if command=='wait_for_trigger':
			self.script.append("wait_for_trigger({ARG1})".format(ARG1=args[0]));
			return

		if command=='update_frequency':
			self.script.append("update_frequency({ARG1}, int({ARG2}), {ARG3})".format(ARG1=args[0], ARG2=args[1], ARG3=args[2]));
			return
		
		if command=='frame_rotation_2pi':
			self.script.append("frame_rotation_2pi({ARG1}, {ARG2})".format(ARG1=args[0], ARG2=args[1]));
			return

	def init_manager(self, ip_address=None):
		'''
			Initialize QM manager device from IP address.
		'''

		if ip_address!=None:
			self.__IPaddress = ip_address;
			
		self._manager = QuantumMachinesManager(host=self.__IPaddress);

	def connect(self, config=None):
		'''
			Connect to device (OPX+) via instantiation.
		'''

		if config!=None:
			self.config = config;
			
		self.hardware = self._manager.open_qm(self.config);

	def reconnect(self):
		pass 

	def execute_job(self, _program=None):
		'''
			Execute program as job, either from given program or
			from attribute. For the second it will compile script
			attribute into program context, store it as program and
			execute.
		'''

		if _program!=None:
			if isinstance(_program, type(program())):
				self.program = _program;
		else:
			with program() as self.program:
				for instruction in self.script:
					exec(instruction);		

		
		self.job = self.hardware.execute(self.program);
		self.result = self.job.result_handles.wait_for_all_values();

		self._programs.append(self.program);
		self._jobs.append(self.job);
		self._results.append({'type': 'execution', 'result': self.result});

	def simulate(self, simulation_time, config=None, _program=None):
		'''
			Simulate
		'''
		if config!=None:
			self.connect(config);
		if _program!=None:
			if isinstance(_program, type(program())):
				self.program = _program;
		else:
			with program() as self.program:
				for instruction in self.script:
					exec(instruction);		

		self.job = self._manager.simulate(self.config, self.program, SimulationConfig(simulation_time));
		# self.result = self.job.result_handles.wait_for_all_values();

		self._jobs.append(self.job);
		# self._results.append({'type': 'simulation', 'result': self.result});

	def get_program(self, ALL=False):
		'''
			Return last program if ALL is False, return list of programs 
			otherwise.
		'''

		if ALL:
			return self._programs

		else:
			return self.program

	def get_job_results(self, controllers=['con1']):
		'''
			Return last result if ALL is False, return list of results 
			otherwise.
		'''
		
		if controllers=='all':
			self.result_signals['con1']['ANALOG'] = exec("self.job.get_simulated_samples()."+"__dict__['con1']"+".analog")
			self.result_signals['con1']['ANALOG'] = exec("self.job.get_simulated_samples()."+"__dict__['con2']"+".analog")

			self.result_signals['con1']['DIGITAL'] = exec("self.job.get_simulated_samples()."+"__dict__['con1']"+".digital")
			self.result_signals['con1']['DIGITAL'] = exec("self.job.get_simulated_samples()."+"__dict__['con2']"+".digital")

			return self._results

		else:
			for controller in controllers:
				return self.result

	def plot_simulation(self):
		'''
			Plot simulation results from specific controllers
		'''


		exec("self.job.get_simulated_samples()."+"con1"+".plot()");
		# exec("self.job.get_simulated_samples()."+"con2"+".plot()");

	def save_results(self, ):
		pass

	def disconnect(self, all_opx=True, manager=False):
		'''
			Disconnect all the quantum machines open if all_opx is True,
			and disconnect manager if True
		'''

		if all_opx==True:
			self._manager.close_all_quantum_machines();

		if manager!=False:
			self._manager.close();

	def list_open_qm(self):
		'''
			Returns list of open quantum machines devies
		'''
		return self._manager.list_open_quantum_machines()

	def get_QUA_script(self, prog,config, fname):

		generate_qua_script(prog, config, file=fname)

	
	def clear_screen():
	    '''
	        Function that clears the screen by choosing the right OS
	    '''

	    #detect the OS
	    if sys.platform.startswith('win'): #windows systems
	        os.system('cls')

	    elif sys.platform.startswith('linux') or sys.platform == 'darwin':  #unix systems
	        os.system('clear')

	    else:
	        print("Unsupported OS")

	
	def get_chirp_rate(self, duration:int, frequency_range:list, verbose:bool=False):
	    '''
	        Calculates the chirp rates given the frequency range and the time duration

	        INPUTS:
	        -------

	        duration (int): chirp duration in clock cycle units of the OPX i.e., 4ns
	        frequency_range (list): range of the chirp in Hz

	        OUTPUTS:
	        --------

	        Returns the chirp rate in units of Hz/nsec


	        TODO: use QUA compiled commands
	    '''
	    if verbose==True:
	        print('calculated rate: ', (frequency_range[1]-frequency_range[0])/(4*duration), '\n', 'parameters for chirp_rate: ', frequency_range[1], frequency_range[0], 4*duration)
	    return (frequency_range[1]-frequency_range[0])/(4*duration)

	def AODs_alloc_report(self, config:dict, used_ports:list=[7,8,9,10]):
	    '''
	        Function to get all the QUA elements from the specified ports of the opx unit. This is useful to generate the QUA_QPU_MAP
	        in order to set the elements names in the map.

	        In other words, shows the AOD allocations of the tones/qubits.

	        INPUTS:
	        -------

	        config (dict): QUA configuration python dictionary
	        used_ports (list): list of ports included in the report of AOD allocations

	        OUTPUTS:
	        --------

	        _AODs_alloc (list): list containing the report of the elements per port including the assigned threads of the OPX
	    '''

	    MHZ_to_Hz = 1e-6 #set transformation from MHz to Hz
	    print('\n')

	    _AODs_alloc=[[] for _ in range(len(used_ports))]

	    print(' AOD1 X (Port {PORT1X}): \n ++++++++++++++++'.format(PORT1X=used_ports[0]))
	    for key, val in config['elements'].items():

	        #check elements sharing port 0 of the list
	        if val['singleInput']['port'][1] == used_ports[0]:

	            try: #check if oscillator is assigned
	                print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=val['intermediate_frequency']*MHZ_to_Hz))
	                _AODs_alloc[0].append(key)
	            except:
	                try:    #check if intermediate frequency is assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))
	                    _AODs_alloc[0].append(key)
	                except: #check if thread assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= '--', OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))

	    print('\n AOD1 Y (Port {PORT1X}): \n ++++++++++++++++'.format(PORT1X=used_ports[1]))
	    for key, val in config['elements'].items():

	         #check elements sharing port 1 of the list
	        if val['singleInput']['port'][1] == used_ports[1]:

	            try: #check if oscillator is assigned
	                print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=val['intermediate_frequency']*MHZ_to_Hz))
	                _AODs_alloc[1].append(key)
	            except:
	                try: #check if intermediate frequency is assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))
	                    _AODs_alloc[1].append(key)
	                except: #check if thread assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= '--', OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))

	    print('\n AOD2 X (Port {PORT1X}): \n ++++++++++++++++'.format(PORT1X=used_ports[2]))
	    for key, val in config['elements'].items():
	        # print(key ,'|', val['thread'],'\n' )

	         #check elements sharing port 2 of the list
	        if val['singleInput']['port'][1] == used_ports[2]:

	            try: #check if oscillator is assigned
	                print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=val['intermediate_frequency']*MHZ_to_Hz))
	                _AODs_alloc[2].append(key)

	            except:
	                try: #check if intermediate frequency is assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))
	                    _AODs_alloc[2].append(key)
	                except: #check if thread assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= '--', OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))

	    print('\n AOD2 Y (Port {PORT1X}): \n ++++++++++++++++'.format(PORT1X=used_ports[3]))
	    for key, val in config['elements'].items():

	         #check elements sharing port 3 of the list
	        if val['singleInput']['port'][1] == used_ports[3]:

	            try: #check if oscillator is assigned
	                print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=val['intermediate_frequency']*MHZ_to_Hz))
	                _AODs_alloc[3].append(key)
	            except:
	                try: #check if intermediate frequency is assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= val['thread'], OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))
	                    _AODs_alloc[3].append(key)
	                except: #check if thread assigned
	                    print(key, ' | {THREAD} | {OSCILLATOR} | {FREQUENCY} '.format(THREAD= '--', OSCILLATOR="--", FREQUENCY=0*MHZ_to_Hz))

	    return _AODs_alloc

	def QPU_QUA_map(self, report:list, map_dict:dict):
	    '''
	        Function that substitute the values of the map of QUA elements in the QPU map.

	        INPUTS:
	        -------

	        report (list): report returnes by the AODs_alloc_report() [maybe have to be generalized and change name]
	        map_dict (dict): dictionary of the QUA QPU map


	        OUTPUTS:
	        --------

	        map_dict (dict): dictionary of the mapping of the QPU including the names of the elements from the config
	    '''

	    _qubit_idx = 0
	    for _xalloc in range(0,len(report[2]),1):


	        for _yalloc in range(0, len(report[3]), 1):
	            map_dict['elements'][_qubit_idx] = [ [ report[2][_xalloc], report[3][_yalloc] ], [ report[0][_xalloc], report[1][_yalloc] ] ]
	            _qubit_idx+=1

	    return map_dict

	
	def initialize_allocations(self, allocations, frequencies, phases):

	    for _allocation,_frequency, _phase in zip(allocations, frequencies, phases):

	        reset_frame(_allocation)
	        frame_rotation(phases, _allocation)
	        update_frequency(_allocation, frequency)

	def _set_tone_fp(self, allocation, frequency, phase):

	    reset_frame(_allocation)
	    frame_rotation(phases, _allocation)
	    update_frequency(_allocation, frequency)

	
	def _address(self, register, new_amplitudes:list=None, new_frequencies:list=None, new_phases:list=None, _duration=None, _map=None):

	    if isinstance(register, int): #single qubit gate in a single qubit at the layer

	        #get map qubit index -> qua elements
	        # _map = QPU_MAP['elements'][register]

	        #overwriting frequency if used
	        if not isinstance(new_frequencies, type(None)):
	            update_frequency(_map[0][0], new_frequencies[0], units='Hz', keep_phase=True) # 1x0 keep_phase: True for continuos phase, False for phase coherent
	            update_frequency(_map[0][1], new_frequencies[1], units='Hz', keep_phase=True) # 1y0
	            # update_frequency(_map[1][0], new_frequencies[2], units='Hz', keep_phase=True) # 2x0
	            # update_frequency(_map[1][1], new_frequencies[3], units='Hz', keep_phase=True) # 2y0

	        #overwriting phase if used
	        if not isinstance(new_phases, type(None)):
	            frame_rotation(new_phases[0], _map[0][0])
	            frame_rotation(new_phases[1], _map[0][1])
	            # frame_rotation(new_phases[2], _map[1][0])
	            # frame_rotation(new_phases[3], _map[1][1])


	        #overwriting amplitudes if used
	        if not isinstance(new_phases, type(None)):
	            #play the tones in X&Y of the 1&2 AODs
	            play('address'*amp(new_amplitudes[0]), _map[0][0], duration=_duration) #ON during pulse time
	            play('address'*amp(new_amplitudes[1]), _map[0][1], duration=_duration) #ON during pulse time
	            # play('address'*amp(new_amplitudes[2]), _map[1][0], duration=_duration) #ON during pulse time
	            # play('address'*amp(new_amplitudes[3]), _map[1][1], duration=_duration) #ON during pulse time
	        
	        else: #play the tones with variables.py values

	            if _duration!=None: #with set duration
	                
	                #play the tones in X&Y of the 1&2 AODs
	                play('address'*amp(carrier_frequency_factors_tones['CHA07']), _map[0][0], duration=_duration) #ON during pulse time
	                play('address'*amp(carrier_frequency_factors_tones['CHA08']), _map[0][1], duration=_duration) #ON during pulse time
	                # play('address'*amp(carrier_frequency_factors_tones['CHA09']), _map[1][0], duration=_duration) #ON during pulse time
	                # play('address'*amp(carrier_frequency_factors_tones['CHA10']), _map[1][1], duration=_duration) #ON during pulse time
	            
	            else: #duration given by the pulse 'address'
	                
	                #play the tones in X&Y of the 1&2 AODs
	                play('address'*amp(carrier_frequency_factors_tones['CHA07']), _map[0][0]) #always ON
	                play('address'*amp(carrier_frequency_factors_tones['CHA08']), _map[0][1]) #always ON
	                # play('address'*amp(carrier_frequency_factors_tones['CHA09']), _map[1][0]) #always ON
	                # play('address'*amp(carrier_frequency_factors_tones['CHA10']), _map[1][1]) #always ON


	    elif isinstance(register, list): #simultaneous single or multiqubit gates

	        for _reg in register:


	            if len(_reg)==1: #single qubit gate

	                #get map qubit index -> qua elements
	                _map = QPU_MAP['elements'][_reg[0]]

	                #updating frequency if used
	                if not isinstance(new_frequencies, type(None)):
	                    update_frequency(new_frequencies[0], _map[0][0], units='Hz', keep_phase=True) # keep_phase: True for continuos phase, False for phase coherent
	                    update_frequency(new_frequencies[1], _map[0][1], units='Hz', keep_phase=True)
	                    update_frequency(new_frequencies[2], _map[1][0], units='Hz', keep_phase=True)
	                    update_frequency(new_frequencies[3], _map[1][1], units='Hz', keep_phase=True)

	                #updating phase if used
	                if not isinstance(new_phases, type(None)):
	                    frame_rotation(new_phases[0], _map[0][0])
	                    frame_rotation(new_phases[1], _map[0][1])
	                    frame_rotation(new_phases[2], _map[1][0])
	                    frame_rotation(new_phases[3], _map[1][1])

	                #play the tones in X&Y of the 1&2 AODs
	                play('address'*amp(carrier_frequency_factors_tones['CHA07']), _map[0][0])
	                play('address'*amp(carrier_frequency_factors_tones['CHA08']), _map[0][1])
	                play('address'*amp(carrier_frequency_factors_tones['CHA09']), _map[1][0])
	                play('address'*amp(carrier_frequency_factors_tones['CHA10']), _map[1][1])


	            else: #multiqubit gate

	                for i in range(len(_reg)):
	                    globals()[f'_chunk{i+1}'] = _reg[i]

	                    _map = QPU_MAP['elements'][globals()[f'_chunk{i+1}']]

	                    #updating frequency if used
	                    if not isinstance(new_frequencies, type(None)):
	                        update_frequency(new_frequencies[0], _map[0][0], units='Hz', keep_phase=True) # keep_phase: True for continuos phase, False for phase coherent
	                        update_frequency(new_frequencies[1], _map[0][1], units='Hz', keep_phase=True)
	                        update_frequency(new_frequencies[2], _map[1][0], units='Hz', keep_phase=True)
	                        update_frequency(new_frequencies[3], _map[1][1], units='Hz', keep_phase=True)

	                    #updating phase if used
	                    if not isinstance(new_phases, type(None)):
	                        frame_rotation(new_phases[0], _map[0][0])
	                        frame_rotation(new_phases[1], _map[0][1])
	                        frame_rotation(new_phases[2], _map[1][0])
	                        frame_rotation(new_phases[3], _map[1][1])

	                    #play the tones in X&Y of the 1&2 AODs
	                    play('address'*amp(carrier_frequency_factors_tones['CHA09']), _map[0][0])
	                    play('address'*amp(carrier_frequency_factors_tones['CHA09']), _map[0][1])
	                    play('address'*amp(carrier_frequency_factors_tones['CHA10']), _map[1][0])
	                    play('address'*amp(carrier_frequency_factors_tones['CHA10']), _map[1][1])

	#TODO: schedules, 2-qubit gates solution, corrections, strict timming, wait, check generalization of using AODxy or multiple DP 
	def _gate(self, qubit_idx:int, gate:str, args:list, calibration:dict):

	    if gate=='x':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        wait(100)
	        play('CW'*amp(0.5), 'DP__beam1')
	        play('CW'*amp(0.5), 'DP__beam2')

	    if gate=='y':

	        # correction in the AOD? according to position (i.e., qubit_idx)

	        update_frequency('DP__beam2', 10e6) #reset phase

	        #gate schedule
	        play('gaussian'*amp(0.5), 'DP__beam1')
	        play('gaussian'*amp(0.5), 'DP__beam2')

	    if gate=='z':
	        # correction in the AOD? according to position (i.e., qubit_idx)

	        update_frequency('DP__beam2', 10e6) #reset phase

	        #gate schedule
	        play('gaussian'*amp(0.5), 'DP__beam1')
	        play('gaussian'*amp(0.5), 'DP__beam2')

	    if gate=='h':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('gaussian_pi'*amp(0.5), 'DP__beam1')
	        play('gaussian_pi'*amp(0.5), 'DP__beam2')

	    if gate=='Rx':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('CW'*amp(0.5), 'DP__beam1')
	        play('CW'*amp(0.5), 'DP__beam2')

	    if gate=='Ry':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('gaussian_pi'*amp(0.5), 'DP__beam1')
	        play('gaussian_pi'*amp(0.5), 'DP__beam2')

	    if gate=='Rz':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('gaussian_pi'*amp(0.5), 'DP__beam1')
	        play('gaussian_pi'*amp(0.5), 'DP__beam2')
	        
	    if gate=='cz':
	        wait(100)

	        #TODO: 2 qubit gates have to run via 2D AOD tones, not a single switch
	        #control qubit(s)
	        update_frequency('DP__beam2', 10e6) #for correction
	        play('gaussian'*amp(0.5), 'DP__beam1')
	        play('gaussian'*amp(0.5), 'DP__beam2')

	        #target qubit(s)
	        update_frequency('DP__beam2', 10e6) #for correction
	        play('gaussian'*amp(0.5), 'DP__beam1')
	        play('gaussian'*amp(0.5), 'DP__beam2')

	    if gate=='cnot':
	        wait(100)

	        #TODO: 2 qubit gates have to run via 2D AOD tones, not a single switch
	        #control qubit(s)
	        update_frequency('DP__beam2', 10e6)
	        play('CW'*amp(0.5), 'DP__beam1')
	        play('CW'*amp(0.5), 'DP__beam2')

	        #target qubit(s)
	        update_frequency('DP__beam2', 10e6)
	        play('CW'*amp(0.5), 'DP__beam1')
	        play('CW'*amp(0.5), 'DP__beam2')

	    if gate=='transcoding':

	        play('gaussian_pi'*amp(0.5), 'DP__beam1')

	    if gate=='reset':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('gaussian_pi'*amp(0.5), 'DP__beam1')
	        play('gaussian_pi'*amp(0.5), 'DP__beam2')

	    if gate=='measurement':

	        # correction in the AOD? according to position (i.e., qubit_idx)


	        reset_frame('DP__beam1') #reset phase

	        #gate schedule
	        play('gaussian_pi'*amp(0.5), 'DP__beam1')
	        play('gaussian_pi'*amp(0.5), 'DP__beam2')

	def operation(self, layer:dict):

	    if 'alloc' in layer: #single tone operation

	        global _map
	        _map = QPU_MAP['elements'][layer['alloc'][0]] #get _map for aligning gates and allocations
	        # align(_map[0][0], _map[0][1], _map[1][0], _map[1][1], 'DP__beam1', 'DP__beam3')
	        
	        #single qubit operation in layer
	        _gate(layer['alloc'][0], layer['gate'][0], layer['args'], layer['calibration']) #execute gate schedule on that qubit     #TODO: pass arguments if only the gate requires it i.e., if it is Rx, Ry, Rz
	        if 'alloc_settings' in layer: #check if there is any overwritting of the allocations
	            _address(register=layer['alloc'][0], new_amplitudes= layer['alloc_settings']['amplitudes'], new_frequencies=layer['alloc_settings']['frequencies'], new_phases=layer['alloc_settings']['phases'], _duration=layer['args'][3], _map=_map) #set (overwrite) qubit address following the QPU_MAP
	        else:
	            _address(layer['alloc'][0],  _duration=layer['args'][2], _map=_map) #set qubit address following the QPU_MAP

	    elif 'allocs' in layer: #multi tone operation
	        #multiple qubit operation in layer
	        _address(layer['allocs']) #set qubit address following the QPU_MAP
	        for _alloc, _gatename, _args in zip(layer['allocs'], layer['gates'], layer['args']):
	            _gate(_alloc, _gatename, _args) #execute gate schedule on that qubit     
	    
	
	#TODO
	def qubit_calibration(self, cal_type, idx, params):
	    '''
	        Qubit calibration running at before the layer of the quantum circuit. 
	        It executes the schedules for Ramsey or Rabi or Echo or any combination
	        of the three protocols. It is specified in the calibrariton layer. 

	        Multiple specified qubit calibration will do interlayer qubit calibration.
	        (Migth be just one needed).

	        Ex.

	        'calibration': {'waveform':{'active':False, 'params': [] },
	                        'area': {'active':False, 'params': []},
	                        'qubit': {'active': True, 'params': [<RabiParams>, <RamseyParams>, <EchoParams>]}        <--------
	                        },
	    '''

	    if cal_type=='ramsey':
	        '''
	            Ramsey calibration of the qubit executing two pi/2 pulses
	        '''
	        print('Not implemented yet.')
	        pass

	    if cal_type=='rabi':
	        '''
	            Rabi calibration executing a sequence of continuous driving pulse
	        '''
	        print('Not implemented yet.')
	        pass

	    if cal_type=='echo':
	        '''
	            Echo calibration executing a sequence of Ramsey pulses with intermediate pulses
	        '''
	        print('Not implemented yet.')
	        pass

	#TODO
	def area_calibration(self, gate, idx, params):
	    '''
	        Area calibration running layer as a calibration layer to adjust area under the pulse acquired
	        signal to calibrate the Amplitude and time duration.

	        Ex.
	        'calibration': {'waveform':{'active':False, 'params': [] },   
	                        'area': {'active':True, 'params': [<Area>, <Time>, <#shots>]},        <--------
	                        'qubit': {'active': False, 'params': []}        
	                        },
	    '''
	    print('Not implemented yet.')
	    pass

	def waveform_calibration(self, gate, idx, params):
	    '''
	        Waveform calibration running for given layer to calibrate the operations on it. Based on a PID
	        approach to tune the parameters of the PID, number of shots, the target and the alpha parameter.

	        Ex. 
	        'calibration': {'waveform':{'active':True, 'params': [<target>, <P>, <I>, <D>, <alpha>, <#shots>] },        <--------
	                        'area': {'active':False, 'params': []}, 
	                        'qubit': {'active': False, 'params': []}        
	                        },
	    '''
	    target, gain_P, gain_I, gain_D, alpha, N_shots, = params

	    # with program() as Feed_forward:

	    #variable declaration
	    _saving = declare(bool, value=False)

	    # adc_st = declare_stream(adc_trace=True)
	    n = declare(int)
	    integ_iCHA01 = declare(fixed)
	    integ_iCHA02 = declare(fixed)

	    amplitude_iCHA01 = declare(fixed, value=0.0)
	    amplitude_iCHA02 = declare(fixed, value=0.0)

	    error_iCHA01 = declare(fixed, value=0)
	    integrator_error_iCHA01 = declare(fixed, value=0)
	    derivative_error_iCHA01 = declare(fixed)
	    old_error_iCHA01 = declare(fixed, value=0.0)
	    total_error_iCHA01 = declare(fixed)

	    error_iCHA02 = declare(fixed, value=0)
	    integrator_error_iCHA02 = declare(fixed, value=0)
	    derivative_error_iCHA02 = declare(fixed)
	    old_error_iCHA02 = declare(fixed, value=0.0)
	    total_error_iCHA02 = declare(fixed)

	    #stream declaration
	    error_st_iCHA01 = declare_stream()
	    integral_st_iCHA01 = declare_stream()
	    integrator_error_st_iCHA01 = declare_stream()
	    amp_st_iCHA01 = declare_stream()
	    total_error_st_iCHA01 = declare_stream()

	    error_st_iCHA02 = declare_stream()
	    integral_st_iCHA02 = declare_stream()
	    integrator_error_st_iCHA02 = declare_stream()
	    amp_st_iCHA02 = declare_stream()
	    total_error_st_iCHA02 = declare_stream()


	    with for_(n, 0, n < N_shots, n+1):
	        # align('DP__beam1', 'PD_1', 'PD_1')

	        # the pulse is played by the _gate() function meanwhile waveform_calibration() is acquiring the data from the input ports of the OPX
	        # using PD_1 and PD_2 and calculating the correction via both channels.                
	        measure("acquire",  "PD_1",  None,  integration.full('constant', integ_iCHA01, 'feedback')) #(<integration weights>, <variable output>, <analog port>)
	        measure("acquire",  "PD_2",  None,  integration.full('constant', integ_iCHA02, 'feedback')) #(<integration weights>, <variable output>, <analog port>)
	        
	        #calculate the error
	        assign(error_iCHA01, (target-integ_iCHA01)<<11) 
	        # assign(error_iCHA02, (target-integ_iCHA02)<<11) 
	        
	        #calculate the integrator error with exponentially decreasing weights with coefficient alpha
	        assign(integrator_error_iCHA01, (1 - alpha) * integrator_error_iCHA01 + alpha * error_iCHA01) 
	        assign(integrator_error_iCHA02, (1 - alpha) * integrator_error_iCHA02 + alpha * error_iCHA02) 

	        #calculate the derivative error
	        assign(derivative_error_iCHA01, old_error_iCHA01-error_iCHA01) 
	        assign(derivative_error_iCHA02, old_error_iCHA01-error_iCHA02)

	        #save new value of amplitude
	        assign(amplitude_iCHA01, amplitude_iCHA01 + (gain_P * error_iCHA01 + gain_I * integrator_error_iCHA01 + gain_D * derivative_error_iCHA01))
	        assign(amplitude_iCHA02, amplitude_iCHA02 + (gain_P * error_iCHA02 + gain_I * integrator_error_iCHA02 + gain_D * derivative_error_iCHA02))

	        # assign(amplitude_iCHA01, amplitude_iCHA01 + (gain_P * error_iCHA01 + gain_I * integrator_error_iCHA01 + gain_D * derivative_error_iCHA01)) #correct the amplitude accordingly PID when correcting directly in the element
	        set_dc_offset(element='PD_1', element_input='single', offset=amplitude_iCHA01) #correct the amplitude via an offset of the element port for iCHA01
	        set_dc_offset(element='PD_2', element_input='single', offset=amplitude_iCHA02) #correct the amplitude via an offset of the element port for iCHA01

	        #save old error to be error
	        assign(old_error_iCHA01, error_iCHA01) 
	        assign(old_error_iCHA02, error_iCHA02)

	        #saving iCHA01
	        save(integ_iCHA01, integral_st_iCHA01)
	        save(error_iCHA01, error_st_iCHA01)
	        save(amplitude_iCHA01, amp_st_iCHA01)
	        save(total_error_iCHA01, total_error_st_iCHA01)
	        save(integrator_error_iCHA01, integrator_error_st_iCHA01)

	        #saving iCHA02
	        save(integ_iCHA02, integral_st_iCHA02)
	        save(error_iCHA02, error_st_iCHA02)
	        save(amplitude_iCHA02, amp_st_iCHA02)
	        save(total_error_iCHA02, total_error_st_iCHA02)
	        save(integrator_error_iCHA02, integrator_error_st_iCHA02)

	    # with if_(_saving==True):
	    with stream_processing():

	        integral_st_iCHA01.save_all('integral_iCHA01')
	        error_st_iCHA01.save_all('error_iCHA01')
	        amp_st_iCHA01.save_all('amplitude_iCHA01')
	        integrator_error_st_iCHA01.save_all('integrator_error_iCHA01')
	        total_error_st_iCHA01.save_all('total_error_iCHA01')

	        integral_st_iCHA02.save_all('integral_iCHA02')
	        error_st_iCHA02.save_all('error_iCHA02')
	        amp_st_iCHA02.save_all('amplitude_iCHA02')
	        integrator_error_st_iCHA02.save_all('integrator_error_iCHA02')
	        total_error_st_iCHA02.save_all('total_error_iCHA02')
	    
	def waveform_tuning_monitor(self, job, N_shots=[]):
	    
	    #extracting info from the execution
	    job.result_handles.wait_for_all_values()
	    res = job.result_handles
	    res.wait_for_all_values()
	    
	    # adc1 = res.get("adc1").fetch_all()
	    iCHA01_error = res.get('error_iCHA01').fetch_all()['value']
	    iCHA01_total_error = res.get('total_error_iCHA01').fetch_all()['value']
	    iCHA01_integral = res.get('integral_iCHA01').fetch_all()['value']
	    iCHA01_amplitude = res.get('amplitude_iCHA01').fetch_all()['value']
	    iCHA01_integrator_error = res.get('integrator_error_iCHA01').fetch_all()['value']

	    iCHA02_error = res.get('error_iCHA02').fetch_all()['value']
	    iCHA02_total_error = res.get('total_error_iCHA02').fetch_all()['value']
	    iCHA02_integral = res.get('integral_iCHA02').fetch_all()['value']
	    iCHA02_amplitude = res.get('amplitude_iCHA02').fetch_all()['value']
	    iCHA02_integrator_error = res.get('integrator_error_iCHA02').fetch_all()['value']

	    Time = [x*0.38*0+x for x in range(N_shots)] # I calculated on the scope 0.38us between pulses
	    # print(integral[N_shots:-1].std() )# just checking the std after the lock
	    
	    # plots
	    plt.figure(figsize=(30,4))
	    plt.plot(Time, iCHA01_error, '.-', linewidth=5, markersize=12)
	    plt.title('Intensity lock error', fontsize=12)
	    plt.xlabel('Time [μs]', fontsize=12)
	    plt.ylabel('Amplitude Error [arb. units]', fontsize=12)
	    plt.xticks(fontsize= 12)
	    plt.yticks(fontsize= 12)
	    
	    plt.figure(figsize=(30,4))
	    plt.plot(Time, iCHA01_integral)
	    plt.title('Intensity')
	    plt.figure(figsize=(30,4))
	    plt.plot(Time, iCHA01_amplitude)
	    plt.title('Applied amplitude')
	    plt.figure(figsize=(30,4))
	    plt.plot(Time, iCHA01_integrator_error)
	    plt.title('integrator error')
	    plt.figure(figsize=(30,4))
	    plt.plot(Time, iCHA01_total_error)
	    plt.title('total error')

	    plt.show()

	    