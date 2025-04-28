#Atomic Quantum information Processing Tool (AQIPT) - Real devices module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2022-09-01
# Last update: 2023-05-26

#libs
import numpy as np
import matplotlib.pyplot as plt


import time, os, platform
import copy

import AQiPT.modules.control.AQiPTcontrol as control
import AQiPT.modules.kernel.AQiPTkernel as kernel
from AQiPT.modules.directory import AQiPTdirectory as dirPath


'''
	IMPORTANT NOTE:
	---------------

	The recognized drivers of real devices, must be in the folder AQiPT>hardware>drivers>Analog/DAQ/Data/Digital any other folder or type is not
	included. The structure of the driver must follow:

		folder of the driver name: <NAME>
		.py file name: <NAME>
		class container name of the driver: <NAME>

	i.e., all <NAME> must be the same string.

	The real drivers contains the class instead of the instance, such can be reused for the multiple similar devices installed. 

	For external drivers, the driver on AQiPT file tree is a pointer to the instance of the module containing the driver i.e., <NAME> for the code below

	driver_EXTERNAL.py
	|
	|	from EXTERNAL_module import DEVICE_DRIVER
	|
	|	class _<NAME>:
	|	|
	|	|	def __init__(self, args):
	|	|		self._EXT_driver = args['external_driver']
	|	
	|	<NAME> = _<NAME>(args=DEVICE_DRIVER)

'''

AQiPT_DRIVER_ANALAOG = 'AQiPT.hardware.drivers.real.Analog.';
AQiPT_DRIVER_DAQ = 'AQiPT.hardware.drivers.real.DAQ.';
AQiPT_DRIVER_DATA = 'AQiPT.hardware.drivers.real.Data.';
AQiPT_DRIVER_DIGITAL = 'AQiPT.hardware.drivers.real.Digital.';

class real_drivers:


	def __init__(self, pathDir=None):

		self._pathDir = pathDir;

		if platform.system() == 'Windows':
			self._real_dev_directory_LST = [direc[0] for direc in os.walk(self._pathDir) if direc[0].count('\\') <11 and direc[0].count('\\')>9 ];

		elif platform.system() == 'Linux':
			self._real_dev_directory_LST = [direc[0] for direc in os.walk(self._pathDir) if direc[0].count('/') <10 and direc[0].count('/')>8 ];

		self._analog_dev = None;
		self._daq_dev = None;
		self._data_dev = None;
		self._digital_dev = None;
		
		#WARNING: sometimes the order of the directories below changes 
		self._init_analog(path2folder= [element for element in self._real_dev_directory_LST if element.endswith('Analog')][0]);
		self._init_daq(path2folder= [element for element in self._real_dev_directory_LST if element.endswith('DAQ')][0]);
		self._init_data(path2folder= [element for element in self._real_dev_directory_LST if element.endswith('Data')][0]);
		self._init_digital(path2folder= [element for element in self._real_dev_directory_LST if element.endswith('Digital')][0]);

	def _init_analog(self, path2folder):
		
		self._analog_dev = analog_drivers(path2folder);
		self._analog_dev.wrap();


	def _init_daq(self, path2folder):
		self._daq_dev = daq_drivers(path2folder);
		self._daq_dev.wrap();


	def _init_data(self, path2folder):
		self._data_dev = data_drivers(path2folder);
		self._data_dev.wrap();


	def _init_digital(self, path2folder):
		self._digital_dev = digital_drivers(path2folder);
		self._digital_dev.wrap();



class analog_drivers:

	def __init__(self, path2folder=None):
		self._directory = path2folder;
	
	def wrap(self):

		if platform.system() == 'Windows':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('\\')<12 and  x[0].count('\\')>9 ]:
				_driver_name = string.rsplit('\\', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_ANALAOG+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

		elif platform.system() == 'Ubuntu':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('/')<11 and  x[0].count('/')>9 ]:

				_driver_name = string.rsplit('/', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_ANALAOG+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))

				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))
		
		elif platform.system() == 'Linux':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('/')<11 and  x[0].count('/')>9 ]:

				_driver_name = string.rsplit('/', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_ANALAOG+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))

				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))
		
class daq_drivers:

	def __init__(self, path2folder=None):
		self._directory = path2folder;

	def wrap(self):

		if platform.system() == 'Windows':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('\\')<12 and  x[0].count('\\')>9 ]:

				_driver_name = string.rsplit('\\', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_DAQ+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

		elif platform.system() == 'Ubuntu':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('/')<11 and  x[0].count('/')>9 ]:

				_driver_name = string.rsplit('/', 1)[1];

				dirPath.setCurrentDir(self._directory);
				print(_driver_name)
				try:
					module_name = AQiPT_DRIVER_DAQ+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

class data_drivers:

	def __init__(self, path2folder=None):
		self._directory = path2folder;
	
	def wrap(self):

		if platform.system() == 'Windows':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('\\')<12 and  x[0].count('\\')>9 ]:

				_driver_name = string.rsplit('\\', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_DATA+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

		elif platform.system() == 'Ubuntu':
			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('/')<11 and  x[0].count('/')>9 ]:

				_driver_name = string.rsplit('/', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_DATA+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

class digital_drivers:

	def __init__(self, path2folder=None):
		self._directory = path2folder;
	
	def wrap(self):

		if platform.system() == 'Windows':

			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('\\')<12 and  x[0].count('\\')>9 ]:

				_driver_name = string.rsplit('\\', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_DIGITAL+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

		elif platform.system() == 'Ubuntu':

			for string in [x[0] for x in os.walk(self._directory) if  x[0].count('/')<11 and  x[0].count('/')>9 ]:

				_driver_name = string.rsplit('/', 1)[1];

				dirPath.setCurrentDir(self._directory);

				try:
					module_name = AQiPT_DRIVER_DIGITAL+_driver_name;
					_init_module = __import__(module_name, fromlist=[_driver_name]);
					module = getattr(_init_module, _driver_name);

					setattr(self, _driver_name,  getattr(module, _driver_name))
				except:
					print("{bcolorsi}Warning: no driver found for {DRIVER} {bcolorsf}".format(DRIVER=_driver_name, bcolorsf='\033[0m', bcolorsi='\033[93m'))

drivers = real_drivers(pathDir=kernel.directory.hardware_drivers_dir+'real');