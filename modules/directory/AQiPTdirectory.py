#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Directory module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2021-04-08
# Last update: 2022-02-07

#libs
import os, json 

def setCurrentDir(pathDir):

	'''
		Example:
		import os

		pathDir = "/home/user/Desktop/"; #path directory laptop with current date
		setCurrentDir(pathDir)
		os.getcwd()
	'''
	os.chdir(pathDir)


def loadJSON(path=None, filename=None, printON=False):

	'''
		Load JSON files from path and file name or just load file in current path 
		directory.
	'''

	if path==None:
		with open('C:\\Users\\AQiPT\\modules\\directory\\directories.json', 'r') as json_file:
			directories = json.load(json_file);
			if printON==True:
				print(json.dumps(directories, indent=1, sort_keys=True));
			return directories
		pass

	elif isinstance(path, str):

		with open(path+filename, 'r') as json_file:
			directories = json.load(json_file);
			if printON==True:
				print(json.dumps(directories, indent=1, sort_keys=True));
			return directories
		pass

class directory_tree:

	def __init__(self, args):

		self._spec_json = loadJSON(args['path'], args['filename'], args['printON']);

		self.hardware_specs_dir = self._spec_json['hardware specifications'];
		self.hardware_drivers_dir = self._spec_json['hardware drivers'];
		self.config_dir = self._spec_json['configuration'];
		self.data_depository_dir = self._spec_json['data depository'];
		self.compiler_dir = self._spec_json['compiler'];
		self.ctrl_mod_dir = self._spec_json['control module'];
		self.daq_mod_dir = self._spec_json['daq module'];
		self.analysis_dir = self._spec_json['analysis module'];
		self.logger_mod_dir = self._spec_json['datalogger module'];
		self.emulator_mod_dir = self._spec_json['emulator module'];
		self.GUI_mod_dir = self._spec_json['interface module'];
		self.kernel_mod_dir = self._spec_json['kernel module'];
		self.notebooks_dir = self._spec_json['notebooks'];
		self.versions_dir = self._spec_json['versions'];

		self.__dictDir = {"+ Hardware specifications": self.hardware_specs_dir,
						  "+ Hardware drivers": self.hardware_drivers_dir,
						  "+ Configuration": self.config_dir,
						  "+ Data depository": self.data_depository_dir,
						  "+ Compiler": self.compiler_dir,
						  "+ Module-control": self.ctrl_mod_dir, 
						  "+ Module-DAQ": self.daq_mod_dir,
						  "+ Module-analysis": self.analysis_dir,
						  "+ Module-datalogger": self.logger_mod_dir,
						  "+ Module-emulator": self.emulator_mod_dir,
						  "+ Module-interface": self.GUI_mod_dir,
						  "+ Module-kernel": self.kernel_mod_dir,
						  "+ Notebooks": self.notebooks_dir,
						  "+ Versions": self.versions_dir
						  };

	def tree_dictionary(self):
		print('Directory tree AQiPT: \\n----------------------')
		for k, v in self.__dictDir.items():
			print(f'{k:<4}:{v} \\n')

