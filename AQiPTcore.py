#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Core

# Author: Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#                         Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): Angel Alvarez. Universidad Simon Bolivar. Quantum Information and Communication Group.
# Created: 2021-04-08
# Last update: 2022-12-14

import time, os, sys, inspect
import ast, astor

import warnings
warnings.filterwarnings('ignore')

import ctypes

import numpy as np
import random
import matplotlib.pyplot as plt

from pandas import DataFrame

import scipy.stats as stats
import scipy.signal as signal

import inspect
from pydantic_settings import BaseSettings
from typing import List, Tuple, Dict, Any, Union

from AQiPT.modules.directory import AQiPTdirectory as dirPath

from IPython.display import HTML, Javascript, display


'''
    TO-DO:
        -units class
''' 

PACKAGE= 'Atomic Quantum Information Processing Toolbox (AQiPT - /ɪˈkwɪpt/)'
PACKAGE_YEAR= 2024
PACKAGE_VERSION= '2.0.0 - beta.2'

color_lst=['lightskyblue', 'darkviolet','aqua', 'aquamarine', 
            'blue',
            'blueviolet', 'brown', 'burlywood', 'cadetblue',
            'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
            'crimson', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
            'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise',  'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
            'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
            'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
            'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
            'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
            'lightgoldenrodyellow', 'lightgray', 'lightgrey',
            'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
            'antiquewhite', 'lightslategray', 'lightslategrey',
            'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
            'linen', 'magenta', 'maroon', 'mediumaquamarine',
            'mediumblue', 'mediumorchid', 'mediumpurple',
            'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
            'mediumturquoise', 'mediumvioletred', 'midnightblue',
            'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
            'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
            'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
            'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
            'plum', 'powderblue', 'purple', 'red', 'rosybrown',
            'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
            'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
            'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
            'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
            'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
            'yellow', 'yellowgreen'];

# directory = dirPath.directory_tree({'path': 'C:\\Users\\EQM\\Desktop\\AQiPT_vNewPC_20230525\\AQiPT_vLaptop\\AQiPT\\modules\\directory\\',
#                                     'filename': 'directories_windows.json',
#                                     'printON': False})
directory = dirPath.directory_tree({'path': '/home/mmorgado/Desktop/eqm/PhD_docs/PhD_thesis_notebooks/AQiPT/configuration/directories/',
                                    'filename': 'directories_ubuntu.json',
                                    'printON': False})



#####################################################################################################
#citing AQiPT
#####################################################################################################

def cite():
    '''
        Print a citation of AQiPT.
    '''
    package_name = PACKAGE
    authors = ["M.Morgado", "S.Whitlock"]
    year = PACKAGE_YEAR
    version = PACKAGE_VERSION
    url = "https://github.com/manuelmorgado/AQiPT"
    
    citation = f"{' & '.join(authors)} ({year}). \n{package_name} \n"
    if version:
        citation += f"(version {version}) \n"
    if url:
        citation += f"Available at {url} \n"
    
    print(citation)


#####################################################################################################
#libs used in AQiPT
#####################################################################################################

def summary_libs(verbose=False):
    '''
        Generate a summary table of used in AQiPT.
    '''

    html = "<table>"
    html += "<tr><th>Software</th><th>Version</th></tr>"

    packages = [(PACKAGE, PACKAGE_VERSION)]
    try:
        packages += [("Numpy", np.__version__)]
    except:
        pass
    try:
        packages += [("SciPy", scp.__version__)]
    except:
        pass

    try:
        packages += [("plt", matplotlib.__version__)]
    except:
        pass
    try:
        packages += [("Cython", Cython.__version__)]
    except:
        pass
    try:
        packages += [("IPython", IPython.__version__)]
    except:
        pass
    try:
        packages += [("Python", sys.version)]
    except:
        pass
    try:packages += [("OS", "%s [%s]" % (os.name, sys.platform))]
    except:
        pass

    for name, version in packages:
        html += "<tr><td>%s</td><td>%s</td></tr>" % (name, version)

    if verbose:
        html += "<tr><th colspan='2'>Additional information</th></tr>"
        # Add additional information here if needed for your package

    html += "<tr><td colspan='2'>%s</td></tr>" % time.strftime(
        '%a %b %d %H:%M:%S %Y %Z')
    html += "</table>"

    return HTML(html)

#####################################################################################################
#general params class
#####################################################################################################

class general_params():

    def __init__(self, args):

        self._data = args

        #experimental atributes/params
        self.sampling = args['sampling']
        self.bitdepth = args['bitdepth']

        #dynamic atributes/params
        self.dyn_time = args['time_dyn']

    def getData(self):
        return self._data

    def timebase(self):
        return np.linspace(0, self.dyn_time, self.sampling)

#####################################################################################################
#solvers wrappers
#####################################################################################################

#function for QME scan solver
def QME_scan(H_tot, psi0, times, cops, mops, opts):
    i=0;
    for H in H_tot:
        result = qt.mesolve(H, psi0, times, cops, mops, options=opts);
        qt.qsave(result,'det-'+str(i)); #storing result
        i+=1;

#function for QME solver   
def QME_sol(H, psi0, times, cops, mops, i, opts):
    result = qt.mesolve(H, psi0, times, cops, mops, options=opts)
    qt.qsave(result,'det-'+str(i)); #storing result

#####################################################################################################
#general utils function
#####################################################################################################
   
def digitize(data, bitdepth, bottom, top):  #Finn & Shannon's code
    d = np.clip(data, bottom, top);
    a = top-bottom;
    return (np.round(((d/a)-bottom)*(2**bitdepth-1))/(2**bitdepth-1)+bottom)*a

def time2index(time, times):
    sampling_rate = len(times)
    t_i = times[0]; t_f = times[len(times)-1];
    
    if t_i<t_f:
        try:
            return int(time*sampling_rate/(t_f- t_i))
        except:
            pass
    elif t_f<t_i:
        try:
            return int(time*sampling_rate/abs(t_i-t_f))
        except:
            pass

def saveWaveform(awg_args:dict, wf_args_lst:list, waveforms_lst:list, fileformat='.csv'):
    if fileformat == '.csv':

        for i in range(len(waveforms_lst)):
            metadata = ["waveformName," + str(wf_args_lst[i]['name']), "waveformPoints," + str(awg_args['sampling']-2), "waveformType,WAVE_ANALOG_16"]
            filename = "waveforms_files/ "+ str(wf_args_lst[i]['name']) + fileformat;

            with open(filename, 'w') as fout:
                for line in metadata:
                    fout.write(line+'\n')

                # np.savetxt(filename, (waveforms_lst[i]).astype(np.uint16) , delimiter=",")
                np.savetxt(filename, waveforms_lst[i] , delimiter=",")
                print(max(waveforms_lst[i]))
        print('Saved waveforms!')

def get_size(obj, seen=None):

    size = sys.getsizeof(obj);

    if seen is None:
        seen = set();

    obj_id = id(obj);

    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle self-referential objects

    seen.add(obj_id);

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()]);
        size += sum([get_size(k, seen) for k in obj.keys()]);

    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen);

    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj]);

    return size

def get_explicit_variables():
    frame = inspect.currentframe().f_back
    vars_until_globals = {}
    while frame:
        for var_name, var_value in frame.f_locals.items():
            if var_name not in vars_until_globals and (isinstance(var_value, int) or isinstance(var_value, float) and not isinstance(var_value, bool)):
                print(var_value)
                vars_until_globals[var_name] = var_value
        if 'globals' in frame.f_locals:
            break
        frame = frame.f_back
    return vars_until_globals

def rand_hex(ndigits=3):
    # Generate a random hexadecimal number with ndigits digits

    return '0x'+''.join(random.choices('0123456789ABCDEF', k=ndigits))
    
def print_classes(modules_lst):
    for module in modules_lst:
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                print(obj)

def _print_from_id(ID):

    if isinstance(ID, str):
        _id = int(ID, 16)
        print(ctypes.cast(_id, ctypes.py_object).value)

    if isinstance(ID, int):
        print(ctypes.cast(ID, ctypes.py_object).value)

def DFsearch(df: DataFrame, identifier: str):

    if not isinstance(df, DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    for head in list(df.head(0)):

        try:

            _object = df[df[str(head)].str.contains(str(identifier))];

            if _object.empty:
                continue
            else:
                return _object
   
        except:
            pass

def remove_keywords_from_dict(D:dict, kList:list):
    newD = {}
    for key, value in D.items():
        if not any(keyword in key for keyword in kList):
            newD[key] = value
    return newD

def print_nonzero(matriz):
    filas = len(matriz)
    columnas = len(matriz[0])

    for i in range(filas):
        for j in range(columnas):
            if matriz[i][j] != 0.:
                print(f"Non-zero element: {matriz[i][j]}, Indexes: ({i}, {j})")
                return {matriz[i][j]}, ({i}, {j})

def is_dict_of_dicts(some_dict):
    for value in some_dict.values():
        if not isinstance(value, dict):
            return False
    return True

def is_list_of_lists(some_list):
    for element in some_list:
        if not isinstance(element, list):
            return False
    return True

def is_list_of_strings(some_list):
    for element in some_list:
        if not isinstance(element, str):
            return False
    return True

def check_elements_equal(lst):
    if len(lst) == 1:
        return lst[0][0] == lst[0][1]
    
    for inner_list in lst:
        if inner_list[0] != inner_list[1]:
            return False
    
    return True

def check_dict_template(target_dict, template_dict):
    for key, value in template_dict.items():
        if key not in target_dict:
            return False
    return True

def get_fft(signal, duration, sampling, plotON=True, _color='dodgerblue'):

   _fft_signal = np.fft.rfft(signal)[:-1]
   _fft_signal = _fft_signal/np.max(np.abs(_fft_signal))
   _fft_domain = np.fft.rfftfreq(sampling*duration, duration/sampling)[:sampling//2]

   if plotON:
      # plt.figure(figsize=(23,3))
      plt.stem(_fft_domain, _fft_signal, label='fft', linefmt=_color)
      plt.xlabel('Frequency [Hz]')
      plt.ylabel('fft(signal)')
      plt.legend()

   return _fft_domain, _fft_signal

# System call
os.system("")

# Class of different styles
class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def _replace_function(script_path, function_name, new_function_code):
    '''
        Replaces a function definition in a Python script with the given function code.

        INPUTS:
        -------
            script_path (str): Path to the Python script to modify.
            function_name (str): Name of the function to replace.
            new_function_code (str): New function definition as a string.

        OUTPUTS:
        --------
            str: The updated script code.

        Example:

                script_path = "slave.py"
                function_name = "func1"

                new_function_code = """
                def func1(a,b,c):
                    """
                        new plot
                    """
                    return a**2 + b-c
                """

                try:
                    #replace the function in the script
                    updated_code = replace_function(script_path, function_name, new_function_code)

                    #dump the updated script back to the file
                    with open(script_path, 'w') as file:
                        file.write(updated_code)

                    print(f"Function '{function_name}' successfully replaced! \n")

                except ValueError as e:
                    print(f"Error: {e}")
    '''

    #read the original script
    with open(script_path, 'r') as file:
        original_code = file.read()

    #parse the original script into an AST
    tree = ast.parse(original_code)



    #parse the new function code into an AST
    new_function_ast = ast.parse(new_function_code).body[0]

    #ensure the new code is a function definition
    if not isinstance(new_function_ast, ast.FunctionDef):
        raise ValueError("New function code must be a valid function definition.")

    #iterate over the tree to find the function to replace
    for i, node in enumerate(tree.body):
        
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            
            print(f"Old '{function_name}': \n")
            print(style.YELLOW + str(astor.to_source(tree.body[i])) + style.RESET)

            print(f"Replacing '{function_name}' by: \n")
            print(style.BLUE + str(astor.to_source(new_function_ast)) + style.RESET)

            #replace the function in the AST
            tree.body[i] = new_function_ast
            break
    else:
        raise ValueError(f"'{function_name}' not found in the script.")

    #convert the modified AST back to source code
    updated_code = astor.to_source(tree)

    #return the updated code
    return updated_code



#####################################################################################################
#transpiler classes
#####################################################################################################

class _SimulationConfig(BaseSettings):
    '''
        Class for configuring transpiled quantum circuit simulation using the native gates 
        implemented in the compiler.
    '''

    time_simulation: float = 5
    sampling: int = int(1e3)
    bitdepth: int = 16
    nsteps: int = 120000
    rtol: float = 1e-6
    max_steps: float = 1e-5
    store_states: bool = True
    ARGS: Dict = {'sampling': sampling,
                   'bitdepth': bitdepth,
                   'time_dyn': time_simulation};
    WAVEFORM_PARAMS: general_params = general_params(ARGS);
 
class _WaveformConfig(BaseSettings):
    '''
        Class for waveform configuration of AQiPT function, pulses and sequences used in the 
        schedule of the time-dependency for the transpiled quantum circuit simulation.
    '''

    DEFAULT_COLORS : Dict = {'detuning' : ['firebrick','lightcoral','coral','chocolate'],
                             'coupling' : ['turquoise', 'paleturquoise','darkturquoise','skyblue'],
                             'other' : ['slateblue', 'plum','violet','mediumorchid']};
    available_functions_shapes: List[str] = ["square", "gaussian"];

class _AtomicConfig(BaseSettings):
    '''
        Class for configuring the atomicModel and atomicQRegister for the simulation of the 
        transpiled quantum circuit using the native gates in the compiler.
    '''
    
    nr_levels: int = 4;
    rydberg_states: List[int] = [2,3];
    l_values: List[int] = [0,1];
    possible_transitions: Any = "All";
    c6_constant: float = -2*np.pi*1520;
    c3_constant: float = -2*np.pi*7950;
    R: float = 2;   
    layout: List[Tuple] = [(0, 0, 0), (0, R, 0), (R, 0, 0), (R, R, 0), (0, 0, R)];
    connectivity: List = ["All", []];
    
class _TranspilerConfig(BaseSettings):
    '''
        Class for transpiler configuration.
    '''

    t_start: float = 0.0;
    t_wait: float = 0.01;
    shape: str = "square";
    normal_frequency: float = 10.0;
    high_frequency: float = 100.0;

class _ExperimentConfig(BaseSettings):

    '''
        Class for experimental setup.
    '''

    hardware_specs: str;
    software_specs: str;
    atom_specs: str;

class BackendConfig(BaseSettings):
    '''
        Class gathering all configuration for simulating the circuit backend.
    '''


    # experiment_config: ExperimentConfig = _ExperimentConfig();
    simulation_config: _SimulationConfig = _SimulationConfig();
    waveform_config: _WaveformConfig = _WaveformConfig();
    atomic_config: _AtomicConfig = _AtomicConfig();
    transpiler_config: _TranspilerConfig = _TranspilerConfig();

#configs
backend_config = BackendConfig();
transpiler_config =  backend_config.transpiler_config;
simulation_config = backend_config.simulation_config;
waveform_config = backend_config.waveform_config;

#simualtions
SAMPLING = simulation_config.sampling;
T_MAX = simulation_config.time_simulation;
BITDEPTH = simulation_config.bitdepth;

#colors
coupling_color = waveform_config.DEFAULT_COLORS['coupling'];
detuning_color = waveform_config.DEFAULT_COLORS['detuning'];
other_color = waveform_config.DEFAULT_COLORS['coupling'];
