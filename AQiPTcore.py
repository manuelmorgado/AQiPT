#Atomic Quantum information Processing Tool (AQiPT) - Core

# Author: Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2023-06-27

import time, os, sys, inspect

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

from AQiPT.modules.directory import AQiPTdirectory as dirPath


'''
    TO-DO:
        -units class
''' 

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
directory = dirPath.directory_tree({'path': '/home/mmorgado/Desktop/AQiPT_vNewPC_20230617/AQiPT/modules/directory/',
                                    'filename': 'directories_ubuntu.json',
                                    'printON': False})



#General params class
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
    """Recursively finds size of objects by W. Jarjoui"""

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
        print(inner_list)
        if inner_list[0] != inner_list[1]:
            return False
    
    return True

def check_dict_template(target_dict, template_dict):
    for key, value in template_dict.items():
        if key not in target_dict:
            return False
    return True