#Atomic Quantum information Processing Tool (AQIPT) - General module

# Author: Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2021-04-15


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
#         result_lst.append(result);
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
