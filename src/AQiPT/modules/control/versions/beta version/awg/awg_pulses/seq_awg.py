import time, os, sys
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from IPython.core.display import HTML
# with open( './custom.css', 'r' ) as f: style = f.read()
# HTML( style )
display(HTML("<style>.container { width:90% !important; }</style>"))

def Rpulses(pwidth=1, shift=0, initime=-10, endtime=10, resolution=1000):
    
    '''
    Rpulses() : function that generates Rectangular Pulses. This function does have the chance to change the
    width of the pulse directly (i.e change pwidth).
    
    INPUT
    pwidth       : width of the pulse, normally the origin is left hand side of the pulse where changes from 0->1
    shift        : offset of the pulse, helpful for the combination of pulses
    initime      : initial value of the time scale ti
    endtime      : final value of the time scale tf
    resolution   : resolution of the pulses, if resolution is too low, then the rectangular pulse becomes trapezoid
    
    OUTPUT
    time    : domain values of the pulse function 
    pulse   : pulse function
    '''
    domain = np.linspace(initime, endtime, resolution, endpoint=True) #domain of the whole function
    HeavisidePos = np.heaviside((domain - shift), 0.5) #mid point set to intermediate value
    HeavisideNeg = np.heaviside(-(domain - shift) + pwidth, 0.5)
    
    pulse = (HeavisidePos + HeavisideNeg - 1) #the (-1) is an offset due to the sum 
    
    '''
    #Example of addition of 2 rectanglar pulse 
    dom, fpulse = Rpulses(pwidth=1, shift=0, initime=-10, endtime=10, resolution=1000)
    dom, fpulse2 = Rpulses(pwidth=2, shift=5, initime=-10, endtime=10, resolution=1000)
    plt.plot(dom, fpulse+fpulse2)
    '''
    return domain, pulse

def buildSequence(TIMEseq, amplst, initPlst, widthlst):
    '''
    buildSequence() : function that generates the pulse sequence of pulses for the different couplings and
    detunings of the Hamiltonians amplitudes.
    
    INPUT
    amplst : amplitude list 
        e.g [Omega1, Omega2]
        
    initPlst : initial times list 
        e.g for Omega1 -> [0, 3, 100]; for Omega2 -> [5, 20] then [[0, 3, 100], [5, 20]]
    
    widthlst : pulse width list
        e.g for Omega1 -> [1, 1, 2]; for Omega2 -> [1, 1, 2] then [[1, 1, 2], [1, 1]]
        
    OUTPUT
    Sequence: list of pulses for each amplitude
    '''
    
    ampPulselst = []
    count=0;
    for amp in amplst:
        pulselst = [];
        for ti,wi in zip(initPlst[count], widthlst[count]):
            dom, fpulse = Rpulses(pwidth = wi, shift = ti, initime = TIMEseq[0], endtime = TIMEseq[len(TIMEseq)-1], resolution=1000)
            pulselst.append(fpulse)
        ampPulse = amp*sum(pulselst)
        ampPulselst.append(ampPulse)
        count+=1;
    
#     for i in ampPulselst:
#         plt.plot(TIMEseq, i)
        
    '''
    #example
    dom = np.linspace(0, 30, 1000, endpoint=True)
    buildSequence([10, 2],  [[0, 3, 20], [5, 20]],  [[1, 1, 2], [1, 1]], dom)
    '''
    return ampPulselst

initime = 0.0; endtime = 1000.0; 
res = 1000;
times = np.linspace(initime, endtime, res, endpoint=True)

#building sequences for the Hamiltonians
ampd1 = 1; ampd2 = 1; ampc1 = 1; ampc2 = 1; ampc3 = 1; ampc4 = 1;
shiftd1 = 0+600; shiftd2 = 100-2; shiftc1 = 0; shiftc2 = 0+2*np.pi; shiftc3 = 10; shiftc4 = 10+2*np.pi;
widthd1 = 3*np.pi+2; widthd2 = 3*np.pi+2; widthc1 = np.pi; widthc2 = np.pi; widthc3 = np.pi; widthc4 = np.pi;

sargs = {'ampd1': ampd1,'ampd2': ampd2,'ampc1': ampc1,'ampc2': ampc2,'ampc3': ampc3,'ampc4': ampc4,
         'shiftd1': shiftd1,'shiftd2': shiftd2,'shiftc1': shiftc1,'shiftc2': shiftc2,'shiftc3': shiftc3,'shiftc4': shiftc4,
        'widthd1': widthd1,'widthd2': widthd2,'widthc1': widthc1,'widthc2': widthc2,'widthc3': widthc3,'widthc4': widthc4}

#drift Hamiltonians
pHdADE = buildSequence(times, [sargs['ampd1']],  [[sargs['shiftd1']]], [[sargs['widthd1']]])

pHdBCE = buildSequence(times, [sargs['ampd2']],  [[sargs['shiftd2']]], [[sargs['widthd2']]])

#control Hamiltonians
# pHcAD = buildSequence(times, [sargs['ampc1']],  [[sargs['shiftc1']]], [[sargs['widthc1']]])

# pHcDE = buildSequence(times, [sargs['ampc2']],  [[sargs['shiftc2']]], [[sargs['widthc2']]])

# pHcEC = buildSequence(times, [sargs['ampc3']],  [[sargs['shiftc3']]], [[sargs['widthc3']]])

# pHcCB = buildSequence(times, [sargs['ampc4']],  [[sargs['shiftc4']]], [[sargs['widthc4']]])

Omega_rf = 1e6
t = np.linspace(0, 2*np.pi, 1000)
for i in range(len(times)):
    seqP=pHdADE[0]+pHdBCE[0]
#     seq[i]=(seqP[i])*np.sin(Omega_rf*times[i])
    seq=(seqP)*np.sin(Omega_rf*t)
    seq1= np.sin(Omega_rf*t)

plt.plot(seq)

