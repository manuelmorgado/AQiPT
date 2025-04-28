#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - AQiShell

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Contributor(s): 
# Created: 2021-04-08
# Last update: 2024-12-14

import os, time, sys
import readline

aqipt_newPC = r'C:\\Users\\EQM\\Desktop'
aqipt_newPC = r'/home/mmorgado/Desktop/PhD_thesis_notebooks/'

os.chdir(aqipt_newPC);
sys.path.append(aqipt_newPC);

print('Changing directory to: ', os.getcwd())

from AQiPT import AQiPTcore as aqipt
from AQiPT.modules.control import AQiPTcontrol as control
from AQiPT.modules.emulator import AQiPTemulator as emulator
from AQiPT.modules.kernel import AQiPTkernel as kernel
#from AQiPT.modules.daq import AQiPTdaq as daq
# from AQiPT.modules.interface.APIs.API import *

#from AQiPT.hardware.drivers.real import *
from AQiPT.hardware.drivers.real.Analog.opxqm import opxqm as op


class Color:
   RESET = '\033[0m'
   GREEN = '\033[32m'
   BLUE = '\033[34m'
   RED = '\033[31m'

def print_prompt():
   print(Color.GREEN + '|ℏʰ\\ ' + Color.RESET, end='', flush=True)

#setup for readline to handle history and line editing
readline.parse_and_bind("tab: complete")  #enables tab completion (optional)
readline.parse_and_bind("set editing-mode emacs")  #default editing mode
readline.parse_and_bind("set horizontal-scroll-mode on")  #horizontal scrolling

#command history navigation
readline.parse_and_bind("Control-p: history-search-backward")  #'Up' arrow
readline.parse_and_bind("Control-n: history-search-forward")  #'Down' arrow
readline.parse_and_bind("Backward-char: backward-char")  #'Left' arrow
readline.parse_and_bind("Forward-char: forward-char")  #'Right' arrow

print('Welcome to AQiShell. \nThis is the Shell for AQiPT module interface.\nUse your pythonized commands in the prompt below and ".q" to exit')

while True:
   #print the prompt before taking input
   print_prompt()
   try:
     x = input()
     if x == '.q':
         sys.exit()

     try:
         y = eval(x)
         if y:
             print(Color.BLUE + str(y) + Color.RESET)
     except:
         exec(x)
   except Exception as e:
     print(Color.RED + "error:", e, Color.RESET)