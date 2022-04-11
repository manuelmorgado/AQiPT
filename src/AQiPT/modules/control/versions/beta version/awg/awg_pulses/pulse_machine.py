#author: M.Morgado
#date: 09.02.2021

# pulse machine class


import time, os, sys
import numpy as np

class pulse_shape:
	
	def __init__(self, name, ID):
		self.name = name
		self.ID = ID

	def rectangular(self, time_dom, start_time, amplitude, phase):
		return

	def gaussian(self, time_dom, start_time, amplitude, phase):
		return

	def square(self, time_dom, start_time, amplitude, phase):
		return

	def sinosoidal(self, time_dom, start_time, amplitude, phase):
		return

	def triangular(self, time_dom, start_time, amplitude, phase):

class gate_player:

	def __init__(self, gate_name, nr_qubits, qubit_labels, nr_pulses):
		self.gate_name = gate_name
		self.nr_qubits = nr_qubits
		self.qubit_labels = qubit_labels
		self.nr_pulses = nr_pulses

	def Uxy(self):

	def CUxy(self):

	def pCUxy(self):

	def pCz(self):

	def CPHASE(self):

	def XY(self):



class sequence_player(self):


class QOC(self, gate, params):





pulse_1 = pulse_machine()

print(pulse_1 )