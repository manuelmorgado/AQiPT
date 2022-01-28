#Atomic Quantum information Processing Tool (AQIPT) - Directory module

# Author: Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
# Created: 2021-04-08
# Last update: 2021-04-15

#libs
import os 

def setCurrentDir(pathDir):

	'''
	Example:
		import os
		
		pathDir = "/home/manuel/Downloads/"; #path directory laptop with current date
		setCurrentDir(pathDir)
		os.getcwd()
	'''
    os.chdir(pathDir)
