#Lab hardware drivers | N5173B Keysight Microwave generator (script test)

#Author(s): Manuel Morgado, Universite de Strasbourg.
#                           Laboratory of Exotic Quantum Matter | Centre Europeen de Sciences Quantiques (CESQ)
#							Universitaet Stuttgart
# 							Physikalische Institute 5, QRydDemo 
#Contributor(s):
#Created: 2022-09-01
#Last update: 2024-07-30

import Lab_drivers.Analog.N5173B as N5173B

visa_address = 'TCPIP::130.79.148.197::inst0::INSTR';

mw_source = N5173B(visa_address);

mw_source.open();

mw_source.play()

mw_source.close();

