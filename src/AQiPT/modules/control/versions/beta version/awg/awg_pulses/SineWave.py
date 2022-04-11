import sys 																# sys is an operating system library. We import sys so we can ensure KeysightSD1 library can be imported
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python') # This is the path to the Python SD1 library
import keysightSD1 														# Here we import the Python SD1 library. All our awg/dig commands are in here

product = 'M3201A' 	# This is the model number of the AWG
chassis = 1 		# Find your chassis number by opening SD1
slot = 7 			# CHANGE THIS NUMBER BASED ON YOUR CHASSIS CONFIGURATION. Check the slot number in SD1 SFP
channel = 1			# This is to make selecting channel 1 more readable in function calls further down

amplitude = 1.5											# (Unit: volts) This is the amplitude at which the AWG will output a signal (0.1 volts peak to peak)
frequency = 1e6											# (Unit: Hz) This is the frequency at which the AWG will output a signal (1 MHz)
waveshape = keysightSD1.SD_Waveshapes.AOU_SINUSOIDAL	# this provides the ID number that calls for a sine wave

awg = keysightSD1.SD_AOU()	# Create an object of the SD_AOU class from the keysightSD1 file and calling it "awg"

awg.openWithSlot(product, chassis, slot) # Connects AWG object with physical AWG module
awg.channelAmplitude(channel, amplitude)	# tells the "awg" object at what amplitude it will output
awg.channelFrequency(channel, frequency)	# tells the "awg" object at what frequency it will output
awg.channelWaveShape(channel, waveshape)	# tells the "awg" object what signal to output

awg.AWGstart(channel)	# tells the "awg" to start at channel 1

awg.close()	# closes the connection between the AWG object and the physical AWG module