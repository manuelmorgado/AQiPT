# -*- encoding: utf-8 -*-

#Lab hardware drivers | AS065 Direct Digital Synthesizer (DDS) [Heidelberg Workshop]

#Author(s): Manuel Morgado, Universite de Strasbourg.
#                           Laboratory of Exotic Quantum Matter | Centre Europeen de Sciences Quantiques (CESQ)
# Contributor(s): S.Whitlock. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#Created: 2023-04-21
#Last update: 2023-04-21


import os
import socket
import subprocess

from time import sleep

#types
DATA_DEV = 0x2DC6C0;
DIGITAL_DEV = 0x2DC6C1;
ANALOG_DEV = 0x2DC6C2;

#status
STATUS_INACTIVE = 0x2F4D60;
STATUS_ACTIVE = 0x2F4D61;
STATUS_BUSY = 0x2F4D62;

#units
#frequency
mHz = 0x87CDA0;
Hz = 0x87CDA1;
MHz = 0x87CDA2;
GHz = 0x87CDA3;
THz = 0x87CDA4;

#time
ns = 0x87CDA5;
mus = 0x87CDA6;
ms = 0x87CDA7;
s = 0x87CDA8;

#voltage
muV = 0x87CDA9;
mV = 0x87CDAA;
V = 0x87CDAB;

#current
muA = 0x87CDAC;
mA = 0x87CDAD;
A = 0x87CDAE;

#power
dBm = 0x87CDAF;
dB = 0x87CDB0;
dBc = 0x87CDB1;

#angle
rad = 0x87CDB2;
deg = 0x87CDB3;

#sampling
Ss = 0x87CDB4;
KSs = 0x87CDB5;
MSs = 0x87CDB6;
GSs = 0x87CDB7;


class AS065:

    '''
        Python class of the AS065 DDS [Heidelberg].

        ATTRIBUTES:
        -----------
            _ID (str) : ID of the device
            _type (str) : type of device: analog, digital and data
            _status (str) : status of the device: active, busy, inactive
            _controller (visa-object) : visa controller object

            __VISAaddress (str) : VISA address of the device
            __nrChannels (int) : number of channels of the device
            __channelsID (str) : channels ID
            __badwidth (dict) : bandwidth of the device
            __sampling (dict) : sampling rate of the device

            triggerConfig (dict) : trigger configuration dictionary
            triggerLevel (dict) : trigger level value
            acquisitionMode (str) : acquisiton mode: normal, average
            saveMode (str) : storage mode waveform, csv, png
            clock (dict) : clock value
            channelsConfig (dict) : channels configuration dictionary
            channelsData (dict) : channels data values
            horizontalConfig (dict) : horizontal configuration dictionary

        METHODS:
        --------
        
            open(self, channel) : open device and set active channels

            close(self) : close device and deactivate all channels
            
            setTrigger(self, mode, level) : set full triger e.g., mode and level

            setSavingMode(self, mode) : set storage mode
            
            setVERTresolution(self, channel, resolution) : set vertical resolution of given channel

            setVERToffset(self, channel, offset) : set vertical offset of given channel

            setHORresolution(self, resolution) : set horizontal resolution of given channel

            setHORoffset(self, offset) : set horizontal offset of given channel

            run(self, channel) : run oscilloscope
            
            save(self, channel, fname, mode) : save data of given channels as fname file in given mode

            deactivateCH(self, channel) : deactivate channel from acquisition and saving

            activateCH(self, channel) : activate channel from acquisition and saving

            autoscale(self) : autoscale function

            plotChannel(self, channelNr) : plot list of channels or given channels

    '''

    def __init__(self, ADDRESS, PORT, ID='0x0', data=None,
                 arduino_model="arduino:avr:uno", board_port="/dev/ttyACM0",
                 program_path="/path/to/myprogram.ino", arduinoIDE_path="/path/to/arduino", avrdude_path="/path/to/avrdude"):

        self._ID = ID;
        self._type = DATA_DEV
        self._status = STATUS_INACTIVE;

        self.__address = ADDRESS; #Ex. HOST = '130.79.148.73'  
        self.__port = PORT; #Ex. PORT = 80  
        self.__nrChannels = 1;
        self.__channelsID = None; #ID+1 and ID+2
        
        self.clock = {'value': 10, 'unit':MHz};


        self._controller = None;
        self._boardModel = arduino_model;
        self._boardPort = board_port;

        self._arduinoPath = arduinoIDE_path;
        self._avrdudePath = avrdude_path;
        
        self._inoFile = program_path;

        self.data = None;


    def flash(self):
        
        command = [self._avrdudePath,
                   "-C" + self._arduinoPath + "/hardware/tools/avr/etc/avrdude.conf",
                   "-v",
                   "-patmega328p",
                   "-carduino",
                   "-P" + self._boardPort,
                   "-b115200",
                   "-D",
                   "-Uflash:w:" + self._inoFile + ":i"
                   ]; #build the avrdude command

        result = subprocess.run(command, capture_output=True); #execute the avrdude command
        print(result.stdout.decode()); #print the output of the avrdude command

    def reset(self):
        pass

    def turnOFF(self):
        pass          

    def add_singleFrequency(self, frequency_value, trigger=1, amplitude_flag=1, amplitude_scale=0.5, phase=0):
        self.data = '{:<013.3f}'.format(0)+'	'+'{:<013.3f}'.format(int(frequency_value))+'	'+'{:<013.3f}'.format(0)+'	'+'{:<013.3f}'.format(trigger)+'	'+'{:<013.3f}'.format(amplitude_flag)+'	'+'{:<013.3f}'.format(amplitude_scale)+'	'+'{:<013.3f}'.format(phase)+'	'+'{:<013.3f}'.format(0)+'	'+'{:<013.3f}'.format(0)+'	'+'{:<013.3f}'.format(0)+'	'+'\n'

    def add_rampFrequency(self, start_frequency_value, end_frequency_value, step_size=1, step_duration=1, trigger=1, amplitude_flag=1, amplitude_scale=0.7):
        self.data = '{:<013.3f}'.format(1)+'	'+'{:<013.3f}'.format(int(start_frequency_value))+'	'+'{:<013.3f}'.format(int(end_frequency_value))+'	'+'{:<013.3f}'.format(float(step_size))+'	'+'{:<013.3f}'.format(float(step_duration))+'	'+'{:<013.3f}'.format(trigger)+'	'+'{:<013.3f}'.format(amplitude_flag)+'	'+'{:<013.3f}'.format(amplitude_scale)+'   '+'{:<013.3f}'.format(1)+'	'+'{:<013.3f}'.format(0)+'	'+'\n'


    def sendRequest(self):

        #connect to the Arduino EthernetClient and send the data
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.__address, self.__port))
            s.sendall(bytes(self.data, encoding='utf-8'))
            # _rcv_msg = s.recv(1024);
            s.close();
            # break





