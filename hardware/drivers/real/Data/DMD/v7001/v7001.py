#Lab hardware drivers | V-7001 Vialux Digital Micromirror Device (DMD)

#Author(s): Manuel Morgado, Universite de Strasbourg.
#                           Laboratory of Exotic Quantum Matter | Centre Europeen de Sciences Quantiques (CESQ)
#Contributor(s):
#Created: 2023-02-28
#Last update: 2023-03-24


import sys, time

import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

from matplotlib import pyplot as plt

from PIL import Image 
from PIL.ExifTags import TAGS, GPSTAGS

from ALP4 import *


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



class v7001:

    '''
        Python class of the V-7001 Vialux Digital Micromirror Device (DMD).

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

    def __init__(self, ADDRESS, ID,
                 triggerConfig= {'mode': 'edge', 'source': 1}, triggerLevel= {'values': 0, 'unit': mV},
                 acquisitionMode= 'normal', saveMode= 'csv',
                 channelsConfig={'channel_1': {'active': False, 'vertical_res': {'values': 1, 'unit': V}, 'vertical_offset': {'values': 1, 'unit': V}},
                                 'channel_2': {'active': False, 'vertical_res': {'values': 1, 'unit': V}, 'vertical_offset': {'values': 1, 'unit': V}}},
                 horizontalConfig= {'horiztonal_res': {'values': 10, 'unit': ms}, 'horizontal_offset': {'values': 0, 'unit': s}}):

        self._ID = ID;
        self._type = DATA_DEV
        self._status = STATUS_INACTIVE;

        self.__nrChannels = 2;
        self.__channelsID = None; #ID+1 and ID+2
        self.__badwidth = {'value': 60, 'unit':MHz};
        self.__sampling = {'values': 2, 'unit':GSs};

        self.triggerConfig = triggerConfig;
        self.triggerLevel = triggerLevel;
        self.acquisitionMode = acquisitionMode;
        self.saveMode = saveMode;
        self.clock = {'value': 10, 'unit':MHz};
        self.channelsConfig = channelsConfig;
        self.channelsData = {'channel_1': None, 'channel': None};
        self.horizontalConfig = horizontalConfig;


        self._controller = None;
        self._IAC = None;
        self._IACs = [];


    def connect(self):

        self._controller = ALP4(version='4.3');
        self._controller.Initialize();

    def loadImage(self, image, bitdepth):

        self._controller.SeqAlloc(nbImg=1, bitDepth=bitdepth); #allocate memory
        self._controller.SeqPut(imgData=image); 

    def loadImages(self, images_lst, bitdepth, initPicture_idx=0, loadedPicturesNr=0):

        self._controller.SeqAlloc(nbImg=len(images_lst), bitDepth=bitdepth); #allocate memory
        self._controller.SeqPut(imgData=image, initPicture_idx=0, loadedPicturesNr=0); 


    def loadDynamicImage(self):

        pass #maybe note necessary

    def clearDisplay(self):

        self._controller.Halt();

    def reset(self, sequenceID=None):

        self._controller.FreeSeq(SequenceId= sequenceID);

    def reconnect(self):

        self._controller.DevControl(controlType=self._controller.ALP_USB_DISCONNECT_BEHAVIOUR, value=self._controller.ALP_USB_RESET);

    def disconnected(self):

        self._controller.Free()


    def setIlluminationTime(self, time):

        self._controller.SetTiming(illuminationTime = time)

    @property
    def getIlluminationTime(self):

        pass


    def setPictureTime(self, time):

        self._controller.SetTiming(pictureTime = time)

    @property    
    def getPictureTime(self):

        pass

    def sendInquiry(self):

        pass
