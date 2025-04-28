#Lab hardware drivers | DSO1002 Agilent Technologies Oscilloscope

#Author(s): Manuel Morgado, Universite de Strasbourg.
#                           Laboratory of Exotic Quantum Matter | Centre Europeen de Sciences Quantiques (CESQ)
#Contributor(s):
#Created: 2022-09-01
#Last update: 2022-09-01


import pyvisa as visa
from time import sleep

import matplotlib.pyplot as plt

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



def visa_device(address):
    '''
        Function for initialize the VISA instrument (driver)

        INPUT:
        ------
            adddress (str): physical address of the device (e.g., TCP/IP, GPIB)

        OUTPUT:
        -------
            (visa-instrument): VISA python object instrument
    '''

    rm = visa.ResourceManager();
    return rm.open_resource(address)

class DSO1002A:

    '''
        Python class of the Agilent technologies DSO1002A oscilloscope.

        ATTRIBUTES:
        -----------
            _ID (str) : ID of the device
            _type (str) : type of device: analog, digital and data
            _status (str) : status of the device: active, busy, inactive
            _controller (visa-object) : visa controller object

            __VISAaddress (str) : VISA address of the device
            __nrChannels (int) : number of channels of the device
            __channelsID (str) : channels ID
            __bandwidth (dict) : bandwidth of the device
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

        self.__VISAaddress = ADDRESS;
        self.__nrChannels = 2;
        self.__channelsID = None; #ID+1 and ID+2
        self.__bandwidth = {'value': 60, 'unit':MHz};
        self.__sampling = {'values': 2, 'unit':GSs};

        self.triggerConfig = triggerConfig;
        self.triggerLevel = triggerLevel;
        self.acquisitionMode = acquisitionMode;
        self.saveMode = saveMode;
        self.clock = {'value': 10, 'unit':MHz};
        self.channelsConfig = channelsConfig;
        self.channelsData = {'channel_1': None, 'channel_2': None};
        self.horizontalConfig = horizontalConfig;


        self._controller = visa_device(self.__VISAaddress);

        #pre-configuring trigger
        if self.triggerConfig['mode']=='edge':
            self._controller.write(":TRIG:EDGE:SOUR CHAN{CHANNEL}".format(CHANNEL=self.triggerConfig['source']));
            self._controller.write(":TRIG:EDGE:LEV {LEVEL}".format(LEVEL=self.triggerLevel['values']));
        
        #pre-configuring acquisiton
        if self.acquisitionMode=='normal':
            self._controller.write(":ACQ:TYPE {MODE}".format(MODE='NORM'));
        elif self.acquisitionMode=='average':
            self._controller.write(":ACQ:TYPE {MODE}".format(LEVEL='AVER'));

        #pre-configuring timebase
        self._controller.write(":TIM:MAIN:OFFS {VALUE}".format(VALUE=self.horizontalConfig['horizontal_offset']['values']));
        self._controller.write(":TIM:MAIN:SCAL {VALUE}".format(VALUE=self.horizontalConfig['horiztonal_res']['values']));


        #pre-configuring channels #TO-DO: fix flexible units
        for ch_idx in range(1, 1+self.__nrChannels):
            self._controller.write(":CHAN"+str(ch_idx)+":DISP {VALUE}".format(VALUE= 1 if self.channelsConfig['channel_'+str(ch_idx)]['active']==True else 0));
            self._controller.write(":CHAN"+str(ch_idx)+":SCAL {VALUE}".format(VALUE= self.channelsConfig['channel_'+str(ch_idx)]['vertical_res']['values']));
            self._controller.write(":CHAN"+str(ch_idx)+":OFFS {VALUE}".format(VALUE= self.channelsConfig['channel_'+str(ch_idx)]['vertical_offset']['values']));




    def open(self, channelNr:int):
        self._status = STATUS_ACTIVE;
        self.channelsConfig['channel_' + str(channelNr)]['active'] = True;


    def close(self):
        self._status = STATUS_INACTIVE;
        self.channelsConfig['channel_' + str(channelNr)] = False;
        self._controller.write(":STOP")


    def setTrigger(self, config: dict =None, level: float =None):
        
        if config!=None:
            self.triggerConfig = config;
        if level!=None:
            self.triggerLevel = level;

        if self.triggerConfig['mode']=='edge':
            self._controller.write(":TRIG:EDGE:SOUR CHAN{CHANNEL}".format(CHANNEL=self.triggerConfig['source']));
            self._controller.write(":TRIG:EDGE:LEV {LEVEL}".format(LEVEL=self.triggerLevel['values']));

    def setSavingMode(self, mode):
        self.saveMode = mode;


    def setVERTresolution(self, channelNr:int, resolution:dict):

        self.channelsConfig['channel_' + str(channelNr)]['vertical_res'] = resolution;

        for ch_idx in range(1, 1+self.__nrChannels):
            self._controller.write(":CHAN"+str(ch_idx)+":SCAL {VALUE}".format(VALUE= self.channelsConfig['channel_'+str(ch_idx)]['vertical_res']['values']));


    def setVERToffset(self, channelNr:int, offset:dict):

        self.channelsConfig['channel_' + str(channelNr)]['vertical_offset'] = offset;

        for ch_idx in range(1, 1+self.__nrChannels):
            self._controller.write(":CHAN"+str(ch_idx)+":OFFS {VALUE}".format(VALUE= self.channelsConfig['channel_'+str(ch_idx)]['vertical_offset']['values']));


    def setHORresolution(self, resolution:dict):

        self.horizontalConfig['horiztonal_res'] = resolution;
        print(self.horizontalConfig['horiztonal_res'])
        self._controller.write(":TIM:MAIN:SCAL {VALUE}".format(VALUE=str(self.horizontalConfig['horiztonal_res']['values'])));


    def setHORoffset(self, offset:dict):

        self.horizontalConfig['horizontal_offset'] = offset;
        self._controller.write(":TIM:MAIN:SCAL {VALUE}".format(VALUE=self.horizontalConfig['horiztonal_res']['values']));


    def run(self, mode="run"):
        
        if mode=="run":
            self._controller.write(":RUN");
        elif mode=="single":
            self._controller.write(":SINGLE");


    def save(self, fname):
        self.write(":SAVe:CSV:STARt '"+str(fname)+"'");



    def deactivateCH(self, channelNr:int):
        self.channelsConfig['channel_' + str(channelNr)]['active'] = False;
        for ch_idx in range(1, 1+self.__nrChannels):
            self._controller.write(":CHAN"+str(ch_idx)+":DISP {VALUE}".format(VALUE= 1 if self.channelsConfig['channel_'+str(ch_idx)]['active']==True else 0));


    def activateCH(self, channelNr:int):
        self.channelsConfig['channel_' + str(channelNr)]['active'] = True;
        for ch_idx in range(1, 1+self.__nrChannels):
            self._controller.write(":CHAN"+str(ch_idx)+":DISP {VALUE}".format(VALUE= 1 if self.channelsConfig['channel_'+str(ch_idx)]['active']==True else 0));

    def autoscale(self):
        self._controller.write("AUT");

    def plotChannel(self, channelNr):
        #TO-DO: fix acquisition and storage in self.channelsData

        if isinstance(channelNr,int):

            plt.figure(figsize=(15,4));
            plt.plot(self.channelsData[channelNr]);
            plt.xlabel('Time');
            plt.ylabel("Signal Channel {NR}".format(NR=channelNr));
            
        else:

            for ch_idx in range(1, 1+self.__nrChannels):
                plt.figure(figsize=(15,4));
                plt.plot(self.channelsData[ch_idx]);
                plt.xlabel('Time');
                plt.ylabel("Signal Channel {NR}".format(NR=ch_idx));
        
        plt.show();


