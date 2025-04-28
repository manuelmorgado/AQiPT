#Atomic Quantum information Processing Tool (AQIPT - /ɪˈkwɪpt/) - Virtual devices module

# Author(s): Manuel Morgado. Universite de Strasbourg. Laboratory of Exotic Quantum Matter - CESQ
#							 Universitaet Stuttgart. 5. Physikalisches Institut - QRydDemo
# Contributor(s): 
# Created: 2022-09-01
# Last update: 2022-12-14

#libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import imageio

from PIL import Image

import time, os

import AQiPT.modules.control.AQiPTcontrol as control
# from AQiPT.AQiPTcore import digitize


def digitize(data, bitdepth, bottom, top): 
    d = np.clip(data, bottom, top);
    a = top-bottom;
    return (np.round(((d/a)-bottom)*(2**bitdepth-1))/(2**bitdepth-1)+bottom)*a

def get_current_time():
    current_time = time.time();  # get the current timestamp in seconds

    # convert the timestamp into the local time
    local_time = time.localtime(current_time);

    # extract the hour, minute, second, microsecond from the local time
    hour = local_time.tm_hour;
    minute = local_time.tm_min;
    second = local_time.tm_sec;
    microsecond = int((current_time - int(current_time)) * 1000000);
    millisecond = int(microsecond / 1000);
    
    _time = f"{hour:02d}:{minute:02d}:{second:02d}.{millisecond:03d}.{microsecond % 1000:03d}";
    print(_time)
    return _time
    
#####################################################################################################
#Virtual camera AQiPT class
#####################################################################################################
class camera:

	def __init__(self, ID=0x79999, power=False,
				 dwellTime=0.1, raisingTime=0.1, exposureTime=0,shutterTime=0.1,
				 imageFormat='.png',width=1024,height=1024,
				 temperature=18):

		self._ID = ID;
		self._type = DATA_DEV;
		self._status = ACTIVE_DEV;

		self.dwellTime = dwellTime;
		self.raisingTime = raisingTime;
		self.exposureTime = exposureTime;
		self.shutterTime = shutterTime;

		self.imageFormat = imageFormat;
		self.width = width;
		self.height = height;

		self.temperature = temperature;
		self.power = power;
		self.data = None;

	def acquireImage(self, width=None, height=None):
		get_current_time()

		if width!=None:
			self.width = width;

		if height!=None:
			self.height = height;

		self._status = BUSY_DEV;

		time.sleep(self.shutterTime)
		time.sleep(self.raisingTime)
		time.sleep(self.exposureTime)
		time.sleep(self.dwellTime)

		self.data = np.random.randint(0, 256, size=(self.width, self.height), dtype=np.uint8);
		self._status = ACTIVE_DEV;

	def getImage(self):
		get_current_time()

		return self.data

	def setTemperature(self, value):
		get_current_time()

		self.exposureTime = value;

	def turnON(self, value=True):
		get_current_time()

		self._status = ACTIVE_DEV;
		self.power = value;

	def turnOFF(self, value=False):
		get_current_time()

		self._status = INACTIVE_DEV;
		self.power = value;

	def imageFormat(self, imageformat='.png'):
		get_current_time()

		self.imageFormat = imageformat;

	def saveImage(self, fname="Default", dirpath="~/Desktop/"):
		get_current_time()

		self._status = BUSY_DEV;
		image = Image.fromarray(self.data);
		image.save(fname+self.imageFormat);

#####################################################################################################
#Virtual shutter AQiPT class
#####################################################################################################
class shutter:

	def __init__(self, ID=0x79998, dwellTime=0.1, shutterTime=0.1, configuration=False):

		self._ID = ID;
		self._type = DIGITAL_DEV;
		self._status = ACTIVE_DEV;

		self.configuration = configuration;
		self.dwellTime = dwellTime;
		self.shutterTime = shutterTime;

	def open(self):
		get_current_time()

		print('Shutter open')
		time.sleep(self.shutterTime)
		time.sleep(self.dwellTime)
		self.configuration = True;

	def close(self):
		get_current_time()

		time.sleep(self.shutterTime)
		time.sleep(self.dwellTime)
		self.configuration = False;

#####################################################################################################
#Virtual DMD AQiPT class
#####################################################################################################
class DMD:

	def __init__(self, ID=0x79997,
				 image=None, showtime=10,loadtime=0.1,
				 bitdepth=1, imagenr=1,
				 frametiming=None, sizex=1024, sizey=780,
				 triggerMode="EXT"):
		
		self._ID = ID;
		self._type = DIGITAL_DEV;
		self._status = ACTIVE_DEV;

		self.imageData = image;
		self.showTime = showtime;

		self.loadTime = loadtime;
		self.bitdepth = bitdepth;
		self.imagesNr = imagenr;
		self.frameTiming = frametiming;

		self.sizeX = sizex;
		self.sizeY = sizey;

		self.triggerMode = triggerMode;


	def Start(self):
		get_current_time()

		print('DMD open')
		self._status = BUSY_DEV;
		time.sleep(self.showTime)
		self._status = ACTIVE_DEV;

	def SeqAlloc(self, imagenr=None, bitdepth=None):
		get_current_time()

		self._status = BUSY_DEV;
		if bitdepth!=None:
			self.bitdepth = bitdepth;
		if imagenr!=None:
			self.imagesNr = imagesNr;

		time.sleep(self.loadTime)
		self._status = ACTIVE_DEV;

	def SeqPut(self, image):
		get_current_time()

		self._status = BUSY_DEV;
		self.imageData = image;
		time.sleep(self.loadTime)
		self._status = ACTIVE_DEV;

	def SetTiming(self, timing=None):
		get_current_time()

		self._status = BUSY_DEV;
		self.frameTiming=timing;
		time.sleep(self.loadTime)
		self._status = ACTIVE_DEV;

	def Run(self):
		get_current_time()

		self._status = BUSY_DEV;
		time.sleep(self.showTime)

	def Halt(self):
		get_current_time()

		self._status = ACTIVE_DEV;

	def Reset(self):
		get_current_time()

		self._status = BUSY_DEV;
		self.imageData = None;
		time.sleep(self.loadTime)
		self._status = ACTIVE_DEV;

	def Stop(self):
		get_current_time()

		self._status = ACTIVE_DEV;
		self.bitdepth = None;
		self.imagesNr = None;

#####################################################################################################
#Virtual DDS AQiPT class
#####################################################################################################
class DDS:

	def __init__(self, ID=0x79996, 
				 ttl=False, frequency=0, amplitude=0, phase=0, 
				 sampling=1e3, bitdepth=8, 
				 clock=10):

		self._ID = ID;
		self._type = ANALOG_DEV;
		self._status = ACTIVE_DEV;

		self.ttl = ttl;
		self.frequency = frequency;
		self.amplitude = amplitude;
		self.phase = phase;

		self.waveform = None;
		self.sampling = sampling;
		self.bitdepth = bitdepth;

		self.clockFrequency = clock;
		self.clockWaveform = None;

	def ON(self):
		get_current_time()

		self.ttl = True;

	def setFrequency(self, value):
		get_current_time()

		self.frequency = value;

	def setAmplitude(self, value):
		get_current_time()

		self.amplitude = value;

	def setPhase(self, value):
		get_current_time()

		self.phase = value;

	def OFF(self):
		get_current_time()

		self.ttl = False;

	def getSignal(self, acquisition_time):
		get_current_time()

		time = np.linspace(acquisition_time[0], acquisition_time[1], self.sampling);
		_waveform = self.amplitude*np.sin(2*np.pi*self.frequency*time + self.phase);
		_waveform_dig = digitize(_waveform, self.bitdepth, min(_waveform), max(_waveform));
		self.waveform = _waveform_dig;

		_waveform_CLOCK = self.amplitude*np.sin(2*np.pi*self.clockFrequency*time );
		_waveform_dig_CLOCK = digitize(_waveform_CLOCK, self.bitdepth, min(_waveform_CLOCK), max(_waveform_CLOCK));
		self.clockWaveform = _waveform_dig_CLOCK;

		return self.waveform, self.clockWaveform, time

#####################################################################################################
#Virtual MW generator AQiPT class
#####################################################################################################
class MW_generator:

	def __init__(self, ID=0x79995,
				 ttl=False, frequency=0, amplitude=0, phase=0, 
			 	 sampling=1e3, bitdepth=8, 
			 	 clock=10):

		self._ID = ID;
		self._type = ANALOG_DEV;
		self._status = ACTIVE_DEV;

		self.ttl = ttl;
		self.frequency = frequency;
		self.amplitude = amplitude;
		self.phase = phase;

		self.waveform = None;
		self.sampling = sampling;
		self.bitdepth = bitdepth;

		self.clockFrequency = clock;
		self.clockWaveform = None;

	def ON(self):
		get_current_time()

		self.ttl = True;

	def setFrequency(self, value):
		get_current_time()

		self.frequency = values;

	def setAmplitude(self, value):
		get_current_time()

		self.amplitude = values;

	def setPhase(self, value):
		get_current_time()

		self.phase = phase;

	def OFF(self):
		get_current_time()

		self.ttl = False;

	def getSignal(self, acquisition_time):
		get_current_time()

		time = np.linspace(acquisition_time[0], acquisition_time[1], self.sampling);
		_waveform = self.amplitude*np.sin(2*np.pi*self.frequency*time + self.phase);
		_waveform_dig = digitize(_waveform, self.bitdepth, min(_waveform), max(_waveform));
		self.waveform = _waveform_dig;

		_waveform_CLOCK = self.amplitude*np.sin(2*np.pi*self.clockFrequency*time );
		_waveform_dig_CLOCK = digitize(_waveform_CLOCK, self.bitdepth, min(_waveform_CLOCK), max(_waveform_CLOCK));
		self.clockWaveform = _waveform_dig_CLOCK;

		return self.waveform, self.clockWaveform, time

#####################################################################################################
#Virtual AWG AQiPT class
#####################################################################################################
class AWG:

	def __init__(self, ID=0x79994,
			 ttl=False, waveforms=None, 
		 	 sampling=1e3, bitdepth=8, nr_channels=0,
		 	 clock=10):

		self._ID = ID;
		self._type = ANALOG_DEV;
		self._status = ACTIVE_DEV;

		self.ttl = ttl;

		self.nr_channels = nr_channels;
		if waveforms==None:
			self.waveforms = [] #[None]*self.nr_channels;
		elif len(waveforms)==self.nr_channels:
			self.waveforms = waveforms;
		else:
			print("Number of channels does not match the number of given waveforms.")

		self.sampling = sampling;
		self.bitdepth = bitdepth;

		self.clockFrequency = clock;
		self.clockWaveform = None;

	def ON(self):
		get_current_time()

		print('AWG open')
		self.ttl = True;

	def OFF(self):
		get_current_time()

		self.ttl = False;

	def setWaveform(self, waveform, channel):
		get_current_time()

		# self.waveforms[channel] = waveform;
		self.waveforms.append(waveform);


	def getSignal(self, acquisition_time, ch_lst=None):
		get_current_time()

		#set all channels by default
		if ch_lst==None:
			ch_lst=np.arange(self.nr_channels);

		#preparing clock signal
		tbase_CLOCK = np.linspace(acquisition_time[0], acquisition_time[1], self.sampling);
		_waveform_CLOCK = np.sin(2*np.pi*self.clockFrequency*tbase_CLOCK );
		_waveform_dig_CLOCK = digitize(_waveform_CLOCK, self.bitdepth, min(_waveform_CLOCK), max(_waveform_CLOCK));
		self.clockWaveform = _waveform_dig_CLOCK;

		#preparing signals
		tbase_lst=[];
		for ch in ch_lst:

			if self.waveforms[ch]!=None:

				if isinstance(self.waveforms[ch], control.pulse) or isinstance(self.waveforms[ch], control.track):
					_waveform_dig = self.waveforms[ch].digiWaveform;
					_tbase = np.linspace(acquisition_time[0], acquisition_time[1], len(_waveform_dig));
					tbase_lst.append(_tbase);
				else:
					_waveform_dig = digitize(self.waveforms[ch], self.bitdepth, min(self.waveforms[ch]), max(self.waveforms[ch]));
					_tbase = np.linspace(acquisition_time[0], acquisition_time[1], len(_waveform_dig));
					tbase_lst.append(_tbase);

				self.waveforms[ch] = _waveform_dig;

			else:
				self.waveforms[ch] = tbase_CLOCK*0;

		return self.waveforms, tbase_lst, self.clockWaveform, tbase_CLOCK

	def showSignal(self, acquisition_time, ch_lst=None):
		get_current_time()

		#TODO: fix unpacking problem in for-loop, even then shows plots

		signals, tbase_signals, clk, tbase_clck = self.getSignal(acquisition_time, ch_lst)
		print(signals, tbase_signals, clk, tbase_clck)
		for signal, tbase in zip(signals, tbase_signals):
			if isinstance(signal, np.ndarray):
				plt.figure()
				plt.plot(tbase_clck, clk, color='gray', alpha=0.6)
				plt.plot(tbase, signal)
				plt.xlabel("Time")
				plt.ylabel("Signal")
				plt.show()

#####################################################################################################
#Virtual laser AQiPT class
#####################################################################################################
class laser:

	def __init__(self, ID=0x79993,
				 ttl=False, waveforms=None, 
				 sampling=1e3, bitdepth=8, nr_channels=1,
				 clock=10):

		self._ID = ID;
		self._type = DATA_DEV;
		self._status = ACTIVE_DEV;

#####################################################################################################
#Virtual oscilloscope AQiPT class
#####################################################################################################
class oscilloscope:

		def __init__(self, ID=0x79992,
					 ttl=False, waveforms=None, 
					 sampling=1e3, bitdepth=8, nr_channels=2,
					 clock=10):

			self._ID = ID;
			self._type = DATA_DEV;
			self._status = ACTIVE_DEV;

#####################################################################################################
#Virtual RTSA AQiPT class
#####################################################################################################
plt.ion()

class RealTimeSpectrogram:

	'''

		#Example
		fs = 1e9                  #sampling frequency of 500 MHz
		nfft = 5*1024               #number of FFT points
		bw = 1e6                  #resolution bandwidth of 1 MHz
		freq_range = (0, 200e6)   #frequency range to display

		#initialize the spectrogram object
		rtsa = RealTimeSpectrogram(fs=fs, 
		                           nfft=nfft, 
		                           bw=bw, 
		                           freq_range=freq_range, 
		                           cmap='bone', 
		                           power_range=(-100, 0), 
		                           tbase=300)

		#100 MHz signal for testing with frequency modulation
		frequency_0 = 100e6     #base frequency of 100 MHz
		duration = 1e-6       #signal duration of 1 ms
		t = np.arange(0, duration, 1/fs)  #time array based on sampling frequency

		a = 3000e6
		v = 50e6
		frequency = frequency_0 + v * t * 1e6 + a * (((t * 1e6)**2) / 2)

		#simulate real-time input with the 100 MHz modulated signal
		nr_frames = 300  #number of frames for the GIF

		for frame_index in range(nr_frames):
		    signal = 0.5 * np.sin(2 * np.pi * frequency[frame_index] * t)  #frequency-modulated signal
		    rtsa.update(signal, frame_index)

		plt.show()

		#generate and display the GIF
		rtsa.generate_gif("spectrogram.gif")
	'''
    def __init__(self, fs, nfft, bw, freq_range, cmap='viridis', power_range=(-100, 0), tbase=100):
        self.tbase = tbase
        self.fs = fs
        self.nfft = nfft
        self.bw = bw
        self.freq_range = freq_range
        self.cmap = cmap
        self.power_range = power_range
        self.spec_matrix = None  #initialized on first update
        self.frames = []  #frames list for GIF

        #compute nperseg dynamically based on desired resolution bandwidth (bw) RBW = frequency_sampling/segment_length
        self.nperseg = int(self.fs / self.bw)  #nperseg based on RBW
        
        #initialize the plot
        self.fig, self.ax = plt.subplots()
        self.im = None  # To be set on first update
        self.colorbar = None
        
        #create directory to store temporary frames
        os.makedirs("frames", exist_ok=True)

    def update(self, signal, frame_index):
        #set the time domain
        time_domain = self.tbase

        #calculate the spectrogram with dynamic resolution bandwidth
        f, t, spec = spectrogram(signal, 
                                 self.fs, 
                                 nperseg=self.nperseg, 
                                 noverlap=self.nperseg//2, 
                                 nfft=self.nfft)

        #power to dB
        spec_dB = 10 * np.log10(spec + 1e-10)

        #set frequency range
        fmin_idx = np.searchsorted(f, self.freq_range[0])
        fmax_idx = np.searchsorted(f, self.freq_range[1])
        spec_dB = spec_dB[fmin_idx:fmax_idx, :]
        f = f[fmin_idx:fmax_idx]  #update f to match the limited range

        #initialize spectrogram matrix (1st update)
        if self.spec_matrix is None:
            num_freq_bins = spec_dB.shape[0]
            self.spec_matrix = np.zeros((time_domain, num_freq_bins))

            #initialize imshow plot with the correct frequency range and colorbar
            self.im = self.ax.imshow(self.spec_matrix, aspect='auto', cmap=self.cmap,
                                     vmin=self.power_range[0], vmax=self.power_range[1],
                                     extent=[self.freq_range[0], self.freq_range[1], 0, time_domain])
            self.ax.set_ylabel('Time [Abs.]')
            self.ax.set_xlabel('Frequency [Hz]')
            self.colorbar = plt.colorbar(self.im, ax=self.ax, label='Power [dB]')

        #include new spectrogram line at the bottom of matrixand shift previous lines up
        self.spec_matrix = np.roll(self.spec_matrix, -1, axis=0)
        self.spec_matrix[-1, :] = spec_dB.mean(axis=1)  #averaging over time axis for a single row update

        #update the image data
        self.im.set_data(self.spec_matrix)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #save the current frame to the frames directory
        frame_path = f"frames/frame_{frame_index:04d}.png"
        self.fig.savefig(frame_path)
        self.frames.append(frame_path)

    def generate_gif(self, gif_filename="spectrogram.gif"):
        #include frames and save as .gif
        with imageio.get_writer(gif_filename, mode="I", duration=0.1) as writer:

            for frame_path in self.frames:

                image = imageio.imread(frame_path)
                writer.append_data(image)

        #remove temporal frames directory
        for frame_path in self.frames:
            os.remove(frame_path)

        os.rmdir("frames")

        #show and generate the GIF
        display(Image(filename=gif_filename))
        print(f"GIF store in: [{gif_filename}](./{gif_filename})")


#types
DATA_DEV = 0x2DC6C0;
DIGITAL_DEV = 0x2DC6C1;
ANALOG_DEV = 0x2DC6C2;

#status
BUSY_DEV = 0x2F4D60;
ACTIVE_DEV = 0x2F4D61;
INACTIVE_DEV = 0x2F4D62;

#units?
