import aqipt as aqipt



#Initializing the arguments for AQiPT functions
tbase = np.linspace(0,1000,1000); #time base for pulses

#square pulse - function #1 params
times_f1 = np.linspace(0, 200, 200); #time domain function
amp_f1 = 2.0;
to_f1 = 100.0; #start time
fwidth_f1 = 60.0; #width of step function
args_f1 = {'amp': amp_f1, 't_o':to_f1, 'width': fwidth_f1}; #arguments for function

#gaussian pulse - function #2 params
times_f2 = np.linspace(-200, 300, 600); #time domain function
amp_f2 = 1.3; #gaussian amplitude
center_f2 = 80.0; #gaussian center
std_f2 = 0.02; #standard deviation
args_f2 = {'g_Amp':amp_f2, 'g_center': center_f2, 'g_std':std_f2}; #arguments for functionction

#sinusoidal MW carrier pulse - function #3 params
times_f3 = np.linspace(0, 200, 200); #time domain function
args_f3 = {'Amp':1, 'freq':50, 'phase':0}; #arguments for function

#sinusoidal AWG carrier pulse - function #4 params
times_f4 = np.linspace(0, 200, 200); #time domain function
args_f4 = {'Amp':1, 'freq':5, 'phase':0}; #arguments for function
########################
###### PRODUCERS #######
########################

#Initializing the AQiPT objects
mw_src = aqipt.producer()
awg_mod = aqipt.producer()

######################
####### PULSES #######
######################

#1st pulse function
wf_pulse1 = aqipt.pulse(tbase)
wf_pulse1.addFunction(tstart, aqipt.function(times_f1, args_f1).step())

#2nd pulse function
wf_pulse2 = aqipt.pulse(tbase)
wf_pulse2.addFunction(tstart, aqipt.function(times_f2, args_f2).gaussian())

#MW carrier function
mw_carrier = aqipt.pulse(tbase)
mw_carrier.addFunction(tstart,aqipt.function(times_f3, args_f3).sinusoidal())

#AWG carrier function
awg_carrier = aqipt.pulse(tbase)
awg_carrier.addFunction(tstart, aqipt.function(times_f4, args_f4).sinusoidal())


######################
####### TRACKS #######
######################

#track 1 for AWG
track_1 = aqipt.track(tbase)
track_1.addTrack(tstart, 
	[wf_pulse1.getPulse()*awg_carrier.getPulse(), 
	wf_pulse2.getPulse()*awg_carrier.getPulse()])

#track 2 for AWG
track_2 = aqipt.track(tbase)
track_2.addTrack(tstart, 
	[wf_pulse2.getPulse()*awg_carrier.getPulse()])

#track 1 for MW
track_3 = aqipt.track(tbase)
track_3.addTrack(tstart, mw_carrier.getPulse())



########################
##### INSTRUCTIONS #####
########################

#intruction for AWG of Ramsey Sequence
instruction_Ramsey_awg = aqipt.instruction(awg_mod, [track1, track2], [ch1, ch2])
#intruction for MW of Ramsey Sequence
instruction_Ramsey_mw = aqipt.instruction(mw_src, [track3], [ch1])



####### SEQUENCE #######
#lab sequence of all producers
sequence_Ramsey = aqipt.sequence(instruction_Ramsey_mw, instruction_Ramsey_mw)

#play sequence with
sequence_Ramsey.play(mw_src, awg_mod)
