# Andor Cameras (iXon 897)

Python driver for Andor Cameras over USB connection.

## Installation

```bash
$ pip install git+https://git.unistra.fr/cesq/eqm-lab/lab-drivers.git
```

## Usage

```python

from tiqi_andor_camera.Andor import *
import numpy as np
import matplotlib.pyplot as plt
import time

ixon897 = Andor()


ixon897.GetCameraSerialNumber()
ixon897.GetDetector()
ixon897.GetTriggerLevelRange()
ixon897.GetAcquisitionTimings()
ixon897.GetTemperature()
ixon897.GetEMCCDGain()
ixon897.GetNumberADChannels()
ixon897.GetBitDepth()
ixon897.GetNumberVSSpeeds() #ixon897.VSSpeeds
ixon897.GetVSSpeed() #ixon897.vsspeed
ixon897.GetNumberHSSpeeds() #ixon897.noHSSpeeds
ixon897.GetHSSpeed() #ixon897.HSSpeeds
ixon897.GetNumberPreAmpGains() #ixon897.noGains
ixon897.GetPreAmpGain() #ixon897.preAmpGain



ixon897.SetFanMode(mode=2)
ixon897.SetAcquisitionMode(mode=3)
ixon897.SetTriggerMode(mode=0)
ixon897.SetReadMode(mode_int=4)

ixon897.SetExposureTime(time=0.01784)
ixon897.SetNumberAccumulations(number=1)
ixon897.SetNumberKinetics(numKin=1)
ixon897.SetKineticCycleTime(time=0.02460)

ixon897.SetVSSpeed(4) 
ixon897.SetVSAmplitude(0)
ixon897.SetHSSpeed(0,0)

ixon897.SetPreAmpGain(2)


ixon897.SetImage(hbin=1,vbin=1,hstart=1,hend=512,vstart=1,vend=512)

image= []
ixon897.StartAcquisition()
# ixon897.dll.SendSoftwareTrigger()
time.sleep(4)

ixon897.GetAcquiredData(image)
image= np.array(image)
image= image.reshape((ixon897.width, ixon897.height))
ixon897.ShutDown()

plt.figure(figsize=(15,15))
plt.imshow(image, cmap='gray')
plt.colorbar()

```
![test](./test.png)
