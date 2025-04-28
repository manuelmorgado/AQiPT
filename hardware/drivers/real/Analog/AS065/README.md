# EQM OPX Quantum Machine (Waveform generator)

Python driver for Heidelberg DDS AS065 over ethernet and USB connection.

## Installation

```bash
$ pip install git+https://git.unistra.fr/cesq/eqm-lab/lab-drivers.git
```

## Usage

```python

from AS065 import AS065

dds_767 = AS065(ADDRESS='130.79.148.72', PORT=80, ID='0x02', data=None)
# dds_767.flash();
dds_767.add_singleFrequency(4000000);
dds_767.add_rampFrequency(start_frequency_value=5000000.000000, 
                          end_frequency_value=10000000.00000,
                          step_size=5002.366378903, 
                          step_duration=0.0001000000000000, #0.001000003200000
                          trigger=1.000000000000, 
                          amplitude_flag=1.000000000000, 
                          amplitude_scale=1.000000000000)
dds_767.sendRequest();

```
