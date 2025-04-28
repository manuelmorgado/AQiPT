# EQM Agilent Technologies DSO1002A (Oscilloscope)

Python driver for Agilent Technologies oscilloscope over USB connection based SCPI commands.

## Installation

```bash
$ pip install git+https://git.unistra.fr/cesq/eqm-lab/lab-drivers.git
```

## Usage

```python
import Lab_drivers.DAQ.Oscilloscope.DSO1002A as DSO1002A

visa_address = 'USB0::0x0957::0x0588::CN53232361::INSTR';
fname = 'measurement_1';

oscilloscope = DSO1002A(ADDRESS=visa_address, ID=0x01);

oscilloscope.open();

oscilloscope.activateCH(1);

oscilloscope.setHORresolution(resolution={'values': 0.1, 'unit': 0x87CDA8});

oscilloscope.save(self, fname);

oscilloscope.close();

```
