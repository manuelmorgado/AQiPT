# EQM OPX Quantum Machine (Waveform generator)

Python driver for analog+digital generator Quantum Machines OPX over QUA ethernet connection.

## Installation

```bash
$ pip install git+https://git.unistra.fr/cesq/eqm-lab/lab-drivers.git
```

## Usage

```python
import Lab_drivers.Analog.generators.opxqm as qm


qm_config = {'specs': 'everythinghere'}


master = qm.opx();


master.configure(qm_config); #master.reconfigure(new_config);


master.add_param(name, value, unit, label):

master.update_param(name, value, unit, label);


master.generate_program(name);

master.write(command, *args);


master.init_manager(ip_address);

master.connect();


master.execute_job();

master.simulate();

master.get_job_results();


master.save_results('job'); #'simulation' otherwise


master.disconnect();




```
