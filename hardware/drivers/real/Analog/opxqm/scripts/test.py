
import numpy as np
import opxqm as op
import matplotlib.pyplot as plt

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


Gauss_Length = 200
Constant_Length = 500
Constant_tone_Amp = 0.4
Measurement_pulse_length = 1000
simulation_duration = 1000
ip_address = '130.79.148.122'

config = {
    'version': 1,
    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
                3: {'offset': 0.0},
                4: {'offset': 0.0},
                5: {'offset': 0.0},
                6: {'offset': 0.0},
                7: {'offset': 0.0},
                8: {'offset': 0.0},
                9: {'offset': 0.0},
                10: {'offset': 0.0},
            },
            'digital_outputs': {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
            },
        },
        'con2': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
                3: {'offset': 0.0},
                4: {'offset': 0.0},
                5: {'offset': 0.0},
                6: {'offset': 0.0},
                7: {'offset': 0.0},
                8: {'offset': 0.0},
                9: {'offset': 0.0},
                10: {'offset': 0.0},
            },
            'digital_outputs': {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
            },
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
            },
        }
    },
    'elements': {

        'AOM1': {'singleInput': {'port': ('con1', 1),
                                },
                 'intermediate_frequency': 10e6,
                 'operations': {'Constant': 'Constant',
                                'Gauss': 'Gauss',
                                'Measurement_Pulse': 'Measurement_Pulse',
                               },
                 'digitalInputs': {'switch': {'port': ('con1', 1),
                                              'delay': 136,
                                              'buffer': 0,
                                              },
                                  },
                 'outputs': {'out1': ('con1', 1)},
                 'time_of_flight': 28,
                 'smearing': 0,},
        'AOM2': {
            'singleInput': {
                'port': ('con1', 2),
            },
            'intermediate_frequency': 10e6,
            'operations': {
                'Constant': 'Constant',
                'Gauss': 'Gauss',
                'Measurement_Pulse': 'Measurement_Pulse',

            },
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 1),
                    'delay': 136,
                    'buffer': 0,
                },
            },
            'outputs': {
                'out1': ('con1', 1)
            },
            'time_of_flight': 28,
            'smearing': 0,
        },
        'AOM3': {
            'singleInput': {
                'port': ('con2', 2),
            },
            'intermediate_frequency': 10e6,
            'operations': {
                'Constant': 'Constant',
                'Gauss': 'Gauss',
                'Measurement_Pulse': 'Measurement_Pulse',

            },
            'digitalInputs': {
                'switch': {
                    'port': ('con2', 1),
                    'delay': 136, #ns
                    'buffer': 0, #wider extra time x2 longest |----- PULSE -----|
                },
            },
            'outputs': {
                'out1': ('con2', 1)
            },
            'time_of_flight': 28, #min 28 ns in integer 4
            'smearing': 0, #similar to buffer due to electronic imperfections
        },
        'IQ_Element': {
            'mixInputs': {
                'I': ('con1', 3), #(controller, port)
                'Q': ('con1', 4),
                'lo_frequency': 0,
            },
            'digitalInputs': {
                'switch': {
                    'port': ('con1', 1),
                    'delay': 140,
                    'buffer': 8,
                },
            },
            'intermediate_frequency': 100e6,
            'operations': {
                'IQ_Gauss_Pulse': 'IQ_Gauss_Pulse'
        },
        },
    },
    "pulses": {
        "Constant": {
            'operation': 'control',
            'length': Constant_Length, #ns
            'waveforms': {
                'single': 'const_wf',
            },
            'digital_marker': 'ON',
        },
        'Gauss': {
            'operation': 'control',
            'length': Gauss_Length,
            'waveforms': {
                'single': 'Gauss_wf',
            },
            'digital_marker': 'OFF',
        },
        'Measurement_Pulse': {
            'operation': 'measurement',
            'length': Measurement_pulse_length,
            'waveforms': {
                'single': 'const_wf',
            },
            'integration_weights': { #time resolution of 4ns
                'integW_cos': 'integW_cos',
                'integW_sine': 'integW_sine',
            },
            'digital_marker': 'ON',
        },
        'IQ_Gauss_Pulse': {
            'operation': 'control',
            'length': Gauss_Length,
            'waveforms': {
                'I': 'Gauss_wf',
                'Q': 'zero_wf',
            },
            'digital_marker': 'ON',
        },
    },
    'waveforms': {
        'const_wf': {
            'type': 'constant',
            'sample': Constant_tone_Amp
        },
        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },
        'Gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(Constant_tone_Amp, 0, Gauss_Length/6, Gauss_Length),
        }
    },
    'digital_waveforms': { #GS/s

        'ON': {
            'samples': [(1, 0)] #(TTL_ON, duration) #0: till the end of the pulse # it is possible to concatenate tuples for sequences of TTL (1,10),(0,10),
        },

        'OFF': {
            'samples': [(0, 0)]
        },
    },
    'integration_weights': {

        'integW_cos': {
            'cosine': [1.0] * int(Measurement_pulse_length/4),
            'sine': [0.0] * int(Measurement_pulse_length/4),
        },

        'integW_sine': {
            'cosine': [0.0] * int(Measurement_pulse_length/4),
            'sine': [1.0] * int(Measurement_pulse_length/4), #remember the steps of 4ns
        },
    },}


QM = op.opxqm(ADDRESS=ip_address)
QM.configure(config)
QM.generate_program()
QM.init_manager()
QM.connect()


QM.write('play', '"Constant"', '"AOM1"')
QM.write('aling', '"AOM1"', '"AOM2"')
QM.write('play', '"Constant"', '"AOM2"')
QM.write('aling', '"AOM1"', '"AOM3"')
QM.write('wait', '200')
QM.write('play', '"Gauss"', '"AOM3"')

# QM.simulate(simulation_time=simulation_duration)
QM.execute_job()
# QM.plot_simulation()

# plt.show()