import Lab_drivers.DAQ.Oscilloscope.DSO1002A as DSO1002A

visa_address = 'USB0::0x0957::0x0588::CN53232361::INSTR';

oscilloscope_1 = DSO1002A(visa_address);

oscilloscope.open();

oscilloscope.acquire(format='.csv')

oscilloscope.close();
