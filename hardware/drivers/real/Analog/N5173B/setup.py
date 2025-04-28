from setuptools import setup, find_packages

setup(name='N5173B',
      version='1.0',
      description='Python driver for Microwave generator Keysight N5173B over USB connection based SCPI commands.',
      url='url',
      author='M.Morgado',
      author_email='manuelmorgadov@gmail.com',
      python_requires= '>=3.6',
      install_requires=[
          'pyvisa'
        ],
      packages=find_packages())
