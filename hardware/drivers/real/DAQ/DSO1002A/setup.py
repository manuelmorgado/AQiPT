from setuptools import setup, find_packages

setup(name='DSO1002A',
      version='1.0',
      description='Python driver for Agilent Technologies oscilloscope over USB connection based SCPI commands.',
      url='url',
      author='M.Morgado',
      author_email='morgadovargas@unistra.fr',
      python_requires= '>=3.6',
      install_requires=[
          'pyvisa',
          'matplotlib'
        ],
      packages=find_packages())
