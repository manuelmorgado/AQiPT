from setuptools import setup, find_packages

setup(name='opxqm',
      version='1.0',
      description='Python driver for analog+digital generator Quantum Machines OPX over QUA ethernet connection.',
      url='url',
      author='M.Morgado',
      author_email='manuelmorgadov@gmail.com',
      python_requires= '>=3.6',
      install_requires=[
          'qm-qua',
        ],
      packages=find_packages())
