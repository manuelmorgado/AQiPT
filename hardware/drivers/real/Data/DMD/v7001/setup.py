from setuptools import setup, find_packages

setup(name='v7001',
      version='1.0',
      description='Python driver for DMD controller over USB connection.',
      url='url',
      author='M.Morgado',
      author_email='morgadovargas@unistra.fr',
      python_requires= '>=3.6',
      install_requires=[
          'ctypes',
          'platform',
          'numpy',
          'PIL',
          'matplotlib',
          'time',
          'scipy',
          'warnings',
          'hexalattice'
        ],
      packages=find_packages())
