from setuptools import setup, find_packages

setup(name='andor',
      version='0.0.1',
      description='Python driver to connect over USB to an Andor camera.',
      url='https://git.unistra.fr/cesq/eqm-lab/lab-drivers.git',
      author='M.Morgado',
      author_email='manuelmorgadov@gmail.com',
      python_requires='>=3.6',
      install_requires=[ctypes,
                        PIL,
                        time,
                        platform
      ],
      packages=find_packages())
