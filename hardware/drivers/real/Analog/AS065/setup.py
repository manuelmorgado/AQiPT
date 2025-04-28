from setuptools import setup, find_packages

setup(name='AS065',
      version='1.0',
      description='Python driver for DDS Heidelberg via ethernet connection.',
      url='url',
      author='M.Morgado',
      author_email='morgadovargas@unistra.fr',
      python_requires= '>=3.6',
      install_requires=[
          'socket',
          'subprocess',
        ],
      packages=find_packages())
