from setuptools import setup, find_packages

setup(
    name = 'AQiPT',
    version = '2.1.0',
    author = "Manuel Morgado, S. Whitlock",
    author_email = "morgadovargas@unistra.fr, whitlock@unistra.fr ",
    description = ("AQiPT: Atomic Quantum information Processing Toolbox "),
    license = "BSD-3",
    url = "https://github.com/manuelmorgado/AQiPT/",
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    classifiers=[
        "Development Status :: 5 - Alpha",
        "Topic :: Quantum Information",
        "License :: OSI Approved :: BSD-3 License",
    ],
    install_requires=[
        'numpy',
        'inspect',
        'qutip',
        'qiskit',
        'scipy',
        'networkx',
        'matplotlib',
        'tqdm',
        'qiskit',
        'pandas',
        'multiprocessing',
        'plotly',
        'pairinteraction',
        'sched',
        'arc'   
    ],
)
