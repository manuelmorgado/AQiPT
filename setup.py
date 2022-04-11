from setuptools import setup, find_packages

setup(
    name = 'AQiPT',
    version = '1.0.0',
    author = "Manuel Morgado, S. Whitlock",
    author_email = "morgadovargas@unistra.fr, whitlock@unistra.fr ",
    description = ("AQiPT: Atomic Quantum information Processing Toolbox "),
    license = "BSD-3",
    url = "https://github.com/manuelmorgado/AQiPT/",
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Quantum Information",
        "License :: OSI Approved :: BSD-3 License",
    ],
    install_requires=[
        'numpy',
        'qutip',
        'scipy',
        'networkx',
        'matplotlib',
        'tqdm'     
    ],
)