from setuptools import setup
from setuptools import find_packages

setup(
    name='pandautils',
    version='0.0.1',
    description='Library with convenient tricks to import and handle .root files in pandas format.',
    author='Michela Paganini',
    author_email='michela.paganini@yale.edu',
    url='https://github.com/mickypaganini/YaleATLAS',
    install_requires=['pandas', 'numpy', 'root_numpy'],
    packages=find_packages()
)
