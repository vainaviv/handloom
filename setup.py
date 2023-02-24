from setuptools import setup, find_packages

setup(
    name='TUSK',
    version='1.0.0',
    url='https://github.com/vainaviv/tusk.git',
    author='Vainavi Viswanath',
    author_email='vainaviv@berkeley.edu',
    description='Code for TUSK: Tracing to Untangling Semi-Planar Knots. More information found here: https://sites.google.com/view/tusk-rss/home',
    packages=find_packages(),    
    install_requires=['numpy >= 1.23.5'],
)