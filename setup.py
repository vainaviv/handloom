from setuptools import setup, find_packages
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = [i.strip() for i in f.readlines()]
print(install_requires)

setup(
    name='TUSK',
    version='1.0.0',
    url='https://github.com/vainaviv/tusk.git',
    author='Vainavi Viswanath, Kaushik Shivakumar',
    author_email='vainaviv@berkeley.edu, kaushiks@berkeley.edu',
    description='Code for TUSK: Tracing to Untangling Semi-Planar Knots. More information found here: https://sites.google.com/view/tusk-rss/home',
    packages=find_packages(),    
    python_requires='>=3.9',
    install_requires=install_requires
)