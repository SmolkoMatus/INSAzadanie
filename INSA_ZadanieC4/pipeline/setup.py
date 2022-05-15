import os
from pathlib import Path
from setuptools import find_packages, setup

NAME = 'model'
DESCRIPTION = ''
URL = 'https://github.com/SmolkoMatus/INSAzadanie/tree/main/INSA_ZadanieC4/pipeline/model'
EMAIL = 'matus.smolko@student.tuke.sk'
AUTHOR = 'Matus Smolko'
REQUIRES_PYTHON = '>=3.8.0'

def load_requirements(fname='requirements.txt'):
  with open(fname) as fd:
    return fd.read().splitlines()

ROOT = Path(__file__).resolve().parent
with open(ROOT / NAME / 'VERSION') as f:
  __version__ = f.read()

setup(
  name=NAME,
  version=__version__,
  description=DESCRIPTION,
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  url=URL,
  packages=find_packages(exclude=('*.log', 'tests')),
  package_data={'model': ['VERSION']},
  include_package_data=True,
  install_requires=load_requirements(),
  license='MIT',
  classifiers=['Programming Language :: Python :: 3.8']
)
