from setuptools import setup, find_packages
#from ibapi import get_version_string

import sys

if sys.version_info < (3,1):
    sys.exit("Only Python 3.1 and greater is supported")

setup(
    name='valkyrie',
    version="0.0.1",
    packages=find_packages(),
    url='',
    license='',
    author='bb',
    author_email='huanghe03@gmail.com',
    description='Project Valkyrie'
)
