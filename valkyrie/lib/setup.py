from setuptools import setup
#from ibapi import get_version_string

import sys

if sys.version_info < (3,1):
    sys.exit("Only Python 3.1 and greater is supported")

setup(
    name='valkyrie',
    version="1.0.0",
    packages=['valkyrie'],
    url='',
    license='',
    author='bb',
    author_email='huanghe03@gmail.com',
    description='Project Valkyrie'
)
