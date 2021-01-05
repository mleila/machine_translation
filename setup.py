import os
from distutils.core import setup

packages = ['translator']
scripts = [os.path.join('bin', f) for f in os.listdir('./bin')]

setup(name='machine_translator',
      version='0.1',
      packages=packages,
      scripts=scripts
      )
