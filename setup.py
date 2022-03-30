#!/usr/bin/env python

from distutils.core import setup

setup(name='Catch-A-Waveform',
      version='1.1',
      description='Dadabots forked implementation of the paper: "Catch-A-Waveform: Learning to Generate Audio from a Single Short Example" (NeurIPS 2021) by Gal Greshler',
      author='Zack Zukowski',
      author_email='thedadabot@gmail.com',
      url='https://www.github.org/dada-bots/Catch-A-Waveform',
      packages= ['/', 'models', 'utils'],
     )