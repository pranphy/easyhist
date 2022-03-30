#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

# author : Prakash [प्रकाश]
# date   : 2019-09-11 22:07

import setuptools

long_description = None

with open("README.md",'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="easyhist",
    version="0.2.0",
    author="Prakash Gautam",
    author_email="pranphy@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    #install_requres['numpy','matplotlib'],

    url='https://github.com/pranphy/easyhist',

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,

)
