#!/usr/bin/env python
# old functional:
from distutils.core import setup
import setuptools

import os


def read_requirements():
    """parses requirements from requirements.txt"""
    reqs_path = os.path.join(".", "requirements.txt")
    install_reqs = parse_requirements(reqs_path, session=PipSession())
    reqs = [str(ir.req) for ir in install_reqs]
    return reqs


setup(
    name="authornetvis",
    description="Coauthorship network visualisation (interactive)",
    author="various",
    author_email="russelljarvis@protonmail.com",
    url="https://russelljjarvis@github.com/russelljjarvis/CoauthorNetVis.git",
    packages=setuptools.find_packages(),
)
