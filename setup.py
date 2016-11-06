"""
IndicoServer setup
For development:
    `python setup.py develop`
"""
from setuptools import setup, find_packages
import glob
import os

if __name__ == "__main__":
    setup(
        name="etra",
        packages = find_packages(), 
        install_requires=["setuptools", "nose", "coverage", "six",
            "chainer", "nvidia-ml-py"],
        version="0.1.2",
        author = 'Truong Do',
        author_email = 'truongdq54@gmail.com',
        url = 'https://github.com/truongdo/etra', # use the URL to the github repo
    )
