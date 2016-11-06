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
        packages = ['etra'], 
        install_requires=open(os.path.join(
            os.path.dirname(__file__),
            "req.txt"), 'rb').readlines(),
        version="0.1.0",
        author = 'Truong Do',
        author_email = 'truongdq54@gmail.com',
        url = 'https://github.com/truongdo/etra', # use the URL to the github repo
    )
