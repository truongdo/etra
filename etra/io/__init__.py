#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
import sys
import codecs
import subprocess
from contextlib import contextmanager

def read_from_pipe(program):
    cmd = program
    proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    return proc.stdout

def write_to_pipe(program):
    proc = subprocess.Popen([program], stdin=subprocess.PIPE, shell=True)
    return proc.stdin, proc

@contextmanager
def smart_open(fname=None, mode="r", encode=None):
    is_pipe = False
    is_pipe_write = False
    proc = None
    fd = None
    try:
        fd, proc, is_pipe_write, is_pipe = get_file_descriptor(fname, mode, encode)
        yield fd
    finally:
        if fd:
            if is_pipe_write and proc:
                proc.communicate()
            if not is_pipe:
                fd.close()

def get_file_descriptor(fname, mode, encode=None):
    fd = None
    is_pipe_write = False
    is_pipe = False
    proc = None
    if (not fname) or (fname == "-"):
        if "r" in mode:
            fd = sys.stdin
        else:
            fd = sys.stdout
    elif "|" == fname[-1]:
        is_pipe = True
        fname = fname[:-1]
        fd = read_from_pipe(fname)
    elif "|" == fname[0]:
        fname = fname[1:]
        is_pipe_write = True
        fd, proc = write_to_pipe(fname)
    else:
        if encode:
            fd = codecs.open(fname, mode, encode)
        else:
            fd = open(fname, mode)
    return fd, proc, is_pipe_write, is_pipe


class SmartFile(object):
    def __init__(self, fname=None, mode="r"):
        self.fd, self.proc, self.is_pipe_write, self.is_pipe = get_file_descriptor(fname, mode)
        self.name = fname

    def close(self):
        if self.fd:
            if self.is_pipe_write and self.proc:
                self.proc.communicate()
            if not self.is_pipe:
                self.fd.close()

    def write(self, data):
        self.fd.write(data)
