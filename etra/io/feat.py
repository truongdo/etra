#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""
from __future__ import print_function
import sys
import struct
import numpy as np
from os.path import abspath
import random
import tempfile
from etra.io import SmartFile, smart_open


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        if i + n > len(l):
            yield l[i:i + n] + [l[-1] for x in range(i + n - len(l))]
        else:
            yield l[i:i + n]

class FeatWriter(object):
    def __init__(self, fname_scp=None, fname_ark=None):
        self.fd_scp = SmartFile(fname_scp, "w")
        if not fname_ark:
            fname_ark = fname_scp.replace(".scp", ".ark") if ".scp" in fname_scp else fname_scp + ".ark"
        self.fd_ark = SmartFile(fname_ark, "w")
        self.start = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.fd_scp.close()
        self.fd_ark.close()

    def write(self, name, feat):
        nrow, ncol = feat.shape
        feat = feat.flatten()
        byte_size = feat.nbytes + 8  # 8 is header size
        if self.fd_scp:
            self.fd_scp.write("%s %s:%d:%d\n" % (name, abspath(self.fd_ark.name), self.start, byte_size))
        if self.fd_ark:
            self.fd_ark.write(struct.pack("<2i", nrow, ncol))
            self.fd_ark.write(feat)
        self.start += byte_size

class FeatReader(object):
    def __init__(self, fname=None, data_type=np.float32):
        self.name = fname
        self.data_type = data_type
        if fname:
            self.fd_scp = SmartFile(fname, "r")
        self.file_list = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.fd_scp.close()

    def open(self):
        self.file_list = self.fd_scp.fd.readlines()
        self.load()

    def read_all(self, shuffle=False):
        if shuffle:
            random.shuffle(self.file_list)
        for line in self.file_list:
            name, finfo = line.strip().split()
            fn_feat, start, length = finfo.split(':')
            start, length = int(start), int(length)
            yield name, self.read_ra((fn_feat, start, length))

    def size(self):
        return len(self.file_list)

    def read(self, name):
        return self.read_ra((self.fname_dict[name]))

    def read_ra(self, info):
        fn_feat, start, length = info
        header_size = 8
        length = int(length)
        with smart_open(fn_feat, 'rb') as fd_ark:
            fd_ark.seek(start)
            data = fd_ark.read(length)
            nrow, ncol = struct.unpack("<2i", data[0:header_size])
            feat = np.fromstring(data[header_size:], dtype=self.data_type)
            feat.shape = (nrow, ncol)
            return feat

    def copy(self):
        a = FeatReader()
        a.name = self.name
        a.file_list = self.file_list
        a.fname_list = self.fname_list
        a.fname_dict = self.fname_dict
        return a

    def load(self, reverse=True):
        self.fname_list = []
        for line in self.file_list:
            name, finfo = line.strip().split()
            fn_feat, start, length = finfo.split(':')
            length, start = int(length), int(start)
            self.fname_list.append((name, start, length, fn_feat))
        self.fname_dict = {name: (fn_feat, start, length) for name, start, length, fn_feat in self.fname_list}

    def get_list(self, bsize=1):
        return chunks(self.fname_list, bsize)

class RawFeatReader(object):
    def __init__(self, fname=None):
        self.fd = SmartFile(fname, "r")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()
        pass

    def read_all(self, dtype=np.float32):
        while not self.fd.fd.closed:
            header_size = 12
            header = self.fd.fd.read(header_size)
            if not header:
                break
            name_len, nrow, ncol = struct.unpack("<3i", header)
            item_size = 4  # both int and float has size of 4
            name = self.fd.fd.read(name_len)
            data = self.fd.fd.read(nrow * ncol * item_size)
            feat = np.fromstring(data, dtype=np.float32)
            feat.shape = (nrow, ncol)
            yield name, feat

class RawFeatWriter(object):
    def __init__(self, fname=None):
        self.fd_ark = SmartFile(fname, "w")
        self.start = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def write(self, name, feat):
        nrow, ncol = feat.shape
        feat = feat.flatten()
        self.fd_ark.write(struct.pack("<3i", len(name), nrow, ncol))
        self.fd_ark.write(name)
        self.fd_ark.write(feat)
