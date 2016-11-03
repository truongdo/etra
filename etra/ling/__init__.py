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
from etra.io import smart_open
from collections import OrderedDict

def load_vocab(fname, encode=None):
    vocab = OrderedDict()

    with smart_open(fname, "r", encode=encode) as fin:
        for line in fin:
            w, idx = line.strip().split()
            idx = int(idx)
            vocab[w] = idx
    return vocab

def pad(data, val, max_len):
    for row in data:
        row_len = len(row)
        if row_len > max_len:
            print("error padding")
            exit(1)
        for i in range(max_len- row_len):
            row.append(val)
