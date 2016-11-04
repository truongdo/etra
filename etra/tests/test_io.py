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
import unittest
import tempfile
import os
import sys
import numpy as np
import etra.io as myio

class IoTest(unittest.TestCase):
    def setUp(self):
        self.tmp_file = tempfile.mkstemp()[1]
    
    def test_open(self):
        with myio.smart_open("ls |") as fin:
            self.assertIsInstance(fin, file)
            self.assertEqual(fin.name, "<fdopen>")
        with myio.smart_open("|cat -") as fin:
            self.assertIsInstance(fin, file)
            self.assertEqual(fin.name, "<fdopen>")
            self.assertEqual(fin.mode, "wb")
        with myio.smart_open() as fin:
            self.assertIsInstance(fin, file)
            self.assertEqual(fin.name, "<stdin>")
        with myio.smart_open(self.tmp_file) as fin:
            self.assertIsInstance(fin, file)
            self.assertEqual(fin.name, self.tmp_file)
        with myio.smart_open(self.tmp_file, "w") as fout:
            fout.write("hello\n")
        a = open(self.tmp_file).readline().strip()
        self.assertEqual(a, "hello")

    def tearDown(self):
        os.remove(self.tmp_file)
        pass

if __name__ == "__main__":
    unittest.main()
