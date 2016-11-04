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
import etra.io.feat as feat_io

class FeatIoTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_feat_io(self):
        with feat_io.FeatWriter("/tmp/test.scp", "/tmp/test.ark") as fout:
            fout.write("name", np.asarray([[1,2]], dtype=np.float32))

        with feat_io.FeatReader("/tmp/test.scp") as feat_in:
            feat = feat_in.read("name")
            np.testing.assert_equal(feat, [[1,2]])

            for name, x in feat_in.read_all():
                np.testing.assert_equal(x, [[1, 2]])

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
