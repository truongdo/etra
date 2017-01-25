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
import tempfile
import numpy as np
import os
from os.path import join
import subprocess
import etra.utils.bleu as bleu

def get_bleu(data_dir, name="text"):
    with open(join(data_dir, name)) as fn_mt:
        hyp = []
        ref = []
        wrong_format_sen = 0
        total = 0
        for line in fn_mt:
            total += 1
            if "|||" not in line:
                print("Wrong format", line)
                continue

            one_pred, one_ref = line.strip().split("|||")
            if not one_pred:
                wrong_format_sen += 1
                continue

            one_pred = one_pred.split()
            one_ref = one_ref.split()
            hyp.append(one_pred)
            ref.append(one_ref)

        assert len(ref) == len(hyp)
        stats = [0 for i in xrange(10)]

        for (r,h) in zip(ref, hyp):
            stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(h,r))]
        return len(ref), total, bleu.bleu(stats)

if __name__ == "__main__":
    data_dir = sys.argv[1]

    n_sen_ok, total, bleu_mt_task = get_bleu(data_dir, name="text")
    print("Bleu score for MT task: %.4f (%d/%d sentences)" % (bleu_mt_task, n_sen_ok, total))
