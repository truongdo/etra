#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 truong-d <truong-d@ahclab40>
#
# Distributed under terms of the MIT license.

"""

"""
from __future__ import print_function
import chainer
import chainer.links as L
import chainer.functions as F
import base

class MTEncoder(base.BaseEncoder):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(MTEncoder, self).__init__(isize, osize, depth, drop_ratio)
        self._train = True
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, value):
        self._train = value
        super(MTEncoder, self).train(value)

    def __call__(self, x_list):
        batch_size = x_list[0].data.shape[0]
        e_list = super(MTEncoder, self).encode(x_list)
        S = F.reshape(F.concat(e_list, axis=1), (batch_size, len(x_list), -1))
        return S, e_list[-1]

    def reset(self):
        super(MTEncoder, self).reset()


class MTDecoder(base.BaseDecoder):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(MTDecoder, self).__init__(isize, isize, depth, drop_ratio)
        self.add_link('wc', L.Linear(isize * 2, isize))
        self.add_link('wo', L.Linear(isize, osize))
        self._train = True
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, value):
        self._train = value
        super(MTDecoder, self).train(value)

    def update(self, x):
        self.h = super(MTDecoder, self).__call__(x)
        return self.h

    def __call__(self, att_in=None, output_hidden=False):
        if att_in is not None:
            h = F.tanh(self.wc(F.concat((self.h, att_in), axis=1)))
        else:
            h = self.h
        self.o = self.wo(h)
        if output_hidden:
            return self.o, h
        else:
            return self.o

    def reset(self, e_l=None):
        super(MTDecoder, self).reset()
        if e_l is not None:
           self.start_with(e_l)

    def start_with(self, x):
        o = super(MTDecoder, self).__call__(x)


class MTAttention(base.BaseAttention):
    def __init__(self, osize):
        super(MTAttention, self).__init__(osize)

    def __call__(self, enc_mat, h, output_weight=False):
        return super(MTAttention, self).__call__(enc_mat, h, output_weight=output_weight)

