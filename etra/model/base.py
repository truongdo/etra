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
from chainer import ChainList

class StackLSTM(ChainList):
    def __init__(self, I, O, depth=1, drop_ratio=0):
        chain_list = []
        for i in range(depth):
            start = I if i == 0 else O
            chain_list.append(L.LSTM(start, O))
        self._drop_ratio = drop_ratio
        super(StackLSTM, self).__init__(*chain_list)
        self._train = True
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, value):
        self._train = value

    def reset_state(self):
        for lstm in self:
            lstm.reset_state()

    def __call__(self, inp, is_train=False):
        ret = None
        for i, hidden in enumerate(self):
            h = inp if i == 0 else ret
            ret = hidden(h)
        if self.train and self._drop_ratio:
            return F.dropout(ret, train=is_train, ratio=self._drop_ratio)
        else:
            return ret

    def get_state(self):
        ret = []
        for lstm in self:
            ret.append((lstm.c, lstm.h))
        return ret

    @property
    def h(self):
        return self.get_state()[-1][-1]

    @h.setter
    def h(self, h):
        for lstm_self in self:
            lstm_self.h = h

    def set_state(self, state):
        for lstm_self, lstm_in in zip(self, state):
            lstm_self.c, lstm_self.h = lstm_in


class BaseAttention(chainer.Chain):
    def __init__(self, osize):
        super(BaseAttention, self).__init__()
        self.osize = osize

    def __call__(self, enc_mat, h, output_weight=False):
        weights = F.softmax(F.batch_matmul(enc_mat, h))
        att = F.reshape(F.batch_matmul(weights, enc_mat, transa=True), (h.data.shape[0], self.osize))
        if not output_weight:
            return att
        else:
            return att, weights

    def reset(self):
        pass


class BaseDecoder(chainer.Chain):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(BaseDecoder, self).__init__(
                dec=StackLSTM(isize, osize, depth, drop_ratio)
             )
        self._train = True
    
    def train(self, value):
        self._train = value
        self.dec.train = value

    def reset(self):
        self.dec.reset_state()

    def __call__(self, x):
        return self.dec(x)


class BaseEncoder(chainer.Chain):
    def __init__(self, isize, osize, depth=1, drop_ratio=0.):
        super(BaseEncoder, self).__init__(
                encF=StackLSTM(isize, osize, depth, drop_ratio),
                encB=StackLSTM(isize, osize, depth, drop_ratio),
                aw = L.Linear(osize * 2, osize)
                )
        self._train = True
    
    def train(self, value):
        self._train = value
        self.encF.train = value
        self.encB.train = value

    def _encode_forward(self, x):
        return self.encF(x)

    def _encode_backward(self, x):
        return self.encB(x)

    def encode(self, x_list):
        self.reset()
        fx_list = []  # forward encoded list
        bx_list = []  # backward encoded list
        for idx, x in enumerate(x_list):
            fx = self._encode_forward(x)   # encode forward
            bx = self._encode_backward(x_list[-idx - 1])  # encode backward
            fx_list.append(fx)
            bx_list.append(bx)

        e_list = []
        for idx in xrange(len(x_list)):
            fx_i = fx_list[idx]
            bx_i = bx_list[-idx - 1]
            e_i = self.aw(F.concat((fx_i, bx_i), axis=1))
            e_list.append(e_i)
        return e_list

    def reset(self):
        self.encF.reset_state()
        self.encB.reset_state()
