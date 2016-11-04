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
import component as com
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
import os
from chainer import serializers
import json
import ConfigParser
from chainer import optimizers

xp = np

class ATT_NMT(chainer.Chain):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_size, hsize, depth=1, \
            drop_ratio=0.):
        print('~~~~~~~ Model parameters ~~~')
        print('~ Embed size:', emb_size)
        print('~ Hidden size:', hsize)
        print('~ Layer depth:', depth)
        print('~ Drop ratio:', drop_ratio)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        self.emb_size = emb_size
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.hsize = hsize
        self.depth = depth
        self.drop_ratio = drop_ratio

        enc = com.MTEncoder(emb_size, hsize, depth, drop_ratio)
        att = com.MTAttention(hsize)
        dec = com.MTDecoder(hsize, hsize, depth, drop_ratio)
        h2o = L.Linear(hsize, trg_vocab_size)

        e2h = L.Linear(emb_size, hsize)
        src_emb = L.EmbedID(src_vocab_size, emb_size)
        trg_emb = L.EmbedID(trg_vocab_size, emb_size)

        super(ATT_NMT, self).__init__(enc=enc, att=att, dec=dec, e2h=e2h, src_emb=src_emb, trg_emb=trg_emb, h2o=h2o)
        self.S = None
        self._train = True
    
    @property
    def train(self):
        return self._train
    
    @train.setter
    def train(self, value):
        self._train = value
        self.enc.train = value
        self.dec.train = value

    def to_gpu(self):
        global xp
        xp = cuda.cupy
        super(ATT_NMT, self).to_gpu()

    def encode(self, x_batch):
        assert self.src_emb is not None
        x_list = []
        for i in xrange(x_batch.shape[1]):
            src_w = chainer.Variable(xp.asarray(x_batch[: , i], dtype=np.int32), volatile=not self.train)
            src_w = self.src_emb(src_w)
            x_list.append(src_w)

        self.S, e_l = self.enc(x_list)
        self.dec.reset(e_l)

    def reset(self):
        self.S = None
        self.enc.reset()
        self.dec.reset()
        self.att.reset()

    def decode(self, x, no_att=False, oweight=False):
        h = self.dec.update(self.e2h(x))
        weight = None
        if no_att:
            o = self.dec()
        else:
            att, weight = self.att(self.S, h, output_weight=True)
            o = self.dec(att_in=att)
        o = self.h2o(o)
        if oweight:
            return o, weight
        else:
            return o
    
    @classmethod
    def load(cls, model_dir):
        # Input files
        fn_model = os.path.join(model_dir, 'nnet.mdl')
        fn_cfg = os.path.join(model_dir, 'nnet.cfg')

        config = ConfigParser.RawConfigParser()
        config.read(fn_cfg)
        emb_size = config.getint('Structure', 'emb_size')
        src_vocab_size = config.getint('Structure', 'src_vocab_size')
        trg_vocab_size = config.getint('Structure', 'trg_vocab_size')
        hsize = config.getint('Structure', 'hsize')
        depth = config.getint('Structure', 'depth')
        drop_ratio = config.getfloat('Structure', 'drop_ratio')

        nnet = cls(src_vocab_size, trg_vocab_size, emb_size, hsize, depth, drop_ratio)
        serializers.load_npz(fn_model, nnet)

        return nnet

    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        config = ConfigParser.RawConfigParser()
        config.add_section('Structure')
        config.set('Structure', 'emb_size', str(self.emb_size))
        config.set('Structure', 'src_vocab_size', str(self.src_vocab_size))
        config.set('Structure', 'trg_vocab_size', str(self.trg_vocab_size))
        config.set('Structure', 'hsize', str(self.hsize))
        config.set('Structure', 'depth', str(self.depth))
        config.set('Structure', 'drop_ratio', str(self.drop_ratio))

        # Output files
        fn_model = os.path.join(model_dir, 'nnet.mdl')
        fn_cfg = os.path.join(model_dir, 'nnet.cfg')
        
        # Write out models
        serializers.save_npz(fn_model, self)
        
        with open(fn_cfg, 'w') as fd_cfg:
            config.write(fd_cfg)
