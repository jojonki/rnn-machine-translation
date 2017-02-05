# -*- coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

xp = cuda.cupy

class MachineTranslation(chainer.Chain):
    def __init__(self, jv, ev, k):
        super(MachineTranslation, self).__init__(
            embedx=L.EmbedID(jv, k),
            embedy=L.EmbedID(ev, k),
            H=L.LSTM(k, k),
            W=L.Linear(k, ev)
        )
        
    def __call__(self, jvocab, jline, evocab, eline):
        self.H.reset_state()
        for i in range(len(jline)):
            wid = jvocab[jline[i]]
            x_k = self.embedx(Variable(xp.array([wid], dtype=xp.int32)))
            h = self.H(x_k)
        x_k = self.embedx(Variable(xp.array([jvocab['<eos>']], dtype=xp.int32)))
        tx = Variable(xp.array([evocab[eline[0]]], dtype=xp.int32))
        h = self.H(x_k)
        accum_loss = F.softmax_cross_entropy(self.W(h), tx)
        for i in range(len(eline)):
            wid = evocab[eline[i]]
            x_k = self.embedy(Variable(xp.array([wid], dtype=xp.int32)))
            next_wid = evocab['<eos>'] if (i == len(eline) - 1) else evocab[eline[i+1]]
            tx = Variable(xp.array([next_wid], dtype=xp.int32))
            h = self.H(x_k)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss
