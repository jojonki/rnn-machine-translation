# -*- coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as f
import chainer.links as l

from data_loader import DataLoader
from mt import MachineTranslation

jvocab, jlines, evocab, elines, id2enword = DataLoader.load('ja.txt', 'en.txt')

# restrict dataset size
# jlines = jlines[0:1000]
# elines = elines[0:1000]

demb = 100 # word distributed represantation param
model = MachineTranslation(len(jvocab), len(evocab), demb)
cuda.get_device(0).use() # gpu magic
model.to_gpu() # gpu magic
optimizer = optimizers.Adam()
optimizer.setup(model)
for epoch in range(50):
    print("ecpoch: %d..." % epoch)
    for i in range(len(jlines) - 1):
        jln = jlines[i].split()
        jlnr = jln[::-1] # inverse input improve accuracy
        eln = elines[i].split()
        model.H.reset_state()
        model.cleargrads()
        loss = model(jvocab, jlnr, evocab, eln)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    outfile = "./model/mt-" + str(epoch) + ".model"
    serializers.save_npz(outfile, model)
        