# -*- coding: utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

from data_loader import DataLoader
from mt import MachineTranslation

jvocab, jlines, evocab, elines, id2enword = DataLoader.load('ja.txt', 'en.txt')

# choose sentences you want to test
test_data = jlines[1000:1020]

def mt(model, jline):
    s = []
    model.H.reset_state()
    for i in range(len(jline)):
        wid = jvocab[jline[i]]
        x_k = model.embedx(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
    x_k = model.embedx(Variable(np.array([jvocab['<eos>']], dtype=np.int32), volatile='on'))
    h = model.H(x_k)
    wid = np.argmax(F.softmax(model.W(h)).data[0])
    s.append(id2enword[wid])
    loop = 0
    while (wid != evocab['<eos>']) and (loop <= 30):
        x_k = model.embedy(Variable(np.array([wid], dtype=np.int32), volatile='on'))
        h = model.H(x_k)
        wid = np.argmax(F.softmax(model.W(h)).data[0])
        s.append(id2enword[wid])
        loop += 1
    return s
               
demb = 100
for epoch in range(10):
    model = MachineTranslation(len(jvocab), len(evocab), demb)
    filename = "./model/mt-" + str(epoch) + ".model"
    print ("============load %s===========" % filename)
    serializers.load_npz(filename, model)
    for i in range(len(test_data) - 1):
        jln = test_data[i].split()
        jlnr = jln[::-1]
        print(i, " ".join(jln) + " -> ", end="") 
        s = mt(model, jlnr)
        print(" ".join(s))
