# -*- coding: utf-8
import codecs

class DataLoader:
    @staticmethod
    def load(ja_file, en_file):
        jvocab = {}
        jlines = codecs.open(ja_file, 'r', 'utf-8').read().split('\n')
        for i in range(len(jlines)):
            lt = jlines[i].split()
            for w in lt:
                if w not in jvocab:
                    jvocab[w] = len(jvocab)
        jvocab['<eos>'] = len(jvocab)

        evocab = {}
        id2enword = {}
        elines = codecs.open(en_file, 'r', 'utf-8').read().split('\n')
        for i in range(len(elines)):
            lt = elines[i].split()
            for w in lt:
                if w not in evocab:
                    id = len(evocab)
                    evocab[w] = id
                    id2enword[id] = w

        id = len(evocab)
        evocab['<eos>'] = id
        id2enword[id] = '<eos>'

        return jvocab, jlines, evocab, elines, id2enword
