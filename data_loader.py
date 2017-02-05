# -*- coding: utf-8
import codecs

class DataLoader:
    @staticmethod
    def load(src_file, dst_file):
        src_vocab = {}
        src_lines = codecs.open(src_file, 'r', 'utf-8').read().split('\n')
        for i in range(len(src_lines)):
            lt = src_lines[i].split()
            for w in lt:
                if w not in src_vocab:
                    src_vocab[w] = len(src_vocab)
        src_vocab['<eos>'] = len(src_vocab)

        dst_vocab = {}
        id2word = {}
        dst_lines = codecs.open(dst_file, 'r', 'utf-8').read().split('\n')
        for i in range(len(dst_lines)):
            lt = dst_lines[i].split()
            for w in lt:
                if w not in dst_vocab:
                    id = len(dst_vocab)
                    dst_vocab[w] = id
                    id2word[id] = w

        id = len(dst_vocab)
        dst_vocab['<eos>'] = id
        id2word[id] = '<eos>'

        return src_vocab, src_lines, dst_vocab, dst_lines, id2word
