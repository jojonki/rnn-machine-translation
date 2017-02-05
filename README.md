# RNN Machine Translation with Chainer and CUDA.
This is a RNN encoder-decoder machine translation program with Chainer. This supports GPU execution with CUDA. 
You can disable GPU by replacing 'xp' with 'np', and remove two lines below after declaration of the model.
```
cuda.get_device(0).use() # gpu magic
model.to_gpu() # gpu magic
```

## Environment
- Python 3.5.2 |Anaconda 4.1.1 (64-bit)
- Chainer 1.20.0.1

## Preparation
You need to prepare a pair language dataset which the line numbers sentences must correspond.

## Reference
This program is almost same as below book's sample.
- [Chainerによる実践深層学習 8章](http://amzn.asia/jhtYkTq)