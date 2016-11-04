# Etra: a machine translation framework based on attentional neural network techniques #

## Requirements ##
Minumun requirements:

* Python 2.7.x
* Numpy
* Chainer


## Installation ##
### Install with CPUs ###
Beware that using CPUs for training is very slow, we strongly recommend to install etra on machines which have GPU cards.
```
pip install -e .
```

### Install with GPUs ###
`etra` is built on top of `chainer`, so to install `etra` on GPU machines, make sure that you have `CUDA` installed correctly, follow
the [CUDA instruction](https://github.com/pfnet/chainer#installation-with-cuda).

Then, run
```
pip install -e .
```

After installation complete, you can verify whether `etra` has been installed with GPU correctly by running,

`python -c "import chainer; print chainer.cuda.available"`
If output is "False", you have NOT configure CUDA properly.

## Examples ##
Examples on how to build an end-to-end MT system for any language pairs can be found at [egs](https://github.com/truongdo/etra/tree/master/egs/nmt)
