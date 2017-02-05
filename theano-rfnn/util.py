
"""
Code for: "Structured Receptive Fields in CNNs"
By Joern-Henrik Jacobsen, Jan van Gemert, Zhongyu Lou, Arnold W.M. Smeulders
https://arxiv.org/pdf/1605.02971v2.pdf

Based on Code by Alec Radford: https://github.com/Newmu
Network architecture inspired by "Stochastic Pooling for Regularization of Deep Convolutional Neural Networks" by Matthew D. Zeiler and Rob Fergus

Dependencies: Python, Theano, Lasagne, Pylearn2, Numpy, Pylab, Scipy, cudnn v5

Author: J.-H. Jacobsen, Jul. 2016
"""

import theano
from theano import tensor as T
import numpy as np
import scipy.ndimage.filters as filters
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
    theano.config.floatX='float32'

def init_basis_hermite(sigma,bases,extent):
    filterExtent = extent
    x = np.arange(-filterExtent, filterExtent+1, dtype=np.float)
    imSize = filterExtent*2+1
    impulse = np.zeros( (np.int(imSize), np.int(imSize)) )
    impulse[(np.int(imSize))/2,(np.int(imSize))/2] = 1.0
    nrBasis = 15
    hermiteBasis = np.empty( (np.int(nrBasis), np.int(imSize), np.int(imSize)) )
    g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
    g = g/g.sum()
    g1 = sigma * -(x/ np.square(sigma)) * g
    g2 = np.square(sigma) * ( (np.square(x)-np.power(sigma,2)) / np.power(sigma,4)) * g
    g3 = np.power(sigma,3) * -( (np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
    g4 = np.power(sigma,4) * ( ( (np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8) ) ) * g
    gauss0x = filters.convolve1d(impulse, g, axis=1)
    gauss0y = filters.convolve1d(impulse, g, axis=0)
    gauss1x = filters.convolve1d(impulse, g1, axis=1)
    gauss1y = filters.convolve1d(impulse, g1, axis=0)
    gauss2x = filters.convolve1d(impulse, g2, axis=1)
    gauss0 = filters.convolve1d(gauss0x, g, axis=0)
    hermiteBasis[0,:,:] = gauss0
    vmax = gauss0.max()
    vmin = -vmax
    #print vmax, vmin
    hermiteBasis[1,:,:] = filters.convolve1d(gauss0y, g1, axis=1) # g_x
    hermiteBasis[2,:,:] = filters.convolve1d(gauss0x, g1, axis=0) # g_y
    hermiteBasis[3,:,:] = filters.convolve1d(gauss0y, g2, axis=1) # g_xx
    hermiteBasis[4,:,:] = filters.convolve1d(gauss0x, g2, axis=0) # g_yy
    hermiteBasis[5,:,:] = filters.convolve1d(gauss1x, g1, axis=0) # g_yy
    hermiteBasis[6,:,:] = filters.convolve1d(gauss0y, g3, axis=1) # g_xxx
    hermiteBasis[7,:,:] = filters.convolve1d(gauss0x, g3, axis=0) # g_yyy
    hermiteBasis[8,:,:] = filters.convolve1d(gauss1y, g2, axis=1) # g_xxy
    hermiteBasis[9,:,:] = filters.convolve1d(gauss1x, g2, axis=0) # g_yyx
    hermiteBasis[10,:,:] = filters.convolve1d(gauss0y, g4, axis=1) # g_xxxx
    hermiteBasis[11,:,:] = filters.convolve1d(gauss0x, g4, axis=0) # g_yyyy
    hermiteBasis[12,:,:] = filters.convolve1d(gauss1y, g3, axis=1) # g_xxxy
    hermiteBasis[13,:,:] = filters.convolve1d(gauss1x, g3, axis=0) # g_yyyx
    hermiteBasis[14,:,:] = filters.convolve1d(gauss2x, g2, axis=0) # g_yyxx
    
    return theano.shared(floatX(hermiteBasis[0:bases,:,:]))

def init_bias(units):
    return theano.shared(floatX(np.zeros(units)))

def init_alphas(nrFilters,channels,nrBasis):
    return theano.shared(floatX(np.random.uniform(low=-1.0,high=1.0,size=(nrFilters,channels,nrBasis))))

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X
