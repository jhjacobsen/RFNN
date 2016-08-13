
"""
Code for: "Structured Receptive Fields in CNNs"
By Joern-Henrik Jacobsen, Jan van Gemert, Zhongyu Lou, Arnold W.M. Smeulders
https://arxiv.org/pdf/1605.02971v2.pdf

Based on Code by Alec Radford: https://github.com/Newmu
Network architecture inspired by "Stochastic Pooling for Regularization of Deep Convolutional Neural Networks" by Matthew D. Zeiler and Rob Fergus

Dependencies: Python, Theano, Lasagne, Pylearn2, Numpy, Pylab, Scipy, cudnn v5

Author: J.-H. Jacobsen, Jul. 2016
"""

import argparse
import numpy as np
import theano
from pylab import *
from progress.bar import Bar
from theano import tensor as T
from util import floatX, init_basis_hermite, init_bias, init_alphas, init_weights, rectify, softmax, dropout
from mnist_loader import mnist
from theano.sandbox.cuda.dnn import dnn_conv as cuconv
from theano.sandbox.cuda.dnn import dnn_pool as cupool
from lasagne.updates import adadelta
from pylearn2.expr.normalize import CrossChannelNormalizationBC01 as crossnorm

#---------------------------
# Model Definition
#---------------------------

def model(X, w_L1, w_L2, w_L3, w_L4, p_drop_conv, p_drop_hidden):
    crossnormalizer = crossnorm(alpha = 1e-4, k=2, beta=0.75, n=9)
    l1b = rectify(cuconv(X,w_L1, border_mode=(5,5)))
    l1 = cupool(l1b,(3,3), stride=(2, 2))
    l1 = crossnormalizer(l1)
    l1 = dropout(l1, p_drop_conv)

    l2b = rectify(cuconv(l1,w_L2, border_mode='full'))
    l2 = cupool(l2b,(3,3), stride=(2, 2))
    l2 = crossnormalizer(l2)
    l2 = dropout(l2, p_drop_conv)

    l3b = rectify(cuconv(l2,w_L3, border_mode='full'))
    l3 = cupool(l3b, (3,3), stride=(2, 2))
    l3 = crossnormalizer(l3)

    l4 = T.flatten(l3, outdim=2)

    pyx = dropout(l4, p_drop_hidden)
    pyx = softmax(T.dot(l4, w_L4))
    return l1, l2, l3, pyx

#--------------------------
# Load MNIST
# ntrain = # of samples in randomly chosen subset
# This is to reproduce Fig. 5 in the paper
#--------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--ntrain', nargs=1, type=int)
parser.add_argument('--epochs', nargs=1, type=float)
args = parser.parse_args()

trX, teX, trY, teY = mnist(args.ntrain, onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()
lr = T.scalar()
epochs = T.scalar()

#-------------------------
# Init Basis and Alphas
#-------------------------

bases_L1 = 10
sigma_L1 = 1.5
bases_L2 = 6
sigma_L2 = 1
bases_L3 = 6
sigma_L3 = 1

basis_L1 = init_basis_hermite(sigma_L1,bases_L1)
basis_L2 = init_basis_hermite(sigma_L2,bases_L2)
basis_L3 = init_basis_hermite(sigma_L3,bases_L3)

alphas_L1 = init_alphas(64,1,bases_L1)
alphas_L2 = init_alphas(64,64,bases_L2)
alphas_L3 = init_alphas(64,64,bases_L3)

w_L1 = T.sum( alphas_L1[:,:,:,None,None] * basis_L1[None,None,:,:,:], axis=2)
w_L2 = T.sum( alphas_L2[:,:,:,None,None] * basis_L2[None,None,:,:,:], axis=2)
w_L3 = T.sum( alphas_L3[:,:,:,None,None] * basis_L3[None,None,:,:,:], axis=2)
w_L4 = init_weights((3136, 10))

#-------------------------
# Set up function
#-------------------------

noise_l1, noise_l2, noise_l3, noise_py_x = model(X, w_L1, w_L2, w_L3, w_L4, 0.2, 0.7)
l1, l2, l3, py_x = model(X, w_L1, w_L2, w_L3, w_L4, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [alphas_L1, alphas_L2, alphas_L3, w_L4]
updates = adadelta(cost, params, learning_rate=lr, rho=0.95, epsilon=1e-6)

train = theano.function(inputs=[X, Y, lr], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

#-------------------------
# Train model
#-------------------------

d={}
batch_size = 25
epochs = args.epochs[0]
epoch_count = np.array(args.epochs).astype(int)

bar = Bar(' Training RFNN on MNIST - Epoch ', max=epoch_count)
for i in range(epoch_count):
    lr= 5.0 *(epochs-i)/(epochs)
    for start, end in zip(range(0, len(trX), batch_size-1), range(batch_size, len(trX)+1, batch_size)):
        cost = train(trX[start:end], trY[start:end], lr)
    y_x=predict(teX)
    bar.next()
bar.finish()
print ('Final Accuracy %(accuracy)f' % \
{"accuracy": np.mean(np.argmax(teY, axis=1) == y_x)})
