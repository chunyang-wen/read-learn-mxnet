#!/usr/bin/python
#coding:utf-8

import sys
import numpy as np
import mxnet as mx

# hyper paramters
batch_size = 32
data_size = 1000
test_size = 100

# Y = 2 * X + Z
train_X = np.random.uniform(0, 1, [data_size, 2])
train_Y = np.array([train_X[i][0] + 2 * train_X[i][1] for i in xrange(data_size)])
train_iter = mx.io.NDArrayIter(data={'X':train_X}, label={'Y':train_Y}, batch_size=batch_size, shuffle=True)

test_X = np.random.uniform(0, 1, [test_size, 2])
test_Y = np.array([test_X[i][0] + 2 * test_X[i][1] for i in xrange(test_size)])
test_iter = mx.io.NDArrayIter(data={'X':test_X}, label={'Y':test_Y}, batch_size=batch_size, shuffle=False)

X = mx.sym.Variable('X')
Y = mx.sym.Variable('Y')

#fc1 = mx.sym.FullyConnected(data=X, num_hidden=128, name='fc1')
#ac1 = mx.sym.Activation(data=fc1, act_type='relu', name='ac1')

#fc2 = mx.sym.FullyConnected(data=ac1, num_hidden=1, name='fc2')
fc2 = mx.sym.FullyConnected(data=X, num_hidden=1, name='fc2')
res = mx.sym.LinearRegressionOutput(data=fc2, label=Y, name='lrout')

model = mx.mod.Module(symbol=res,
        data_names=['X'],
        label_names=['Y'])

model.fit(train_data=train_iter,
        eval_data=None,
        optimizer='sgd',
        optimizer_params = {'learning_rate':0.001, 'momentum':0.9},
        num_epoch=1000)

for predict, _, batch in model.iter_predict(test_iter):
    print predict[0].asnumpy()
    actual = batch.data[0].asnumpy()
    y = [ i[0] + 2 * i[1] for i in actual]
    print y
    break

sys.exit(0)
metric = mx.metric.create('mse')

model.score(test_iter, metric)
print metric


