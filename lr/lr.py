#!/bin/env python
#coding:utf-8

import sys

import mxnet as mx
import numpy as np

# Training data
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in xrange(100)])
batch_size = 1

# Evaluation data

eval_data = np.array([[7,2], [6, 10], [12, 2]])
eval_label = np.array([11, 26, 16])

train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size, shuffle=True, label_name='lin_reg_label')
eval_iter  = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)


X = mx.sym.Variable('data')
Y = mx.sym.Variable('lin_reg_label')
fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name='lro')

model = mx.mod.Module(
        symbol = lro,
        data_names=['data'],
        label_names=['lin_reg_label']
        )

network_graph = mx.viz.plot_network(symbol=lro)
network_graph.render('haha')

model.fit(train_iter, eval_iter,
        optimizer_params={'learning_rate':0.005, 'momentum':0.9},
        num_epoch=1000,
        batch_end_callback = mx.callback.Speedometer(batch_size, 2))

print "Predict result: "
print model.predict(eval_iter).asnumpy()

metric = mx.metric.MSE()

print "MSE: "
print model.score(eval_iter, metric)
