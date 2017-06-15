#!/bin/env python
#coding:utf-8

import os
import sys
import mxnet as mx

import logging
#logger = logging.getLogger()
#logger.setLevel(logging.DEBUG)

data_path = os.getenv("DATA_SOURCE")
train_data_path = os.path.join(data_path, 'ml-100k', 'u1.base')
test_data_path = os.path.join(data_path, 'ml-100k', 'u1.test')
batch_size = 30

def load_data(path):
    users = []
    items = []
    scores = []
    max_user=None
    max_item=None
    with open(path) as fd:
        for line in fd:
            try:
                user, item, score, _ = line.split('\t')
            except:
                continue
            if max_user is None:
                max_user = user
            elif max_user < user:
                max_user = user

            if max_item is None:
                max_item = item
            elif max_item < item:
                max_item = item

            users.append(user)
            items.append(item)
            scores.append(score)
        user_iter = mx.nd.array(users)
        item_iter = mx.nd.array(items)
        scores_iter = mx.nd.array(scores)
    return mx.io.NDArrayIter(data={'user': user_iter, 'item': item_iter},
            label={'score': scores_iter}, batch_size=batch_size, shuffle=True), max_user, max_item

def net(k, max_user, max_item):
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    user = mx.sym.Embedding(data=user, input_dim=max_user, output_dim=k)
    item = mx.sym.Embedding(data=item, input_dim=max_item, output_dim=k)
    pred = user * item
    pred = mx.sym.sum(data=pred, axis=1)
    pred = mx.sym.Flatten(data=pred)
    pred = mx.sym.LinearRegressionOutput(data=pred, label=score)
    return pred

if __name__ == '__main__':
    train_iter, max_user, max_item = load_data(train_data_path)
    """
    for batch in train_iter:
        print batch
        print batch.data
        print batch.data[0].asnumpy()
        print batch.data[0].shape
        sys.exit(0)
    """
    test_iter, _, _  = load_data(test_data_path)
    mynet = net(64, max_user, max_item)


    model = mx.mod.Module(symbol=mynet,
            data_names=['user', 'item'],
            label_names=['score'])

    """
    print train_iter.provide_data
    print train_iter.provide_label
    model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    print model.data_shapes
    print model.label_shapes
    sys.exit(0)
    """

    hah = mx.viz.plot_network(symbol=mynet)
    hah.render('hah')
    model.fit(train_data=train_iter,
            eval_data=None,
            num_epoch=100,
            optimizer = 'sgd',
            optimizer_params = {'learning_rate': 0.1, 'momentum': 0.9},
            batch_end_callback = mx.callback.Speedometer(batch_size, 2))
    metric = mx.metric.create('rmse')
    model.score(test_iter, metric)
    print metric
    model.save_params('I-am-model')

    """
    for pred, i_batch, batch in model.iter_predict(test_iter):
        print pred[0].asnumpy()
        print batch.data[0].asnumpy()
        print batch.data[1].asnumpy()
        print batch.label[0].asnumpy()
        break
    """
