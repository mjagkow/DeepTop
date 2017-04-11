import numpy as np
import pandas as pd

data_dir = './'

class Dataset(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.shuffle()

    def shuffle(self):
        perm = np.arange(self.n)
        np.random.shuffle(perm)
        self.x = self.x[perm]
        self.y = self.y[perm]
        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id + batch_size >= self.n:
            self.shuffle()

        cur_id = self._next_id
        self._next_id += batch_size
        return self.x[cur_id : self._next_id], self.y[cur_id : self._next_id]



class HiggsDataset(object):
    def __init__(self, data_dir):
        data = np.load(data_dir + "/higgs_train.npy")
        self.train = Dataset(data[:, 1:], data[:, 0])
        data = np.load(data_dir + "/higgs_valid.npy")
        self.valid = Dataset(data[:, 1:], data[:, 0])
        data = np.load(data_dir + "/higgs_test.npy")
        self.test = Dataset(data[:, 1:], data[:, 0])

higgs = HiggsDataset(data_dir)

print (higgs)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(higgs.train.x.mean(0), label='mean')
plt.plot(higgs.train.x.std(0), label='std')
plt.legend()

#for i in dir(p):
#    print (i)
#print (p)

plt.savefig('tst.png')



def linear(x, name, size, bias=True):
    w = tf.get_variable(name + "/W", [x.get_shape()[1], size])
    b = tf.get_variable(name + "/b", [1, size], initializer=tf.zeros_initializer)
    return tf.matmul(x, w) + b


class HiggsLogisticRegression(object):
    def __init__(self, lr=0.1):
        self.x = x = tf.placeholder(tf.float32, [None, 28])
        self.y = tf.placeholder(tf.float32, [None])
        x = linear(x, "regression", 1)
        self.p = tf.nn.sigmoid(x)
        self.loss = loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( x, tf.reshape(self.y, [-1, 1]) ) )
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)



def train(model, dataset, batch_size=128):
    epoch_size = dataset.n / batch_size
    losses = []
    for i in range(epoch_size):
        train_x, train_y = dataset.next_batch(batch_size)
        loss, _ = sess.run([model.loss, model.train_op], {model.x: train_x, model.y: train_y})
        losses.append(loss)
        if i % (epoch_size / 5) == 5:
            tf.logging.info( "%.2f: %.3f", i * 1.0 / epoch_size, np.mean(losses) )
    return np.mean(losses)

import sys
sys.path.append('/afs/cern.ch/work/i/imyagkov/public/dnn/scikit-learn')

from sklearn.metrics import roc_auc_score
#import sk



def evaluate(model, dataset, batch_size=1000):
    dataset.shuffle()
    ps = []
    ys = []
    for i in range(dataset.n / batch_size):
        tx, ty = dataset.next_batch(batch_size)
        p = sess.run(model.p, {model.x: tx, model.y: ty})
        ps.append(p)
        ys.append(ty)
    ps = np.concatenate(ps).ravel()
    ys = np.concatenate(ys).ravel()
    return roc_auc_score(ys, ps)

with tf.variable_scope("model1", reuse=True):
    model = HiggsLogisticRegression()
    tf.initialize_all_variables().run()

logistic_aucs = []
for i in range(25):
    sys.stdout.write("EPOCH: %d " % (i + 1))
    train(model, higgs.train, 8 * 1024)
    valid_auc = evaluate(model, higgs.valid, 20000)
    print "VALID AUC: %.3f" % valid_auc
    logistic_aucs += [valid_auc]
