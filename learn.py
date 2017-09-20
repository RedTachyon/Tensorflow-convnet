#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_preprocessing import *
from graph_construction import *

def train_validation_split(data, labels, n_valid):
    valid_data, train_data = data[:n_valid], data[n_valid:]
    valid_labels, train_labels = labels[:n_valid], labels[n_valid:]
    
    return train_data, train_labels, valid_data, valid_labels

def load_all_data(path_train, path_test, n_valid=1000):
    """
    Loads train and test data from the given paths, normalizes them properly.
    """
    data, mean, std, labels = extract_and_norm_data('train_32x32.mat')
    test_data, test_labels = extract_test_data('test_32x32.mat', mean, std)
    
    train_data, train_labels, valid_data, valid_labels = train_validation_split(data, labels, n_valid)
    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels

def get_batch(data, labels, step, size):
    """
    Get a minibatch of the given size, maintaining the appripriate order.
    
    Args:
        data: np.array, contains features, first dimension corresponds to samples
        labels: np.array, contains labels, first dimension corresponds to samples
        step: int, step of the training
        size: int, size of the minibatch
        
    returns:
        two arrays with shape (size, (...)), containing the data and labels minibatch
    """
    # Rename variables for mathematical elegance
    m = data.shape[0]
    b = size
    i = step
    
    low = int(b * (i % np.ceil(m/b)))
    high = int(b * (i % np.ceil(m/b)) + b)
    return data[low:high], labels[low:high]

def train_model(num_epochs, batch_size, learning_rate):
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_all_data('train_32x32.mat',
                                                                                               'test_32x32.mat')
    model, train_op, accuracy, X, y, prob = classifier(1e-3, True)
    
    with tf.Session() as sess:
        sess.run(model)
        accs = np.array([])
        val_accs = np.array([])
        for step in tqdm(range(500)):
            data_batch, label_batch = get_batch(train_data, train_labels, step, 128)
            _, acc = sess.run([train_op, accuracy], feed_dict={X: data_batch, y: label_batch, prob: .8})
            accs = np.append(accs, acc)
            if not step % 10:
                val_acc = sess.run(accuracy, feed_dict={X: valid_data, y: valid_labels})
                val_accs = np.append(val_accs, val_acc)
        
        plt.plot(range(accs.shape[0]), accs)
        plt.show()
        plt.plot(range(val_accs.shape[0]), val_accs)
        plt.show()
        
if __name__ == '__main__':
    train_model(1, 128, 1e-3)