#!/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from data_preprocessing import *
from graph_construction import *

def train_validation_split(data, labels, n_valid):
    """
    Splits the data to a training and validation sets.
    """
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

def get_big_accuracy(data, labels, accuracy, sess, X, y, batch_size=100):
    """
    Computes the accuracy over the data in case it's too large to simply run the accuracy op on it.
    """
    m = data.shape[0]
    full_score = np.array([])
    for step in tqdm(np.arange(np.ceil(m/batch_size))):
        eval_data, eval_labels = get_batch(data, labels, step, batch_size)
        acc = sess.run(accuracy, feed_dict={X: eval_data, y: eval_labels})
        full_score = np.append(full_score, acc * eval_data.shape[0])
    return np.sum(full_score) / m

def train_model(num_epochs, batch_size, learning_rate):
    """
    Trains the convnet model and measures its accuracy on the train, validation and test sets.
    """
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = load_all_data('train_32x32.mat',
                                                                                               'test_32x32.mat')
    model, train_op, accuracy, X, y, prob = classifier(1e-3, True)
    
    num_iters = int(np.ceil(train_data.shape[0] / batch_size) * num_epochs)
    
    with tf.Session() as sess:
        print("Training the model")
        sess.run(model)
        accs = np.array([])
        val_accs = np.array([])
        
        for step in tqdm(range(num_iters)):
            data_batch, label_batch = get_batch(train_data, train_labels, step, batch_size)
            _, acc = sess.run([train_op, accuracy], feed_dict={X: data_batch, y: label_batch, prob: .8})
            accs = np.append(accs, acc)
            if not step % 10:
                val_acc = sess.run(accuracy, feed_dict={X: valid_data, y: valid_labels})
                val_accs = np.append(val_accs, val_acc)
        
        plt.subplot(211)
        plt.plot(range(accs.shape[0]), accs)
        plt.subplot(212)
        plt.plot(range(val_accs.shape[0]), val_accs)
        
        print("Evaluating the training set")
        train_acc = get_big_accuracy(train_data, train_labels, accuracy, sess, X, y)
        print("Evaluating the validation set")
        val_acc = get_big_accuracy(valid_data, valid_labels, accuracy, sess, X, y)
        print("Evaluating the test set")
        test_acc = get_big_accuracy(test_data, test_labels, accuracy, sess, X, y)
        
        print("Training accuracy: %.3f" % train_acc)
        print("Validation accuracy: %.3f" % val_acc)
        print("Test accuract: %.3f" % test_acc)
        
        plt.show()
        
        
if __name__ == '__main__':
    train_model(1, 128, 1e-3)