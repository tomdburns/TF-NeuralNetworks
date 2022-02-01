"""
Utilities Subroutines for the Tensorflow code
"""

import h5py
import numpy as np
import tensorflow as tf
import math


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
#    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    #print X.shape
    #print Y.shape
    #exit()
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]#.reshape((Y.shape[1], Y.shape[0]))
    #print shuffled_X
    #print shuffled_Y
    #exit()

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
#        mini_batch_X = shuffled_X[ :, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
#        mini_batch_Y = shuffled_Y[ :, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
#        mini_batch_X = mini_batch_X.reshape(mini_batch_X.shape[1], mini_batch_X.shape[0])
#        mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[1], mini_batch_Y.shape[0])
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m, :]
#        mini_batch_X = shuffled_X[ :, num_complete_minibatches * mini_batch_size : m]
#        mini_batch_Y = shuffled_Y[ :, num_complete_minibatches * mini_batch_size : m]
#        mini_batch_X = mini_batch_X.reshape(mini_batch_X.shape[1], mini_batch_X.shape[0])
#        mini_batch_Y = mini_batch_Y.reshape(mini_batch_Y.shape[1], mini_batch_Y.shape[0])
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
#    print mini_batch_Y
#    exit() 
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters, options, size):
    """Used to predict performance of test set"""
#    print X.shape; exit()
    n_layers = options.getint('layers')
    out = []
    for vector in X:
        n_vect = vector.reshape(vector.shape[0], 1)
        params = {}
        for i in range(0, n_layers):
            w_l = 'W{}'.format(i + 1)
            b_l = 'b{}'.format(i + 1)
            W = tf.convert_to_tensor(parameters[w_l])
            b = tf.convert_to_tensor(parameters[b_l])
            params[w_l] = W
            params[b_l] = b
        
#        x = tf.placeholder(tf.float32, [X.shape[0], X.shape[1]])
        x = tf.placeholder(tf.float32, [n_vect.shape[0], n_vect.shape[1]])
        
        z3 = forward_propagation_for_predict(x, params, n_layers, options)
        p = tf.argmax(z3)
        
        sess = tf.Session()
#        prediction = sess.run(p, feed_dict = {x: X})
        prediction = sess.run(p, feed_dict = {x: n_vect})
        #print prediction
        #exit()
        out.append(prediction)
#        
    return np.array(out)
#    return prediction


def forward_propagation_for_predict(X, parameters, n_layers, options):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    trans = options.get('non-linear').upper()

    # Retrieve the parameters from the dictionary "parameters"
    for i in range(0, n_layers):
        w_l = 'W{}'.format(i + 1)
        b_l = 'b{}'.format(i + 1)
        W = parameters[w_l]
        b = parameters[b_l]
        if i == 0:
            Z = tf.add(tf.matmul(W, X), b)
        else:
            Z = tf.add(tf.matmul(W, A), b)
        if 'RELU' in trans:
            A = tf.nn.relu(Z)
        elif 'TANH' in trans:
            A = tf.nn.tanh(Z)
        elif 'SIGMOID' in trans:
            A = tf.nn.sigmoid(Z)
        else:
            print "\n!!!!ERROR!!!!\nInvalid Selection For Non-Linear Transformation:"
            print "{}".format(trans)
            print "Defaulting to SIGMOID\n!!!!!!!!!!!!!\n"
            A = tf.nn.sigmoid(Z)
    return Z
