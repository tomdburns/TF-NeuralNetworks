#!/usr/bin/env python

"""
Artificial Neural Network Code that uses TensorFlow

Version 0.0.1
"""


import os, sys
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas
import pickle as pkl
from glob import glob
from scipy.stats import pearsonr
from pandas import DataFrame as df
from sys import argv
from sklearn.utils import shuffle
from config import Options
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from logging import debug, error, info, warning


if len(argv) > 1:
    script, FILENAME = argv[0], argv[-1]
else:
    print "\nTo Run Script:\n\tscript -o [OPTIONS] FILENAME\n"
    exit()
if not os.path.exists(FILENAME):
    print "File: {} Does not exist".format(FILENAME)
    print "\nTo Run Script:\n\tscript -o [OPTIONS] FILENAME\n"
    exit()


OPTIONS = Options()
CWD = os.getcwd()
CLSC = 'MCC' # Select the type of function to use


def import_file():
    """Imports data from file"""
    raw_x_cols = OPTIONS.gettuple('x_cols')
    raw_y_cols = OPTIONS.gettuple('y_cols')
    x_cols, y_cols = [], []
    for val in raw_x_cols:
        x_cols.append(val)
    for val in raw_y_cols:
        y_cols.append(val)
    #exit()
    if OPTIONS.getbool('shuffle'):
        data = shuffle(pandas.read_csv(FILENAME))
    else:
        data = pandas.read_csv(FILENAME)
    print y_cols, "\n\n\n"
    return data[x_cols], data[y_cols]


def initialize_parameters(size, data_format = 'rows'):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    parameters = {}
    n_layers   = OPTIONS.getint('layers')
    if n_layers > 1:
        vals = [int(val) for val in OPTIONS.gettuple('vector_size')]
    else:
        vals = OPTIONS.getint('vector_size')
        
    ### START CODE HERE ### (approx. 6 lines of code)
    for i in range(0, n_layers):
        w_l = 'W{}'.format(i + 1)
        b_l = 'b{}'.format(i + 1)
        print 'i =', i, ', size =', size, ', vals[i] =', vals[i]

        if data_format == 'rows':
           W = tf.get_variable(w_l, [size, vals[i]],
                               initializer = tf.contrib.layers.xavier_initializer(seed = 1))
           b = tf.get_variable(b_l, [1, vals[i]], initializer = tf.zeros_initializer())
        else:
           W = tf.get_variable(w_l, [vals[i], size],
                               initializer = tf.contrib.layers.xavier_initializer(seed = 1))
           b = tf.get_variable(b_l, [vals[i], 1], initializer = tf.zeros_initializer())
        parameters[w_l] = W
        parameters[b_l] = b
        size = vals[i]

    return parameters


def create_placeholders(n_x, n_y, data_format='rows'):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the
      number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    if data_format == 'rows':
       X = tf.placeholder(tf.float32, shape = (None, n_x), name = "X")
       Y = tf.placeholder(tf.float32, shape = (None, n_y), name = "Y")
    else:
       X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
       Y = tf.placeholder(tf.float32, shape = (n_y, None), name = "Y")
    ### END CODE HERE ###
    
    return X, Y


def forward_propagation(X, parameters, data_format='rows'):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    trans = OPTIONS.get('non-linear').upper()
    n_layers = OPTIONS.getint('layers')
    for i in range(0, n_layers):
        w_l = 'W{}'.format(i + 1)
        b_l = 'b{}'.format(i + 1)

        # Numpy Equivalents:
        # Z1 = np.dot(W1, X) + b1
        # A1 = relu(Z1)
        # Z2 = np.dot(W2, a1) + b2
        # A2 = relu(Z2)
        # Z3 = np.dot(W3,Z2) + b3
        if i == 0:
            if data_format == 'rows':
               Z = tf.add(tf.matmul(X, parameters[w_l]), parameters[b_l])
            else:
               Z = tf.add(tf.matmul(parameters[w_l], X), parameters[b_l])
        else:
            if data_format == 'rows':
               Z = tf.add(tf.matmul(A, parameters[w_l]), parameters[b_l])
            else:
               Z = tf.add(tf.matmul(parameters[w_l], A), parameters[b_l])
        if 'RELU6' in trans:
            A = tf.nn.relu6(Z)
        elif 'CRELU' in trans:
            A = tf.nn.crelu(Z)
        elif 'RELU' in trans:
            A = tf.nn.relu(Z)
        elif 'SELU' in trans:
            A = tf.nn.selu(Z)
        elif 'ELU' in trans:
            A = tf.nn.elu(Z)
        elif 'SOFTPLUS' in trans:
            A = tf.nn.softplus(Z)
        elif 'SOFTSIGN' in trans:
            A = tf.nn.softsign(Z)
        elif 'DROPOUT' in trans:
            A = tf.nn.dropout(Z)
        elif 'BIAS_ADD' in trans:
            A = tf.nn.bias_add(Z)
        elif 'TANH' in trans:
            A = tf.nn.tanh(Z)
        elif 'SIGMOID' in trans:
            A = tf.nn.sigmoid(Z)
        else:
            print_out("\n!!!!ERROR!!!!\nInvalid Selection For Non-Linear Transformation:")
            print_out("{}".format(trans))
            print_out("Defaulting to SIGMOID\n!!!!!!!!!!!!!\n")
            A = tf.nn.sigmoid(Z)

    return Z

    
def size(frame):
    """Calculates the dimensions of the matrix"""
    y = 0
    for val in frame:
        y += 1
    return y


def compute_cost(Z3, Y, classf):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    if classf:
#        logits = tf.transpose(Z3)
#        labels = tf.transpose(Y)
#        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))


#        val  = subcost(Z3, Y)
        cost = tf.metrics.accuracy(Y, Z3)
#        if 'MCC' in CLSC:
#            val = abs(val)
#        if val != 0.0:
#            cost = 1. / val
#        else:
#            cost = 1.0
    else:
        cost = tf.losses.mean_squared_error(Y, Z3)
    #cost = tf.losses.absolute_difference(Y, Z3)
    return cost


def split_sets(dset, scale=False):
    """Splits the dsets into training and test sets"""
    ratio = OPTIONS.getfloat('test_ratio')
    size = int(len(dset) * ratio)
    dset = np.array(dset, dtype=np.float32)
    key = {}
    if scale:
        print_out("  Scaling Set...")
        dset, key = scale_arrays(dset)
        print_out("  Done.")
    return dset[:size], dset[size:], key


def scale_arrays(dat):
    """Scales the input vector arrays"""
    new_dat, vals, key = [], [], {}

    # Pull out the data
    print_out("     Pulling Statistics...")
    for item in dat:
        for idx, val in enumerate(item):
            try:
                vals[idx].append(val)
            except IndexError:
                vals.append([val])
    # Calculate the required statistics
    print_out("     Calculating Statistics...")
    for idx, set in enumerate(vals):
        label = 'col_{}'.format(idx)
        if label not in key:
            key[label] = {}
        key[label]['max']   = max(set)
        key[label]['min']   = min(set)
        key[label]['range'] = max(set) - min(set)
    # Scale the array
    print_out("     Scaling Arrays...")
    for idx, par in enumerate(dat):
        cur = []
        for indx, val in enumerate(par):
            lbl = 'col_{}'.format(indx)
            nval = (val - key[lbl]['min']) / key[lbl]['range']
            cur.append((nval * 2) - 1)
        new_dat.append(cur)

    return np.array(new_dat), key


def cost_result(costs):
    """Prints the cost results to an output file"""
    if '.csv' in FILENAME:
        name = FILENAME.split('.csv')[0]
    else:
        name = FILENAME
    prevs = glob('{}-costs-*.csv'.format(name))
    new = '{}-costs-{}.csv'.format(name, len(prevs))
    out = open(new, 'w')
    out.write('Epoch,Cost\n')
    for item in costs:
        out.write('{},{}\n'.format(item[0], item[1]))
    out.flush()
    out.close()


def pearson(pred, actu):
    """Calculates a pearson R squared"""
    #print pred
    #print actu
    avg  = np.mean(actu)
    stot, sres = 0., 0.
    A, B, C = 0., 0., 0.
    D, E = 0., 0.
    n = len(pred)
    for i, val in enumerate(actu):
        if math.isnan(val[0]):
            continue
        if math.isnan(pred[i][0]):
            continue
        A += (float(val[0]) * float(pred[i][0]))
        B += (float(val[0]) ** 2)
        C += (float(pred[i][0]) ** 2)
        D += val[0]
        E += pred[i][0]
        if math.isnan(A):
            print val[0], pred[i][0]
    #print "=" * 50
    #print "DEBUGGING"
    #print "-" * 50
    #print "A:", A
    #print "B:", B
    #print "C:", C
    #print "Sum Actual   :", D  #np.sum(actu)
    #print "Sum Predicted:", E  #np.sum(pred)
    top = (n * A) - (E * D)
#    top = (n * A) - (np.sum(pred) * np.sum(actu))
#    bot = ((n * B) - (np.sum(actu) ** 2)) * ((n * C) - (np.sum(pred) ** 2))
    bot = ((n * B) - (D ** 2)) * ((n * C) - (E ** 2))
    #print "Top:", top
    #print "Bottom:", bot
    bot = bot ** 0.5
    #print "SquareRoot(Bottom):", bot
    #print "=" * 50
    return top / bot


def logo():
    """Print the code's logo"""
    print_out("Starting....")
    print_out(" -----------------------------------------------------------------------")
    print_out(" ___________                             ___________.__")
    print_out(" \__    ___/___   ____   ________________\_   _____/|  |   ______  _  __")
    print_out("   |    |_/ __ \ /    \ /  ___/ _  \_  __ \    __)  |  |  /  _ \ \/ \/ /")
    print_out("   |    |\  ___/|   |  \\___  ( <_>  )  | \/     \   |  |_(  <_> )     /")
    print_out("   |____| \___  >___|  /____  >____/|__|  \___  /   |____/\____/ \/\_/")
    print_out("              \/     \/     \/                \/")
    print_out("                                                           Version 0.0.1")
    print_out(" -----------------------------------------------------------------------")


def print_out(string, live=False):
    """prints out the results"""
    if '.csv' in FILENAME:
        name = FILENAME.split('.csv')[0]
    else:
        name = FILENAME
    f_name = '{}/{}.tout'.format(CWD, name)
    if os.path.exists(f_name):
        out = open(f_name, 'a')
    else:
        out = open(f_name, 'w')
    out.write('{}\n'.format(string))
    out.flush()
    out.close()
    if not OPTIONS.getbool('silent') and not live:
        print string


def dump_model(model):
    """Saves the model to a binary file"""
    if '.csv' in FILENAME:
        name = FILENAME.split('.csv')[0]
    else:
        name = FILENAME
    prevs = glob('{}-model-*.pkl'.format(name))
    new = '{}-model-{}.pkl'.format(name, len(prevs))
    out = open(new, 'wb')
    pkl.dump(model, out)
    out.flush()
    out.close()
    print_out("\nSaving Model to File: {}-model-{}.pkl\n".format(name, len(prevs)))


def check_convergence(raw):
    """Checks for convergence in the cost function"""
    vals = []
    for val in raw:
        vals.append(float(val[1]))
    conv = False
    avg  = np.mean(vals)
    stdv = np.std(vals)
    print_out("  >: Convergence Check: {}".format(stdv / avg))
    if stdv / avg < OPTIONS.getfloat('conv_tol'):
        conv = True
    return conv


def subcost(Z, Y):
    """Calculates the classic loss function variables"""
#    print "Z Shape:", shape(Z)
#    print "Y Shape:", shape(Y)
#    Z = np.array(Z)
#    Y = np.array(Y)
    print Z
    exit()
    TP, FP, TN, FN = 0., 0., 0., 0.
    for idx, val in enumerate(Z):
        if val == Y[idx]:
            if val == 1:
                TP += 1.
            else:
                TN += 1.
        else:
            if val == 1:
                FP += 1.
            else:
                FN += 1.
    # Accuracy:
    acc = (TP + TN) / (TP + TN + FP + FN)
    # Recall
    rec = TP / (TP + FN)
    # Precision
    pres = TP / (TP + FP)
    # NPV
    npv = TN / (FN + TN)
    # Specificity
    spec = TN / (TN + FP)
    # F1 Score
    f1 = 1. / (((1. / rec) + (1. / pres)) / 2.)
    # MCC
    m_top = (TP * TN) - (FP * FN)
    m_bot = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    if m_bot == 0:
        mcc = 0.0
    else:
        mcc = m_top / m_bot

    Stats = {'MCC': mcc, 'Acc': acc, 'Pres': pres, 'NPV': npv, 'Spec': spec,
             'F1': f1}
    return Stats[CLSC]
    


def main():
    """Main Execution"""
    logo()
    # Import Data
    print_out("Importing Data...")
    x_dat, y_dat    = import_file()
    print_out("Done.\nMaking Sets...")
    X_test, X_train, x_key = split_sets(x_dat, scale=OPTIONS.getbool('x_scale'))
    Y_test, Y_train, y_key = split_sets(y_dat, scale=OPTIONS.getbool('y_scale'))
    print_out("Done.\n")
    m = size(X_train)

    print_out('x_dat.shape = {}'.format(np.array(x_dat.shape)))
    print_out('y_dat.shape = {}'.format(np.array(y_dat.shape)))

    # Set up the neural network
    classification = OPTIONS.getbool('classification')
    learning_rate  = OPTIONS.getfloat('learning_rate')
    X, Y           = create_placeholders(size(x_dat), size(y_dat), data_format = 'rows')
    print_out('X.shape = {}'.format(X.shape))
    print_out('Y.shape = {}'.format(Y.shape))

    parameters    = initialize_parameters(size(x_dat), data_format = 'rows')
    Z             = forward_propagation(X, parameters, data_format = 'rows')
    cost          = compute_cost(Z, Y, classification)

   # Select the optimizer
    selected_Optim = OPTIONS.get('optimizer').upper()
    if 'GRADIENTDESCENT'in selected_Optim:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    elif 'ADAM' in selected_Optim:
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    else:
        print_out("!!!Invalid optimizer selected: {}".format(selected_Optim))
        print_out("Defaulting to GradientDescent.\n")
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    init      = tf.global_variables_initializer()
    init_l    = tf.local_variables_initializer()

    # Import Remaining Parameters
    num_epochs     = OPTIONS.getint('num_epochs')
    minibatch_size = OPTIONS.getint('minibatch_size')
    print_cost     = OPTIONS.getbool('print_cost')
    seed           = OPTIONS.getint('seed')
    costs          = []

    print_out("=" * 72)
    print_out("Fitting Parameters:\n")
    print_out("Total Set Size: {}".format(len(x_dat)))
    print_out("Test Set Ratio: {}".format(OPTIONS.getfloat('test_ratio')))
    print_out("N_Layers      : {}".format(OPTIONS.getint('layers')))
    print_out("Num of Epochs : {}".format(num_epochs))
    print_out("Minibatch Size: {}\n".format(minibatch_size))
    print_out("Optimizer     : {}".format(OPTIONS.get('optimizer')))
    print_out("Non-Linear    : {}".format(OPTIONS.get('non-linear')))
    print_out("Learning Rate : {}".format(OPTIONS.getfloat('learning_rate')))
    print_out("Vector Size   : {}\n".format(OPTIONS.gettuple('vector_size')))
    print_out("X Scaled      : {}".format(OPTIONS.getbool('x_scale')))
    print_out("Y Scaled      : {}".format(OPTIONS.getbool('y_scale')))
    print_out("Shuffled      : {}".format(OPTIONS.getbool('shuffle')))
    print_out("Classifier    : {}".format(OPTIONS.getbool('classification')))
    print_out("=" * 72 + '\n')
    x_dat, y_dat = [], [] # This is to reduce the memory requirement of the code
                          # by clearing these two large variables

    conv      = OPTIONS.getbool('conv')
    conv_freq = OPTIONS.getint('conv_freq')
    conv_tol  = OPTIONS.getfloat('conv_tol')
    conv_min  = OPTIONS.getint('conv_min')
    conv_incl = OPTIONS.getint('conv_incl')
    converged = False
    if conv:
        print_out("Convergence Check Requested.\nUsing Parameters:")
        print_out("  >: Minimum  : {}".format(conv_min))
        print_out("  >: Frequency: {}".format(conv_freq))
        print_out("  >: Tolerance: {}".format(conv_tol))
        print_out("  >: Include  : {}".format(conv_incl))
        print_out("=" * 72 + "\n")

    nan_err = False
    # Start the Model
    with tf.Session() as sess:
        print_out("\n" + "=" * 72)
        print_out("Starting Fitting Using {} Epochs".format(num_epochs))
        print_out("-" * 72)
        # Run the initialization
        sess.run(init)
        sess.run(init_l)
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size
                                                      # minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
 
                _ , minibatch_cost = sess.run([optimizer, cost],
                                     feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % OPTIONS.getint('print_rate') == 0:
                #if OPTIONS.getbool('live'):
                #    print_out(("Cost after epoch %i: %f" % (epoch, epoch_cost)), live = True)
                #    sys.stdout.write("Cost after epoch %i: %f\r" % (epoch, epoch_cost))
                #else:
                print_out(("Cost after epoch %i: %f" % (epoch, epoch_cost)))
            if print_cost == True and epoch % OPTIONS.getint('save_rate') == 0:
                costs.append([epoch, epoch_cost])
            if conv and epoch >= conv_min and epoch % conv_freq == 0:
                if epoch > conv_incl:
                    converged =  check_convergence(costs[-1 * conv_incl:])
                else:
                    converged =  check_convergence(costs)
                if converged or epoch_cost <= 0.000001:
                    convered = True
                    break
                elif math.isnan(epoch_cost):
                    nan_err = True
                    break

        if OPTIONS.getbool('live'):
            print "\n"
        # lets save the parameters in a variable
        if conv:
            print_out("\n" + "=" * 72)
            if converged:
                print_out("Convergence Criteria Met.")
                print_out("Ending Fitting After {} Epochs\n".format(epoch))
            else:
                print_out("Convergence Criteria Not Met.")
                if not nan_err:
                    print_out("Ending Fitting After the Max Epochs: {}\n".format(num_epochs))
                else:
                    print_out("Ended because stuck in NaN loop.\n")

        parameters = sess.run(parameters)
        print_out("\nParameters have been trained!")
        print_out("-" * 72)
        if math.isnan(epoch_cost):
            print_out("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print_out("!!!TRAINING FAILED: Returned All nan Values for Cost!!!")
            print_out("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            out = open('result.ga', 'w')
            out.write('failed,failed')
            out.close()
            exit()

        model = {'parameters': parameters,
                 'options'   : OPTIONS}

        dump_model(model)
        cost_result(costs)
        # Calculate the correct predictions

#        if OPTIONS.getbool('display_cost'):
#            #plt.plot(np.squeeze(costs))
#            y_cost, x_epoch = [], []
#            for val in costs:
#                x_epoch.append(val[0])
#                y_cost.append(val[1])
#            plt.scatter(x_epoch, y_cost)
#            plt.ylabel('cost')
#            plt.xlabel('iterations (per tens)')
#            plt.title("Learning rate =" + str(learning_rate))
#            plt.show()

        # Calculate accuracy on the test set

        Y_train = Y_train.reshape(Y_train.shape[1], Y_train.shape[0])

        train_pred = sess.run(Z, feed_dict={X: X_train})
        test_pred = sess.run(Z, feed_dict={X: X_test})

        train_r = pearsonr(train_pred, Y_train.T)[0] ** 2
        test_r = pearsonr(test_pred, Y_test)[0] ** 2
        print_out("\n" + "=" * 79 + "\n" + "Results:\n")
        print_out("Train Accuracy (Pearson R2): %.3f" % train_r[0])
        print_out("Test Accuracy  (Pearson R2): %.3f" % test_r[0])
        if math.isnan(train_r[0]) or math.isnan(test_r[0]):
            alt_train_r = pearson(train_pred, Y_train.T) ** 2
            alt_test_r = pearson(test_pred, Y_test) ** 2
            print_out("\nAlternate Pearson Calculations:")
            print_out("Train Accuracy (Pearson R2): %.3f" % alt_train_r)
            print_out("Test Accuracy  (Pearson R2): %.3f" % alt_test_r)

            out = open('result.ga', 'w')
            out.write('{},{}'.format(alt_train_r, alt_test_r))
            out.close()
        else:
            out = open('result.ga', 'w')
            out.write('{},{}'.format(train_r[0], test_r[0]))
            out.close()
        print_out("=" * 79)
 
        if OPTIONS.getbool('display_cost'):
            #plt.plot(np.squeeze(costs))
            y_cost, x_epoch = [], []
            for val in costs:
                x_epoch.append(val[0])
                y_cost.append(val[1])
            plt.scatter(x_epoch, y_cost)
            #plt.plot(x_epoch, y_cost)
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        print_out("Program Terminated Normally.")


if __name__ in '__main__':
    main()

