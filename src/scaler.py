#!/usr/bin/env python

"""
A pre-scaler code - scales the input csv prior to running the
Tensorflow code. This allows for a more rapid execution of the
ANN fittings since scaling a large data set is the slowest
portion of the main code.
"""


import sys
import pandas
import numpy as np
import pickle as pkl
from sys import argv


X_COLS = ['b0_c', 'd0_c', 'q1_c', 'q2_c', 'U1_c', 'U2_c', 'b0_n',
          'd0_n', 'q1_n', 'q2_n', 'U1_n', 'U2_n', 'StructuredDensity',
          'InletTemp', 'tads', 'tblow', 'tevac', 'Pint', 'Plow', 'v0']
#X_COLS = []
Y_COLS = ['Purity', 'Recovery', 'Productivity', 'Etot', 'EPSA', 'EComp']
#Y_COLS = []


def print_help(options, descr):
    """Prints the options"""
    print "-" * 79
    print "Help Requested:\n"
    print "Command line submission: tensor_scale [OPTIONS] [infile]\n"
    print "-" * 79
    print "Options:\n"
    for option in options:
        if option == 'name':
            continue
        print ">  ", option, '\t', descr[option],
        print "(Default = {})".format(options[option]), "\n"
    print "-" * 79
    exit()


def import_options():
    """Imports the command line options"""
    global X_COLS, Y_COLS
    # Set Defaults
    options = {'x_scale': True,
               'y_scale': False,
               'infile' : None,
               'name'   : None,
               'type'   : 'basic'}
    # Descriptioin of all options for help function
    descrip = {'x_scale': "-x: Scale the descriptor set",
               'y_scale': "-y: Scale the results set",
               'infile' : "The input file being scaled",
               'type'   : "Type of scaling selected"}

    if '--help' in argv or '-help' in argv:
        print_help(options, decsrip)

    if len(argv) < 2:
        print_help(options, descrip)

    if len(argv) == 2:
        options['infile'] = argv[-1]

    else:
        options['infile'] = argv[-1]
        for item in argv[1:-1]:
            if not item[0] == '-':
                continue
            if 'x' in item:
                if options['x_scale']:
                    options['x_scale'] = False
                else:
                    options['x_scale'] = True
            if 'y' in item:
                if options['y_scale']:
                    options['y_scale'] = False
                else:
                    options['y_scale'] = True
            if 'm' in item:
                options['type'] = 'min-max'
            if 'M' in item:
                options['type'] = 'mean'
            if 's' in item:
                options['type'] = 'stand'
            if 'h' in item:
                print_help(options, descrip)

    if '.' in options['infile']:
        options['name'] = options['infile'].split('.')[0]
    else:
        options['name'] = options['infile']

    if not options['x_scale']:
        X_COLS = []
    if not options['y_scale']:
        Y_COLS = []

    print "=" * 79
    print "Starting Scaling Code with Options:\n"
    for option in options:
        print option, '\t:', options[option]
    print "\nX_COLS:\n", X_COLS
    print "\nY_COLS:\n", Y_COLS
    print "-" * 79

    return options


def basic_scale(dat):
    """Scales the input vector arrays
    -1 : minimum value of the set
    1  : maximum value of the set
    """
    new_dat, vals, key = [], [], {}

    # Pull out the data
    print "     Pulling Statistics..."
    for item in dat:
        for idx, val in enumerate(item):
            try:
                vals[idx].append(val)
            except IndexError:
                vals.append([val])
    # Calculate the required statistics
    print "     Calculating Statistics..."
    for idx, set in enumerate(vals):
        label = 'col_{}'.format(idx)
        if label not in key:
            key[label] = {}
        key[label]['max']   = max(set)
        key[label]['min']   = min(set)
        key[label]['range'] = max(set) - min(set)
    # Scale the array
    print "     Scaling Arrays...\n"
    ldat = len(dat)
    for idx, par in enumerate(dat):
        cur = []
        printProgress(idx, ldat)
        for indx, val in enumerate(par):
            lbl = 'col_{}'.format(indx)
            nval = (val - key[lbl]['min']) / key[lbl]['range']
            cur.append((nval * 2) - 1)
        new_dat.append(cur)
    printProgress(ldat, ldat)
    print "\n"

    return np.array(new_dat), key


def minmax_scale(dat):
    """Scales the input vector arrays using min-max normalization
    0 = minimum value of the set
    1 = maximum value of the set
    """
    new_dat, vals, key = [], [], {}

    # Pull out the data
    print "     Pulling Statistics..."
    for item in dat:
        for idx, val in enumerate(item):
            try:
                vals[idx].append(val)
            except IndexError:
                vals.append([val])
    # Calculate the required statistics
    print "     Calculating Statistics..."
    for idx, set in enumerate(vals):
        label = 'col_{}'.format(idx)
        if label not in key:
            key[label] = {}
        key[label]['max']   = max(set)
        key[label]['min']   = min(set)
        key[label]['range'] = max(set) - min(set)
    # Scale the array
    print "     Scaling Arrays...\n"
    ldat = len(dat)
    for idx, par in enumerate(dat):
        cur = []
        printProgress(idx, ldat)
        for indx, val in enumerate(par):
            lbl = 'col_{}'.format(indx)
            nval = (val - key[lbl]['min']) / key[lbl]['range']
            cur.append(nval)
        new_dat.append(cur)
    printProgress(ldat, ldat)
    print "\n"

    return np.array(new_dat), key


def mean_scale(dat):
    """Scales the input vector arrays using mean normalizat
    0 = mean value of the set
    """
    new_dat, vals, key = [], [], {}

    # Pull out the data
    print "     Pulling Statistics..."
    for item in dat:
        for idx, val in enumerate(item):
            try:
                vals[idx].append(val)
            except IndexError:
                vals.append([val])
    # Calculate the required statistics
    print "     Calculating Statistics..."
    for idx, set in enumerate(vals):
        label = 'col_{}'.format(idx)
        if label not in key:
            key[label] = {}
        key[label]['max']   = max(set)
        key[label]['min']   = min(set)
        key[label]['mean']  = np.mean(set)
        key[label]['range'] = max(set) - min(set)
    # Scale the array
    print "     Scaling Arrays...\n"
    ldat = len(dat)
    for idx, par in enumerate(dat):
        cur = []
        printProgress(idx, ldat)
        for indx, val in enumerate(par):
            lbl = 'col_{}'.format(indx)
            nval = (val - key[lbl]['mean']) / key[lbl]['range']
            cur.append(nval)
        new_dat.append(cur)
    printProgress(ldat, ldat)
    print "\n"

    return np.array(new_dat), key


def stand_scale(dat):
    """Scales the input vector arrays using standardization
    0 = mean value of the set
    """
    new_dat, vals, key = [], [], {}

    # Pull out the data
    print "     Pulling Statistics..."
    for item in dat:
        for idx, val in enumerate(item):
            try:
                vals[idx].append(val)
            except IndexError:
                vals.append([val])
    # Calculate the required statistics
    print "     Calculating Statistics..."
    for idx, set in enumerate(vals):
        label = 'col_{}'.format(idx)
        if label not in key:
            key[label] = {}
        key[label]['max']   = max(set)
        key[label]['min']   = min(set)
        key[label]['mean']  = np.mean(set)
        key[label]['std']   = np.std(set)
        key[label]['range'] = max(set) - min(set)
    # Scale the array
    print "     Scaling Arrays...\n"
    ldat = len(dat)
    for idx, par in enumerate(dat):
        cur = []
        printProgress(idx, ldat)
        for indx, val in enumerate(par):
            lbl = 'col_{}'.format(indx)
            nval = (val - key[lbl]['mean']) / key[lbl]['std']
            cur.append(nval)
        new_dat.append(cur)
    printProgress(ldat, ldat)
    print "\n"
#    print new_dat
    return np.array(new_dat), key


def printProgress (iteration, total, prefix = 'Progress', suffix = 'Complete',
                   decimals = 2, barLength = 50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %.2f%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()


def import_file(options):
    """Imports data from file"""
    data = pandas.read_csv(options['infile'])
    return np.array(data[X_COLS]), np.array(data[Y_COLS])


def write_scaled(xdat, ydat, options):
    """Prints the scaled data to an output file"""
    fname = '{}_{}-Scaled_'.format(options['name'], options['type'])
    xtag, ytag = 'x0', 'y0'
    if options['x_scale']:
        xtag = 'x1'
    if options['y_scale']:
        ytag = 'y1'
    fname += xtag + '-' + ytag + '.csv'
    print "\n\n", '-' * 70
    print "Writing File: {}\n".format(fname)

    newfile = np.append(xdat, ydat, axis = 1)
    out = open(fname, 'w')
    for idx, val in enumerate(X_COLS):
        if idx == 0:
            out.write('{}'.format(val))
        else:
            out.write(',{}'.format(val))
    for idx, val in enumerate(Y_COLS):
        out.write(',{}'.format(val))
    out.write('\n')

    lenf = len(newfile)
    for idx, point in enumerate(newfile):
        printProgress(idx, lenf)
        for indx, val in enumerate(list(point)):
            if indx == 0:
                out.write('{}'.format(val))
            else:
                out.write(',{}'.format(val))
        out.write('\n')
        out.flush()
    printProgress(lenf, lenf)
    print "\n"

    out.close()


def save_key(x_key, y_key, options):
    """Saves the key to the scaling so that the values
    can be recovered"""
    state = {'x_key': x_key, 'y_key': y_key, 'options': options,
             'x_cols': X_COLS, 'y_cols': Y_COLS}
    fname = '{}_{}-ScaleKey_'.format(options['name'], options['type'])
    xtag, ytag = 'x0', 'y0'
    if options['x_scale']:
        xtag = 'x1'
    if options['y_scale']:
        ytag = 'y1'
    fname += xtag + '-' + ytag + '.pkl'
    print "Saving scale key to: {}".format(fname)
    out = open(fname, 'wb')
    pkl.dump(fname, out)
    out.flush()
    out.close()


def main():
    """Main Execution"""
    options = import_options()
    x_dat, y_dat = import_file(options)
    x_key, y_key = {}, {}
    if options['x_scale']:
        print "Scaling X-Values:"
        if options['type'] == 'basic':
            scaled_x, x_key = basic_scale(np.array(x_dat))
        if options['type'] == 'min-max':
            scaled_x, x_key = minmax_scale(np.array(x_dat))
        if options['type'] == 'mean':
            scaled_x, x_key = mean_scale(np.array(x_dat))
        if options['type'] == 'stand':
            scaled_x, x_key = stand_scale(np.array(x_dat))
        print "Done."
    else:
        scaled_x = x_dat
    if options['y_scale']:
        print "Scaling Y-Values:"
        if options['type'] == 'basic':
            scaled_y, y_key = basic_scale(np.array(y_dat))
        if options['type'] == 'min-max':
            scaled_y, y_key = minmax_scale(np.array(y_dat))
        if options['type'] == 'mean':
            scaled_y, y_key = mean_scale(np.array(y_dat))
        if options['type'] == 'stand':
            scaled_y, y_key = stand_scale(np.array(y_dat))
        print "Done."
    else:
        scaled_y = y_dat
    x_dat, y_dat = [], []

    #print scaled_x, scaled_y

    write_scaled(scaled_x, scaled_y, options)
    save_key(x_key, y_key, options)
    print "\nProgram Terminated Normally."
    print "=" * 79


if __name__ in '__main__':
    main()
