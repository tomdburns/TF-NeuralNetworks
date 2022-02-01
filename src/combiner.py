#!/usr/bin/env python

"""
This is used when a large set of data is being processed
and allows the user to save time by simply combining
file instead of re-running the SCALER program for different
combinations
"""


import pandas
from sys import argv
from glob import glob

#if len(argv) != 3:
#    print "to run: tf_combiner XFILE YFILE"
#    exit()

script, file = argv


X_COLS = ['b0_c', 'd0_c', 'q1_c', 'q2_c', 'U1_c', 'U2_c', 'b0_n',
          'd0_n', 'q1_n', 'q2_n', 'U1_n', 'U2_n', 'StructuredDensity',
          'InletTemp', 'tads', 'tblow', 'tevac', 'Pint', 'Plow', 'v0']
Y_COLS = ['Purity', 'Recovery', 'Productivity', 'Etot', 'EPSA', 'EComp']


def main():
    """Main Execution"""
    name = file.split('.csv')[0]
    dat1 = pandas.read_csv(file)
    others = glob('*{}*.csv'.format(name))
    all = []
    for col in dat1:
        print col
    for other in others:
        if file in other:
            continue
        type = other.split('{}_'.format(name))[1].split('-Scaled')[0]
        cur = pandas.read_csv(other)
        new_cols, n_cols = {}, []
        for col in cur:
#            print col, '{}_{}'.format(col, type)
            if '{}_{}'.format(col, type) in all:
                print '{}_{}'.format(col, type)
            else:
                all.append('{}_{}'.format(col, type))
                new_cols[col] = '{}_{}'.format(col, type)
                n_cols.append('{}_{}'.format(col, type)) 
        new = cur.rename(columns = new_cols)
        print "-" * 20
        for col in new:
            print col
        print "-" * 20
        try:
            dat1[n_cols] = new[n_cols]
        except:
            print type
    dat1.to_csv('{}_scaled.csv'.format(name))

    


if __name__ in '__main__':
    main()
