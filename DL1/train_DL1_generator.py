# -*- coding: utf-8 -*-
'''
Info:
    This script can be run directly after
    parallel_generate_data_DL1. It takes as inputs the 
    HDF5 files produced by the first script 
    and uses them to train a Keras NN à la DL1 but using
    a generator.
    # It also plots ROC curve comparisons with 
    # MV2c10 and saves them both in pickle and 
    # pdf format. 
Author: 
    Michela Paganini - Yale/CERN
    michela.paganini@cern.ch
'''

import pandas as pd
import pandautils as pup
import numpy as np
import math
import os
import sys
import deepdish.io as io
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Highway, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from viz import add_curve, calculate_roc, ROC_plotter
import cPickle

MODEL_NAME = '5l_moredata'
INPUT_NAME = '_large_'

def main(hdf5_paths, iptagger, n_train, n_test, n_validate):
    '''
    '''

    train_paths = [f for f in hdf5_paths if 'train' in f]
    test_paths = [f for f in hdf5_paths if 'test' in f]
    validate_paths = [f for f in hdf5_paths if 'validate' in f]

    def batch(paths, iptagger, batch_size, random=True):
        while True:
            if random:
                np.random.shuffle(paths)
            for fp in paths:
                d = io.load(fp)
                X = np.concatenate([d['X'], d[iptagger + '_vars']], axis=1)
                le = LabelEncoder()
                y = le.fit_transform(d['y'])
                w = d['w']
                if random:
                    ix = range(X.shape[0])
                    np.random.shuffle(ix)
                    X, y, w = X[ix], y[ix], w[ix]
                for i in xrange(int(np.ceil(X.shape[0] / float(batch_size)))):
                    yield X[(i * batch_size):((i+1)*batch_size)], y[(i * batch_size):((i+1)*batch_size)], w[(i * batch_size):((i+1)*batch_size)]

    def get_n_vars(train_paths, iptagger):
        # with open(train_paths[0], 'rb') as buf:
        #     d = io.load(buf)
        d = io.load(train_paths[0])
        return np.concatenate([d['X'], d[iptagger + '_vars']], axis=1).shape[1]

    net = Sequential()
    net.add(Dense(50, input_shape=(get_n_vars(train_paths, iptagger), ), activation='relu'))
    net.add(Dropout(0.3))
    net.add(Dense(40, activation='relu'))
    net.add(Dropout(0.2))
    net.add(Dense(16, activation='relu'))
    net.add(Dropout(0.1))
    net.add(Dense(16, activation='relu'))
    net.add(Dropout(0.1))
    net.add(Dense(4, activation='softmax'))

    net.summary()
    net.compile('adam', 'sparse_categorical_crossentropy')

    weights_path = './' + iptagger + '-' + MODEL_NAME + '-progress.h5'
    try:
        print 'Trying to load weights from ' + weights_path
        net.load_weights(weights_path)
        print 'Weights found and loaded from ' + weights_path
    except IOError:
        print 'Could not find weight in ' + weights_path

    # -- train 
    try:
        net.fit_generator(batch(train_paths, iptagger, 256, random=True),
        samples_per_epoch = n_train,
        verbose=True, 
        #batch_size=64, 
        #sample_weight=train['w'],
        callbacks = [
            EarlyStopping(verbose=True, patience=100, monitor='val_loss'),
            ModelCheckpoint(weights_path, monitor='val_loss', verbose=True, save_best_only=True)
        ],
        nb_epoch=200, 
        validation_data=batch(validate_paths, iptagger, 64, random=False),
        nb_val_samples=n_validate
        ) 
    except KeyboardInterrupt:
        print '\n Stopping early.'

    # -- load in best network
    print 'Loading best network...'
    net.load_weights(weights_path)

    print 'Extracting...'
    # # -- save the predicions
    #np.save('yhat-{}-{}.npy'.format(iptagger, MODEL_NAME), yhat)

    # from joblib import Parallel, delayed
    # test = Parallel(n_jobs=1, verbose=5, backend="threading")(
    #     delayed(extract)(filepath, ['pt', 'y', 'mv2c10']) for filepath in test_paths
    # )
    
    test = [extract(filepath, ['pt', 'y', 'mv2c10']) for filepath in test_paths]

    # -- test
    print 'Testing...'
    yhat = net.predict_generator(batch(test_paths, iptagger, 2048, random=False), val_samples=n_test)

    def dict_reduce(x, y):
        return {
            k: np.concatenate((v, y[k]))
            for k, v in x.iteritems()
        }
    test = reduce(dict_reduce, test)

    print 'Plotting...'
    _ = performance(yhat, test['y'], test['mv2c10'], iptagger)

    # -- Performance by pT
    print 'Plotting performance in bins of pT...'
    pt_bins = [10000, 50000, 100000, 150000, 200000, 300000, 500000, max(test['pt'])+1]
    bn = np.digitize(test['pt'], pt_bins)
    from collections import OrderedDict
    rej_at_70 = OrderedDict()

    for b in np.unique(bn):
        rej_at_70.update(
            performance(
                yhat[bn == b],
                test['y'][bn == b],
                test['mv2c10'][bn == b],
                iptagger,
                '{}-{}GeV'.format(pt_bins[b-1]/1000, pt_bins[b]/1000)
            )
        )

    # -- find center of each bin:
    bins_mean = [(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)]
    # -- horizontal error bars of lenght = bin length: 
    xerr = [bins_mean[i]-pt_bins[i+1] for i in range(len(bins_mean))]

    plt.clf()
    _ = plt.errorbar(
        bins_mean, 
        [rej_at_70[k]['DL1_70_bl'] for k in rej_at_70.keys()],
        xerr=xerr,
        #yerr=np.sqrt(bin_heights), 
        fmt='o', capsize=0, color='green', label='DL1' + iptagger, alpha=0.7)
    _ = plt.errorbar(
        bins_mean, 
        [rej_at_70[k]['MV2_70_bl'] for k in rej_at_70.keys()],
        xerr=xerr,
        #yerr=np.sqrt(bin_heights), 
        fmt='o', capsize=0, color='red', label='MV2c10', alpha=0.7)
    plt.legend()
    plt.title('b vs. l rejection at 70% efficiency in pT bins')
    plt.yscale('log')
    plt.xlabel(r'$p_{T, \mathrm{jet}} \ \mathrm{MeV}$')
    plt.ylabel('Background rejection at 70% efficiency')
    plt.xlim(xmax=1000000)
    plt.savefig('pt_bl.pdf')

    plt.clf()
    _ = plt.errorbar(
        bins_mean, 
        [rej_at_70[k]['DL1_70_bc'] for k in rej_at_70.keys()],
        xerr=xerr,
        #yerr=np.sqrt(bin_heights), 
        fmt='o', capsize=0, color='green', label='DL1' + iptagger, alpha=0.7)
    _ = plt.errorbar(
        bins_mean,
        [rej_at_70[k]['MV2_70_bc'] for k in rej_at_70.keys()],
        xerr=xerr,
        #yerr=np.sqrt(bin_heights), 
        fmt='o', capsize=0, color='red', label='MV2c10', alpha=0.7)
    plt.legend()
    plt.title('b vs. c rejection at 70% efficiency in pT bins')
    plt.xlabel(r'$p_{T, \mathrm{jet}} \ \mathrm{MeV}$')
    plt.ylabel('Background rejection at 70% efficiency')
    plt.yscale('log')
    plt.xlim(xmax=1000000)
    plt.savefig('pt_bc.pdf')

    
# -----------------------------------------------------------------

def extract(filepath, keys):
    # with open(filepath, 'rb') as buf:
    #     d = io.load(buf)
    d = io.load(filepath)
    new_d = {k:v for k,v in d.iteritems() if k in keys}
    return new_d

# ----------------------------------------------------------------- 

def performance(yhat, y, mv2c10, iptagger, extratitle=''):
    # -- Find flavors after applying cuts:
    bl_sel = (y == 5) | (y == 0)
    cl_sel = (y == 4) | (y == 0)
    bc_sel = (y == 5) | (y == 4)

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 0])[bl_sel])
    bl_curves = {}
    add_curve(r'DL1' + iptagger, 'green', 
          calculate_roc( y[bl_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 0])[bl_sel][fin1]),
          bl_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( y[bl_sel] == 5, mv2c10[bl_sel]),
          bl_curves)
    cPickle.dump(bl_curves, open('ROC_' + iptagger + '_' + MODEL_NAME + '_genprova_bl.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bl_curves, title=r'DL1 vs MV2c10 '+extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 10000)
    fg.savefig('ROC_' + iptagger + '_' + MODEL_NAME + '_' + extratitle +'_genprova_bl.pdf')

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 1])[bc_sel])
    bc_curves = {}
    add_curve(r'DL1' + iptagger, 'green', 
          calculate_roc( y[bc_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 1])[bc_sel][fin1]),
          bc_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( y[bc_sel] == 5, mv2c10[bc_sel]),
          bc_curves)
    cPickle.dump(bc_curves, open('ROC_' + iptagger + '_' + MODEL_NAME + '_genprova_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bc_curves, title=r'DL1 vs MV2c10 ' + extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 100)
    fg.savefig('ROC_' + iptagger + '_' + MODEL_NAME + '_' + extratitle +'_genprova_bc.pdf')
    plt.close(fg)

    def find_nearest(array, value):
        return (np.abs(array-value)).argmin()

    return {extratitle : 
        {
            'DL1_70_bl' : bl_curves[r'DL1' + iptagger]['rejection'][find_nearest(bl_curves[r'DL1' + iptagger]['efficiency'], 0.7)],
            'DL1_70_bc' : bc_curves[r'DL1' + iptagger]['rejection'][find_nearest(bc_curves[r'DL1' + iptagger]['efficiency'], 0.7)],
            'MV2_70_bl' : bl_curves[r'MV2c10']['rejection'][find_nearest(bl_curves[r'MV2c10']['efficiency'], 0.7)],
            'MV2_70_bc' : bc_curves[r'MV2c10']['rejection'][find_nearest(bc_curves[r'MV2c10']['efficiency'], 0.7)]
        }
    }

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs="+", help="Path to hdf5 files (e.g.: /path/to/pattern*.h5)")
    parser.add_argument('--iptagger', required=True, help="Select 'ip3d' or 'ipmp'")
    parser.add_argument('--n_train', required=True, type=int, help="int, Number of training examples")
    parser.add_argument('--n_test', required=True, type=int, help="int, Number of testing examples")
    parser.add_argument('--n_validate', required=True, type=int, help="int, Number of validating examples")
    args = parser.parse_args()

    sys.exit(main(args.input, args.iptagger, args.n_train, args.n_test, args.n_validate))
