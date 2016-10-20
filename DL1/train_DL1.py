# -*- coding: utf-8 -*-
'''
Info:
    This script can be run directly after
    generate_data_DL1. It takes as inputs the 
    HDF5 files produced by the first script 
    and uses them to train a Keras NN à la DL1.
    It also plots ROC curve comparisons with 
    MV2c10 and saves them both in pickle and 
    pdf format. 
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

def main(iptagger):
	train = io.load(open(os.path.join('..', 'data', 'DL1-' + iptagger + INPUT_NAME + '-train-db.h5'), 'rb'))
	le = LabelEncoder()

	net = Sequential()

	net.add(Dense(50, input_shape=(train['X'].shape[1], ), activation='relu'))
	net.add(Dropout(0.3))

	net.add(Dense(40, activation='relu'))
	net.add(Dropout(0.2))

	net.add(Dense(16, activation='relu'))
	net.add(Dropout(0.1))

	net.add(Dense(16, activation='relu'))
	net.add(Dropout(0.1))

	net.add(Dense(4, activation='softmax'))

	net.summary()
	net.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

	weights_path = iptagger + '-' + MODEL_NAME + '-progress.h5'
	try:
		print 'Trying to load weights from ' + weights_path
		net.load_weights(weights_path)
		print 'Weights found and loaded from ' + weights_path
	except IOError:
		print 'Could not find weight in ' + weights_path

	# -- train 
	try:
		net.fit(train['X'], le.fit_transform(train['y']),
		verbose=True, 
		batch_size=64, 
		sample_weight=train['w'],
		callbacks = [
		    EarlyStopping(verbose=True, patience=100, monitor='val_loss'),
		    ModelCheckpoint(weights_path, monitor='val_loss', verbose=True, save_best_only=True)
		],
		nb_epoch=200, 
		validation_split=0.3
		) 
	except KeyboardInterrupt:
		print '\n Stopping early.'

	# -- load in best network
	net.load_weights(weights_path)

	# -- test
	print 'Testing...'
	test = io.load(open(os.path.join('..', 'data', 'DL1-' + iptagger + INPUT_NAME + '-test-db.h5'), 'rb'))
	#test = io.load(open(os.path.join('..', 'data', 'DL1-' + iptagger + '_large_' + '-test-db.h5'), 'rb'))

	yhat = net.predict(test['X'], verbose=True) 

	# -- save the predicions
	np.save('yhat-{}-{}.npy'.format(iptagger, MODEL_NAME), yhat)

	performance(yhat, test, iptagger)

# ----------------------------------------------------------------- 

def performance(yhat, test, iptagger):
	# -- Find flavors after applying cuts:
	bl_sel = (test['y'] == 5) | (test['y'] == 0)
	cl_sel = (test['y'] == 4) | (test['y'] == 0)
	bc_sel = (test['y'] == 5) | (test['y'] == 4)

	fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 0])[bl_sel])
	bl_curves = {}
	add_curve(r'DL1' + iptagger, 'green', 
	      calculate_roc( test['y'][bl_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 0])[bl_sel][fin1]),
	      bl_curves)
	add_curve(r'MV2c10', 'red', 
	      calculate_roc( test['y'][bl_sel] == 5, test['mv2c10'][bl_sel]),
	      bl_curves)
	cPickle.dump(bl_curves, open('ROC_' + iptagger + '_' + MODEL_NAME + '_bl.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

	fg = ROC_plotter(bl_curves, title=r'DL1 vs MV2c10', min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 10000)
	fg.savefig('ROC_' + iptagger + '_' + MODEL_NAME + '_bl.pdf')

	fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 1])[bc_sel])
	bc_curves = {}
	add_curve(r'DL1' + iptagger, 'green', 
	      calculate_roc( test['y'][bc_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 1])[bc_sel][fin1]),
	      bc_curves)
	add_curve(r'MV2c10', 'red', 
	      calculate_roc( test['y'][bc_sel] == 5, test['mv2c10'][bc_sel]),
	      bc_curves)
	cPickle.dump(bc_curves, open('ROC_' + iptagger + '_' + MODEL_NAME + '_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

	fg = ROC_plotter(bc_curves, title=r'DL1 vs MV2c10', min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 100)
	fg.savefig('ROC_' + iptagger + '_' + MODEL_NAME + '_bc.pdf')

# ----------------------------------------------------------------- 


if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('iptagger', help="select 'ip3d' or 'ipmp'")
    args = parser.parse_args()

    sys.exit(main(args.iptagger))