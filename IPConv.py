from sklearn.preprocessing import StandardScaler
from keras.callbacks import Callback
from keras.layers import containers, LSTM, GRU
from keras.models import Sequential
from keras.layers.core import Highway, Dense, Dropout, AutoEncoder, MaxoutDense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.noise import GaussianNoise
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import np_utils
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Graph

from sklearn.metrics import roc_curve, auc

import pandautils as pup
import pandas as pd
import numpy as np
import os 

import deepdish.io as io
import sys

n_tracks = 30 # max number of tracks to consider per jet

# ------------------------------------------
'''
def sort_tracks(trk, data, SORT_COL, n_tracks):
    
    for i, jet in trk.iterrows(): # i = jet number, jet = all the variables for that jet
        
        if i % 10000 == 0:
            print 'Processing event %s of %s' % (i, trk.shape[0])
        
        # tracks = [[pt's], [eta's], ...] of tracks for each jet
        tracks = np.array([v.tolist() for v in jet.get_values()], dtype='float32')[:, (np.argsort(jet[SORT_COL]))[::-1]]
        
        ntrk = tracks.shape[1] # total number of tracks per jet
        
        data[i, :(min(ntrk, n_tracks)), :] = tracks.T[:(min(ntrk, n_tracks)), :] # take all tracks unless there are more than n_tracks
        
        data[i, (min(ntrk, n_tracks)):, :  ] = -999 # default value for missing tracks

# ------------------------------------------

def scale(data, n_variables):
    
    scale = {}
    for v in xrange(n_variables):
        print 'Scaling feature %s of %s.' % (v, n_variables)
        f = data[:, :, v]
        slc = f[f != -999]
        m, s = slc.mean(), slc.std()
        slc -= m
        slc /= s
        data[:, :, v][f != -999] = slc.astype('float32')
        scale[v] = {'mean' : m, 'sd' : s}

# ------------------------------------------

def process_data(trk):

    # -- classes
    y = trk.jet_truthflav.values
    ip3d = trk[ ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'] ] # new df with ip3d vars only

    trk.drop(['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_truthflav'], axis=1, inplace=True) # no longer needed - not an input
    trk['d0z0sig_unsigned'] = (trk.jet_trk_d0sig.copy() ** 2 + trk.jet_trk_z0sig.copy() ** 2).pow(0.5)
    #track_inputs += ['d0z0sig_unsigned']
    
    n_variables = trk.shape[1] #len(track_inputs) # or trk.shape[1]

    data = np.zeros((trk.shape[0], n_tracks, n_variables), dtype='float32')

    # -- variable being used to sort tracks
    SORT_COL = 'd0z0sig_unsigned'

    # -- call functions to build X (= data)
    sort_tracks(trk, data, SORT_COL, n_tracks)
    print 'Scaling features ...'
    scale(data, n_variables)

    # -- default values
    data[np.isnan(data)] = 0.0
    data[data == -999] = 0.0

    # -- make classes pretty for keras
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

    return {'X' : data, 'y' : y_train, 'ip3d' : ip3d}
'''
# ------------------------------------------

def build_graph(n_variables):

    nb_feature_maps = 64
    ngram_filters = [1, 2, 3, 4, 5] #, 6, 7, 8]

    graph = Graph()
    graph.add_input(name='data', input_shape= (n_tracks, n_variables))

    for n_gram in ngram_filters:
        sequential = containers.Sequential()
        sequential.add(Convolution1D(nb_feature_maps, n_gram, activation='relu', input_shape=(n_tracks, n_variables)))
        # sequential.add(MaxPooling1D(pool_length=n_tracks - n_gram + 1))
        sequential.add(GRU(25)) # GRU
        #sequential.add(Flatten())

        graph.add_node(sequential, name = 'unit_{}'.format(n_gram), input='data')

    graph.add_node(Dropout(0.4), name='dropout', inputs=['unit_{}'.format(n) for n in ngram_filters], create_output=True)

    return graph

# ------------------------------------------

def plot_ROC(y_test, yhat, ip3d, MODEL_FILE):
    
    from viz import calculate_roc, ROC_plotter, add_curve
    
    # -- bring classes back to usual format: [0,2,3,0,1,2,0,2,2,...]
    y = np.array([np.argmax(ev) for ev in y_test])

    # -- for b VS. light
    bl_sel = (y == 0) | (y == 2)
    # -- for c VS. light
    cl_sel = (y == 0) | (y == 1)

    # -- add ROC curves
    discs = {}

    add_curve(r'IP3D', 'black', calculate_roc((y[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ] == 2), 
                                              np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu)[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ]), 
              discs)

    
    add_curve(MODEL_FILE, 'blue', calculate_roc( (y[ bl_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] == 2), 
                                              np.log(yhat[:,2] / yhat[:,0])[bl_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] ), 
              discs)

    print 'Pickling ROC curves'
    import cPickle
    cPickle.dump(discs[MODEL_FILE], open(MODEL_FILE + '.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
#    cPickle.dump(discs['IP3D'], open('ip3d.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print 'Plotting'
    fg = ROC_plotter(discs, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, logscale=True)

    return fg

# ------------------------------------------
'''
if __name__ == '__main__':

    
    track_inputs = ['jet_trk_pt', 'jet_trk_phi', 'jet_trk_d0', 
                    'jet_trk_z0', 'jet_trk_d0sig', 'jet_trk_z0sig', 
                    'jet_trk_chi2', 'jet_trk_nInnHits', 
                    'jet_trk_nNextToInnHits', 'jet_trk_nBLHits', 
                    'jet_trk_nsharedBLHits', 'jet_trk_nsplitBLHits', 
                    'jet_trk_nPixHits', 'jet_trk_nsharedPixHits', 
                    'jet_trk_nsplitPixHits', 'jet_trk_nSCTHits', 
                    'jet_trk_nsharedSCTHits', 'jet_trk_expectBLayerHit']

    
    # -- import data into df called trk
    print 'Loading data...'
    if not (os.path.isfile('./data/train_dict_IPConv.h5') & os.path.isfile('./data/test_dict_IPConv.h5')):
    
        trk_train = pup.root2panda('./data/train/*410000_00*.root', 'JetCollection', branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'])
        trk_test  = pup.root2panda('./data/test/*410000*.root', 'JetCollection', branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'])
        # TEMP?
        #trk_val = pup.root2panda('./data/train/*01_v1.root', 'JetCollection', branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'])

        print 'Processing training sample ...'
        train_dict = process_data(trk_train)
        del trk_train
        io.save('./data/train_dict_IPConv.h5', train_dict)

        print 'Processing test sample...'
        test_dict = process_data(trk_test)
        del trk_test
        io.save('./data/test_dict_IPConv.h5', test_dict)
    '''

#    else:

def main(MODEL_FILE):

    test_dict = io.load('./data/test_dict_IPConv.h5')
    train_dict = io.load('./data/train_dict_IPConv.h5')

    X_train = train_dict['X']
    y_train = train_dict['y']
    n_features = X_train.shape[2]    
    
    # print 'Processing validation sample ...'
    # val_dict = process_data(trk_val)
    # X_val = val_dict['X']
    # y_val = val_dict['y']
    # io.save('./data/val_dict_IPConv.h5', val_dict)
    # validation_data = (X_val, y_val)
    # print validation_data

    X_test = test_dict['X']
    y_test = test_dict['y']
    ip3d = test_dict['ip3d'] # this is a df


    print 'Building model...'
    
    if (MODEL_FILE == 'CRNN'):
        graph = build_graph(n_features) #trk_train.shape[1])

        model = Sequential()

        model.add(graph)
        
        #model.add(MaxoutDense(64, 5, input_shape=graph.nodes['dropout'].output_shape[1:]))
        model.add(Dense(64))

    elif (MODEL_FILE == 'RNN'):

        model = Sequential()
        model.add(GRU(25, input_shape=(n_tracks, n_features))) #GRU
        model.add(Dropout(0.2))
    
        # removing because of tensorflow
        #model.add(MaxoutDense(64, 5))  #, input_shape=graph.nodes['dropout'].output_shape[1:]))
        model.add(Dense(64))

  
    model.add(Dropout(0.4))

    model.add(Highway(activation = 'relu'))

    model.add(Dropout(0.3))
    model.add(Dense(4))

    model.add(Activation('softmax'))

    print 'Compiling model...'
    model.compile('adam', 'categorical_crossentropy')
    model.summary()

    print 'Training:'
    try:
        model.fit(X_train, y_train, batch_size=512,
            callbacks = [
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(MODEL_FILE + '-progress', monitor='val_loss', verbose=True, save_best_only=True)
            ],
        nb_epoch=2, 
        validation_split = 0.2, 
        show_accuracy=True) 
        
    except KeyboardInterrupt:
        print 'Training ended early.'

    # -- load in best network
    model.load_weights(MODEL_FILE + '-progress')
    

    print 'Saving protobuf'
#    gd = model.as_graph_def()
    # write out to a new directory called models
    # the actual graph file is graph.pb
    # the graph def is in the global session
    import tensorflow as tf
    import keras.backend.tensorflow_backend as tfbe

    sess = tfbe._SESSION
    saver = tf.train.Saver()
    tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb', as_text=False)    
#    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model-weights.ckpt")
    print "Model saved in file: %s" % save_path
    
    print saver.as_saver_def().filename_tensor_name
    print saver.as_saver_def().restore_op_name
#    print type(saver.as_saver_def().SerializeToString())

    print model.get_output()
    print 'Saving weights...'
    model.save_weights('./weights/ip3d-replacement_' + MODEL_FILE + '.h5', overwrite=True)

    json_string = model.to_json()
    open(MODEL_FILE + '.json', 'w').write(json_string)

    print 'Testing...'
    yhat = model.predict(X_test, verbose = True, batch_size = 512) 

    print 'Plotting ROC...'
    fg = plot_ROC(y_test, yhat, ip3d, MODEL_FILE)
    #plt.show()
    fg.savefig('./plots/roc_' + MODEL_FILE + '.pdf')
        
# ------------------------------------------

if __name__ == '__main__': 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="type of architecture: CRNN or RNN")
   
    args = parser.parse_args()

    if not( (args.model == 'CRNN') or (args.model == 'RNN') ):
        sys.exit('Error: Unknown model. Pick: CRNN or RNN')
    else:
        sys.exit( main(args.model) )





















