'''
dataprocessing.py -- utilities for processing track level ntuples
'''
import json
import numpy as np
import pandas as pd
import deepdish.io as io
import keras
from keras.utils import np_utils
import pandautils as pup

N_TRACKS = 30

def sort_tracks(trk, data, SORT_COL, n_tracks):
    ''' 
    sort the tracks from trk --> data
    
    Args:
    -----
        trk: a dataframe
        data: an array of shape (nb_samples, nb_tracks, nb_features)
        SORT_COL: a string representing the column to sort the tracks by
        n_tracks: number of tracks to cut off at. if >, truncate, else, -999 pad
    
    Returns:
    --------
        modifies @a data in place. Pads with -999
    
    '''
    # i = jet number, jet = all the variables for that jet 
    for i, jet in trk.iterrows(): 

        if i % 10000 == 0:
                print 'Processing event %s of %s' % (i, trk.shape[0])

        # tracks = [[pt's], [eta's], ...] of tracks for each jet 
        tracks = np.array(
                [v.tolist() for v in jet.get_values()], 
                dtype='float32'
            )[:, (np.argsort(jet[SORT_COL]))[::-1]]

        # total number of tracks per jet      
        ntrk = tracks.shape[1] 

        # take all tracks unless there are more than n_tracks 
        data[i, :(min(ntrk, n_tracks)), :] = tracks.T[:(min(ntrk, n_tracks)), :] 

        # default value for missing tracks 
        data[i, (min(ntrk, n_tracks)):, :  ] = -999 


def scale(data, var_names):
    ''' 
    Args:
    -----
        data: a numpy array of shape (nb_samples, nb_tracks, n_variables)
        n_variables: self explanatory
    
    Returns:
    --------
        modifies data in place
    '''
    
    scale = {}
    for v in xrange(len(var_names)):
        print 'Scaling feature %s of %s.' % (v, len(var_names))
        f = data[:, :, v]
        slc = f[f != -999]
        m, s = slc.mean(), slc.std()
        slc -= m
        slc /= s
        data[:, :, v][f != -999] = slc.astype('float32')
        scale[v] = {'name' : var_names[v], 'mean' : m, 'sd' : s}
    return scale

def process_data(trk, savevars=False):
    ''' 
    takes a dataframe directly from ROOT files and 
    processes, sorts, and scales the data for you
    
    Args:
    -----
        trk: the dataframe in question
    
    Returns:
    --------
        a dictionary with keys = {'X', 'y', and 'ip3d'}
    
    '''
    
    # -- targets                                                                                                                                                                                                 
    y = trk.jet_truthflav.values

    # -- new df with ip3d vars only for later comparison
    ip3d = trk[ ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'] ] 

    # -- add new variables to the df as input features
    # replace jet_trk_phi with jet_trk_DPhi 
    trk['jet_trk_DPhi'] = trk['jet_trk_phi'] - trk['jet_phi']
    # variable also used for ordering tracks
    trk['d0z0sig_unsigned'] = (trk.jet_trk_d0sig.copy() ** 2 + trk.jet_trk_z0sig.copy() ** 2).pow(0.5)

    # -- drop variables from the df that we do not want to use for training
    trk.drop(['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_truthflav', 'jet_trk_phi', 'jet_phi'], axis=1, inplace=True) # no longer needed - not an input                                                                                                                                                                                                                                               

    n_variables = trk.shape[1]
    var_names = trk.keys()

    data = np.zeros((trk.shape[0], N_TRACKS, n_variables), dtype='float32')

    # -- variable being used to sort tracks                                                                                                                                                                      
    SORT_COL = 'd0z0sig_unsigned'

    # -- call functions to build X (= data)                                                                                                                                                                      
    sort_tracks(trk, data, SORT_COL, N_TRACKS)
    
    print 'Scaling features ...'
    scale_params = scale(data, var_names) # dictionary with var name, mean and sd

    # -- default values    
    # we convert the default -999 --> 0 for padding purposes.                                                                                                                                                                                      
    data[np.isnan(data)] = 0.0
    data[data == -999] = 0.0

    # -- make classes pretty for keras (4 classes: b vs c vs l vs tau)                                                                                                                                                                          
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

    if savevars:
        # -- write variable json for lwtnn keras2json converter
        variable_dict = {
            'inputs': [{
                'name': scale_params[v]['name'],
                'scale': float(1.0 / scale_params[v]['sd']),
                'offset': float(-scale_params[v]['mean']),
                'default': None
                } 
                for v in xrange(n_variables)],
            'class_labels': np.unique(y).tolist(),
            'keras_version': keras.__version__
        }
        print variable_dict
        with open('variables.json', 'wb') as jf:
            json.dump(variable_dict, jf)

    return {'X' : data, 'y' : y_train, 'ip3d' : ip3d}

# ------------------------------------------   

if __name__ == '__main__':

    track_inputs = ['jet_trk_pt', 'jet_trk_d0',
                    'jet_trk_z0', 'jet_trk_d0sig', 'jet_trk_z0sig',
                    'jet_trk_chi2', 'jet_trk_nInnHits',
                    'jet_trk_nNextToInnHits', 'jet_trk_nBLHits',
                    'jet_trk_nsharedBLHits', 'jet_trk_nsplitBLHits',
                    'jet_trk_nPixHits', 'jet_trk_nsharedPixHits',
                    'jet_trk_nsplitPixHits', 'jet_trk_nSCTHits',
                    'jet_trk_nsharedSCTHits', 'jet_trk_expectBLayerHit'] # 2 more to be added in `process_data`

    print 'Loading dataframes...'
    # -- currently only training and testing on one file each!
    trk_train = pup.root2panda(
        './data/train/*410000_00*.root', 
        'JetCollection', 
        branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_phi', 'jet_trk_phi']
    )

    trk_test  = pup.root2panda(
        './data/test/*410000*.root', 
        'JetCollection', 
        branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_phi', 'jet_trk_phi']
    )

    print 'Processing training sample ...'
    train_dict = process_data(trk_train, savevars=True)
    del trk_train
    io.save('./data/train_dict_IPConv.h5', train_dict)

    print 'Processing test sample...'
    test_dict = process_data(trk_test)
    del trk_test
    io.save('./data/test_dict_IPConv.h5', test_dict)




