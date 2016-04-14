'''
dataprocessing.py -- utilities for processing track level ntuples
'''
import numpy as np
import pandas as pd
import deepdish.io as io
from keras.utils import np_utils
import pandautils as pup


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


def scale(data, n_variables):
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
    for v in xrange(n_variables):
        print 'Scaling feature %s of %s.' % (v, n_variables)
        f = data[:, :, v]
        slc = f[f != -999]
        m, s = slc.mean(), slc.std()
        slc -= m
        slc /= s
        data[:, :, v][f != -999] = slc.astype('float32')
        scale[v] = {'mean' : m, 'sd' : s}


def process_data(trk):
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
    
    # -- classes                                                                                                                                                                                                 
    y = trk.jet_truthflav.values

    # new df with ip3d vars only
    ip3d = trk[ ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'] ] 


    trk.drop(['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_truthflav'], axis=1, inplace=True) # no longer needed - not an input                                                                             
    
    trk['d0z0sig_unsigned'] = (trk.jet_trk_d0sig.copy() ** 2 + trk.jet_trk_z0sig.copy() ** 2).pow(0.5)                                                                                                                                                                  

    n_variables = trk.shape[1]

    data = np.zeros((trk.shape[0], n_tracks, n_variables), dtype='float32')

    # -- variable being used to sort tracks                                                                                                                                                                      
    SORT_COL = 'd0z0sig_unsigned'

    # -- call functions to build X (= data)                                                                                                                                                                      
    sort_tracks(trk, data, SORT_COL, n_tracks)
    
    print 'Scaling features ...'
    scale(data, n_variables)

    # -- default values    
    # we convert the default -999 --> 0 for padding purposes.                                                                                                                                                                                      
    data[np.isnan(data)] = 0.0
    data[data == -999] = 0.0

    # -- make classes pretty for keras                                                                                                                                                                           
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

    return {'X' : data, 'y' : y_train, 'ip3d' : ip3d}

# ------------------------------------------   

if __name__ == '__main__':

    track_inputs = ['jet_trk_pt', 'jet_trk_phi', 'jet_trk_d0',
                    'jet_trk_z0', 'jet_trk_d0sig', 'jet_trk_z0sig',
                    'jet_trk_chi2', 'jet_trk_nInnHits',
                    'jet_trk_nNextToInnHits', 'jet_trk_nBLHits',
                    'jet_trk_nsharedBLHits', 'jet_trk_nsplitBLHits',
                    'jet_trk_nPixHits', 'jet_trk_nsharedPixHits',
                    'jet_trk_nsplitPixHits', 'jet_trk_nSCTHits',
                    'jet_trk_nsharedSCTHits', 'jet_trk_expectBLayerHit']

    # -- currently only training and testing on one file each!
    trk_train = pup.root2panda(
        './data/train/*410000_00*.root', 
        'JetCollection', 
        branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc']
    )

    trk_test  = pup.root2panda(
        './data/test/*410000*.root', 
        'JetCollection', 
        branches = track_inputs + ['jet_truthflav' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc']
    )

    print 'Processing training sample ...'
    train_dict = process_data(trk_train)
    del trk_train
    io.save('./data/train_dict_IPConv.h5', train_dict)

    print 'Processing test sample...'
    test_dict = process_data(trk_test)
    del trk_test
    io.save('./data/test_dict_IPConv.h5', test_dict)




