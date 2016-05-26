'''
full_ntuple_processing.py -- utilities for processing event level ntuples for IPRNN
LOOSE TRACK SELECTION CURRENTLY HARDCODED IN! TO BE REMOVED!
'''
import json
import numpy as np
import pandas as pd
import deepdish.io as io
import keras
from keras.utils import np_utils
import pandautils as pup

N_TRACKS = 30
VAR_FILE_NAME = 'variables_ntuple_MyTrkSel.json'

def flatten(trk):
    newdf = pd.DataFrame()
    for key in trk.keys():
        newdf[key] = pup.flatten(trk[key])
    return newdf

def applycuts(trk):
    cuts = (abs(trk['jet_eta']) < 2.5) & \
           (trk['jet_pt'] > 20e3) & \
           ((trk['jet_JVT'] > 0.59) | (trk['jet_pt'] > 60e3) | (abs(trk['jet_eta']) > 2.4)) & \
           (trk['jet_aliveAfterOR'] == 1)

    z0 = trk['jet_trk_ip3d_z0']
    notracks = []
    for i in xrange(len(z0)):
        notracks.append(len(z0[i]) == 0)
    hastracks = -np.array(notracks)

    trk = trk[(cuts & hastracks) == 1]
    trk = trk.reset_index(drop=True)
    return trk

def dphi(trk_phi, jet_phi):

    import math
    PI = math.pi
    deltaphi = trk_phi - jet_phi

    for evN in xrange(len(deltaphi)):
        deltaphi[evN][deltaphi[evN] > PI] -= 2*PI
        deltaphi[evN][deltaphi[evN] < -PI] += 2*PI

    return deltaphi

def athenaname(var):
    return var.replace('jet_trk_', '')

def sort_tracks(trk, data, theta, SORT_COL, n_tracks):
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
    #for i, (_, jet) in enumerate(trk.iterrows()): 
    for i, jet in trk.iterrows():

        if i % 10000 == 0:
                print 'Processing event %s of %s' % (i, trk.shape[0])

        # -- LOOSE TRACK SELECTION ENFORCED!
        # pt > 400, numberOfPixelHits + numberOfSCTHits >= 7, numberOfPixelHits >= 1, abs(d0) <=2, abs(z0 * sin(theta)) <=3
        trk_selection = (trk['jet_trk_pt'][i] > 400) & \
                    ((trk['jet_trk_nPixHits'][i] + trk['jet_trk_nSCTHits'][i]) >= 7) & \
                    (trk['jet_trk_nPixHits'][i] >= 1) & \
                    (abs(trk['jet_trk_ip3d_d0'][i]) <= 2) & \
                    (abs(trk['jet_trk_ip3d_z0'][i] * np.sin(theta[i])) <=3 )

        # tracks = [[pt's], [eta's], ...] of tracks for each jet 
        tracks = np.array(
                [v[trk_selection].tolist() for v in jet.get_values()], 
                dtype='float32'
            )[:, (np.argsort(jet[SORT_COL][trk_selection]))[::-1]]

        # total number of tracks per jet      
        ntrk = tracks.shape[1] 

        # take all tracks unless there are more than n_tracks 
        data[i, :(min(ntrk, n_tracks)), :] = tracks.T[:(min(ntrk, n_tracks)), :] 

        # default value for missing tracks 
        data[i, (min(ntrk, n_tracks)):, :  ] = -999 


def scale(data, var_names, savevars):
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
    if savevars: 
        for v in xrange(len(var_names)):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), var_names[v])
            f = data[:, :, v]
            slc = f[f != -999]
            m, s = slc.mean(), slc.std()
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')
            scale[v] = {'name' : athenaname(var_names[v]), 'mean' : m, 'sd' : s}

    else:
        import json
        with open(VAR_FILE_NAME, 'rb') as varfile:
            varinfo = json.load(varfile)

        for v in xrange(len(var_names)):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), var_names[v])
            f = data[:, :, v]
            slc = f[f != -999]
            ix = [i for i in xrange(len(varinfo['inputs'])) if varinfo['inputs'][i]['name'] == athenaname(var_names[v])]
            offset = varinfo['inputs'][ix[0]]['offset']
            scale = varinfo['inputs'][ix[0]]['scale']
            slc += offset
            slc *= scale
            data[:, :, v][f != -999] = slc.astype('float32')

    return scale

def process_data(trk, cut_vars, savevars=False):
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
    trk = applycuts(flatten(trk))

    # -- targets
    y = trk.jet_LabDr_HadF

    # -- new df with ip3d vars only for later comparison
    ip3d = trk[ ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc'] ] 

    # -- add new variables to the df as input features
    # replace jet_trk_phi with jet_trk_DPhi 
    trk['jet_trk_DPhi'] = dphi(trk['jet_trk_phi'], trk['jet_phi'])

    # variable also used for ordering tracks
    trk['d0z0sig'] = (trk.jet_trk_ip3d_d0sig.copy() ** 2 + trk.jet_trk_ip3d_z0sig.copy() ** 2).pow(0.5)

    # -- separate theta from the other variables because we don't train on it. Ugly hack :((
    theta = trk['jet_trk_theta']

    # -- drop variables from the df that we do not want to use for training
    trk.drop(cut_vars + ['jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_LabDr_HadF', 'jet_trk_phi', 'jet_phi', 'jet_trk_theta'], axis=1, inplace=True) # no longer needed - not an input               

    n_variables = trk.shape[1] 
    var_names = trk.keys()

    data = np.zeros((trk.shape[0], N_TRACKS, n_variables), dtype='float32')

    # -- variable being used to sort tracks                                                                                                                                                                      
    SORT_COL = 'd0z0sig'

    # -- call functions to build X (= data)                                                                                                                                                                      
    sort_tracks(trk, data, theta, SORT_COL, N_TRACKS)
    del theta
    
    print 'Scaling features ...'
    #scale_params = scale(data, var_names) # dictionary with var name, mean and sd
    scale_params = scale(data, var_names, savevars)

    # -- default values                                                                                                                                                                                          
    data[np.isnan(data)] = -999
    #data[data == -999] = 0.0

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
            'class_labels': ['pu', 'pc', 'pb', 'ptau'],
            'keras_version': keras.__version__,
            'miscellaneous': {
                'sort_by': SORT_COL
            }
        }
        with open(VAR_FILE_NAME, 'wb') as jf:
            json.dump(variable_dict, jf)

    return {'X' : data, 'y' : y_train, 'ip3d' : ip3d}

# ------------------------------------------   

if __name__ == '__main__':

    track_inputs = ['jet_trk_pt', 'jet_trk_ip3d_d0',
                    'jet_trk_ip3d_z0', 'jet_trk_ip3d_d0sig', 'jet_trk_ip3d_z0sig',
                    'jet_trk_chi2', 'jet_trk_nInnHits',
                    'jet_trk_nNextToInnHits', 'jet_trk_nBLHits',
                    'jet_trk_nsharedBLHits', 'jet_trk_nsplitBLHits',
                    'jet_trk_nPixHits', 'jet_trk_nsharedPixHits',
                    'jet_trk_nsplitPixHits', 'jet_trk_nSCTHits',
                    'jet_trk_nsharedSCTHits', 'jet_trk_expectBLayerHit', 
                    #'jet_trk_dPhi'] # more to be added in `process_data`
                    'jet_trk_phi'] # more to be added in `process_data`

    cut_vars = ['jet_eta', 'jet_pt', 'jet_JVT', 'jet_aliveAfterOR'] # only necessary to remove bad jets

     # -- load and process training set
    print 'Loading training dataframe...'
    trk_train = pup.root2panda(
        './data/Dan/NOtrkSel/train_NOtrkSel.root', 
        'bTag_AntiKt4EMTopoJets', 
        branches = track_inputs + cut_vars + ['jet_LabDr_HadF' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_phi', 'jet_trk_theta']
    )
    print 'Processing training sample ...'
    train_dict = process_data(trk_train, cut_vars, savevars=True)
    del trk_train
    io.save('./data/train_dict_IPConv_ntuple_MyTrkSel.h5', train_dict)

    # -- load and process test set
    print 'Loading test dataframe...'
    trk_test  = pup.root2panda(
        './data/Dan/NOtrkSel/test/user.dguest.8493098.Akt4EMTo._000013_NOtrkSel.root', 
        'bTag_AntiKt4EMTopoJets', 
        branches = track_inputs + cut_vars + ['jet_LabDr_HadF' , 'jet_ip3d_pu', 'jet_ip3d_pb', 'jet_ip3d_pc', 'jet_phi', 'jet_trk_theta']
    )
    print 'Processing test sample...'
    test_dict = process_data(trk_test, cut_vars, savevars=False)
    del trk_test
    io.save('./data/test_dict_IPConv_ntuple_MyTrkSel.h5', test_dict)


    



