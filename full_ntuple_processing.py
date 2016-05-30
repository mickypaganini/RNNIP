'''
full_ntuple_processing.py -- utilities for processing event level ntuples for IPRNN
'''
import json
import numpy as np
import pandas as pd
import deepdish.io as io
import keras
from keras.utils import np_utils
import pandautils as pup

N_TRACKS = 30
VAR_FILE_NAME = 'variables_ntuple.json'

def flatten(event_df):
     '''
    Definition:
    -----------
        Turn event-flat dataframe into jet-flat dataframe
        Note: no event-level branches should be present in event_df
    Args:
    -----
        event_df: event-flat pandas dataframe
    
    Returns:
    --------
        jet_df: jet_flat pandas dataframe 
    '''
    jet_df = pd.DataFrame()
    for key in event_df.keys():
        jet_df[key] = pup.flatten(event_df[key])
    return jet_df


def applycuts(df):
    '''
    Definition:
    -----------
        remove bad jets from df -- those with 0 tracks and those that don't pass b-tagging cuts
    Args:
    -----
        df: jet-flat pandas dataframe
    
    Returns:
    --------
        df: modified jet_flat pandas dataframe where bad jets have been removed
    '''
    # -- b-tagging cuts for AntiKt4EMTopoJets
    cuts = (abs(df['jet_eta']) < 2.5) & \
           (df['jet_pt'] > 20e3) & \
           ((df['jet_JVT'] > 0.59) | (df['jet_pt'] > 60e3) | (abs(df['jet_eta']) > 2.4)) & \
           (df['jet_aliveAfterOR'] == 1)

    # -- remove jets with 0 tracks
    z0 = df['jet_trk_ip3d_z0']
    notracks = []
    for i in xrange(len(z0)):
        notracks.append(len(z0[i]) == 0)
    hastracks = -np.array(notracks)

    # -- apply cuts and reset df indices
    df = df[(cuts & hastracks) == 1]
    df = df.reset_index(drop=True)
    return df

def dphi(trk_phi, jet_phi):
    '''
    Definition:
    -----------
        Calculate Delta Phi between track and jet
        The result will be in the range [-pi, +pi]
    Args:
    -----
        trk_phi: series or np.array with the list of track phi's for each jet
        jet_phi: series of np.array with the phi value for each jet
    
    Returns:
    --------
        deltaphi: np.array with the list of values of delta phi between the jet and each track
    '''

    import math
    PI = math.pi
    deltaphi = trk_phi - jet_phi # automatically broadcasts jet_phi across all tracks

    # -- ensure that phi stays within -pi and pi
    for jetN in xrange(len(deltaphi)):
        deltaphi[jetN][deltaphi[jetN] > PI] -= 2*PI
        deltaphi[jetN][deltaphi[jetN] < -PI] += 2*PI

    return deltaphi

def athenaname(var):
    '''
    Definition:
    -----------
        Quick utility function to turn variable names into Athena format
        This function may vary according to the names of the variables currently being used
    
    Args:
    -----
        var: a string with the variable name

    Returns:
    --------
        a string with the modified variable name
    '''
    return var.replace('jet_trk_ip3d_', '').replace('jet_trk_', '')


def sort_tracks(trk, data, theta, SORT_COL, n_tracks):
    ''' 
    Definition:
        Sort tracks by SORT_COL and put them into an ndarray called data.
        Pad missing tracks with -999 --> net will have to have Masking layer
    
    Args:
    -----
        trk: a dataframe
        data: an array of shape (nb_samples, nb_tracks, nb_features)
        theta: pandas series with the jet_trk_theta values, used to enforce custom track selection
        SORT_COL: a string representing the column to sort the tracks by
        n_tracks: number of tracks to cut off at. if >, truncate, else, -999 pad
    
    Returns:
    --------
        modifies @a data in place. Pads with -999
    
    '''
    
    for i, jet in trk.iterrows():

        if i % 10000 == 0:
                print 'Processing event %s of %s' % (i, trk.shape[0])

        # # -- CUSTOM LOOSE TRACK SELECTION ENFORCED!
        # # pt > 400, numberOfPixelHits + numberOfSCTHits >= 7, numberOfPixelHits >= 1, abs(d0) <=2, abs(z0 * sin(theta)) <=3
        # trk_selection = (trk['jet_trk_pt'][i] > 400) & \
        #             ((trk['jet_trk_nPixHits'][i] + trk['jet_trk_nSCTHits'][i]) >= 7) & \
        #             (trk['jet_trk_nPixHits'][i] >= 1) & \
        #             (abs(trk['jet_trk_ip3d_d0'][i]) <= 2) & \
        #             (abs(trk['jet_trk_ip3d_z0'][i] * np.sin(theta[i])) <=3 )

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


def scale(data, var_names, SORT_COL, savevars):
    ''' 
    Args:
    -----
        data: a numpy array of shape (nb_samples, nb_tracks, n_variables)
        var_names: list of keys to be used for the model
        SORT_COL: string with the name of the column used for sorting tracks
                  needed to return json file in format requested by keras2json.py
        savevars: bool -- True for training, False for testing
                  it decides whether we want to fit on data to find mean and std 
                  or if we want to use those stored in the json file 

    Returns:
    --------
        modifies data in place
    '''
    import json
    
    scale = {}
    if savevars:  
        for v, vname in enumerate(var_names):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), vname)
            f = data[:, :, v]
            slc = f[f != -999]
            # -- first find the mean and std of the training data
            m, s = slc.mean(), slc.std()
            # -- then scale the training distributions using the found mean and std
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')
            scale[v] = {'name' : athenaname(vname), 'mean' : m, 'sd' : s}

        # -- write variable json for lwtnn keras2json converter
        variable_dict = {
            'inputs': [{
                'name': scale[v]['name'],
                'scale': float(1.0 / scale[v]['sd']),
                'offset': float(-scale[v]['mean']),
                'default': None
                } 
                for v in xrange(len(var_names))],
            'class_labels': ['pu', 'pc', 'pb', 'ptau'],
            'keras_version': keras.__version__,
            'miscellaneous': {
                'sort_by': SORT_COL
            }
        }
        with open(VAR_FILE_NAME, 'wb') as varfile:
            json.dump(variable_dict, varfile)

    # -- when operating on the test sample, use mean and std from training sample
    # -- this info is stored in the json file
    else:
        with open(VAR_FILE_NAME, 'rb') as varfile:
            varinfo = json.load(varfile)

        for v, vname in enumerate(var_names):
            print 'Scaling feature %s of %s (%s).' % (v, len(var_names), vname)
            f = data[:, :, v]
            slc = f[f != -999]
            ix = [i for i in xrange(len(varinfo['inputs'])) if varinfo['inputs'][i]['name'] == athenaname(vname)]
            offset = varinfo['inputs'][ix[0]]['offset']
            scale = varinfo['inputs'][ix[0]]['scale']
            slc += offset
            slc *= scale
            data[:, :, v][f != -999] = slc.astype('float32')


def process_data(trk, cut_vars, savevars=False):
    ''' 
    takes a dataframe directly from ROOT files and 
    processes, sorts, and scales the data for you
    
    Args:
    -----
        trk: the dataframe in question
        cut_vars: list of variables just used for ftag cuts
                  these will need to be dropped before training
        savevars: bool -- True for training, False for testing
                  it decides whether we want to fit on data to find mean and std 
                  or if we want to use those stored in the json file 

    Returns:
    --------
        a dictionary with keys = {'X', 'y', and 'ip3d'}
    
    '''
    # -- flatten to jet-flat, apply ftag jet cuts and remove jets with 0 tracks
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
    scale(data, var_names, SORT_COL, savevars)

    # -- default values                                                                                                                                                                                          
    data[np.isnan(data)] = -999

    # -- make classes pretty for keras (4 classes: b vs c vs l vs tau)                                                                                                                                                                          
    for ix, flav in enumerate(np.unique(y)):
        y[y == flav] = ix
    y_train = np_utils.to_categorical(y, len(np.unique(y)))

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


    



