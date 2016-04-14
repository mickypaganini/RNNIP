
import pandautils as pup
import pandas as pd
import numpy as np

import deepdish.io as io

# ------------------------------------------                                                                                                                                                                     

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




