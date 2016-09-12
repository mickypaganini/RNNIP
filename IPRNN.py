''' 
IPRNN.py -- functionality for training the uni-directional
variant of the RNN

Notes:
------
* Default value is -999 (make sure it matches the one in dataprocessing.py)
* Written and tested with keras v1.0.5

Run:
----
python IPRNN.py --input '30trk_hits' --run_name '_dropout' --ntrk 30
'''

import numpy as np
import os
import sys
import cPickle

import deepdish.io as io

from keras.layers import GRU, Dense, Dropout, Activation, Masking
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandautils as pup

MODEL_FILE = 'RNN'

def plot_ROC(y_test, yhat, ip3d, run_name, MODEL_FILE):
    ''' 
    plot a ROC curve for the discriminant
    
    Args:
    -----
        y_test: the truth labels for the test set
        yhat: the predicted probabilities of each class in the test set
    
    Returns:
    --------
        a mpl.figure
    '''
    
    from viz import calculate_roc, ROC_plotter, add_curve
    
    # -- bring classes back to usual format: [0,2,3,0,1,2,0,2,2,...]
    y = np.array([np.argmax(ev) for ev in y_test])

    # -- for b VS. light
    bl_sel = (y == 0) | (y == 2)
    # -- for c VS. light
    cl_sel = (y == 0) | (y == 1)
    # -- for b VS. c
    bc_sel = (y == 1) | (y == 2)

    # -- add ROC curves
    discs = {}

    add_curve(r'IP3D', 'black', calculate_roc((y[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ] == 2), 
                                              np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu)[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ]), 
              discs)

    
    add_curve(MODEL_FILE, 'blue', calculate_roc( (y[ bl_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] == 2), 
                                              np.log(yhat[:,2] / yhat[:,0])[bl_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] ), 
              discs)

    discs_bc = {}

    add_curve(r'IP3D', 'black', calculate_roc((y[ bc_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc).values) ] == 2), 
                                              np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc)[ bc_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pc).values) ]), 
              discs_bc)

    
    add_curve(MODEL_FILE, 'blue', calculate_roc( (y[ bc_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,1]))] == 2), 
                                              np.log(yhat[:,2] / yhat[:,1])[bc_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,1]))] ), 
              discs_bc)

    print 'Pickling ROC curves'
    cPickle.dump(discs[MODEL_FILE], open(MODEL_FILE + run_name +'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(discs_bc[MODEL_FILE], open(MODEL_FILE + run_name +'_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print 'Plotting'
    fg_bl = ROC_plotter(discs, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, logscale=True)
    fg_bl.savefig('./plots/roc' + MODEL_FILE + run_name +'.pdf')

    fg_bc = ROC_plotter(discs_bc, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, logscale=True)
    fg_bc.savefig('./plots/roc' + MODEL_FILE + run_name +'_bc.pdf')


    return fg_bl, fg_bc

def main(file_name, run_name, n_tracks):

    print "Loading hdf5's from ./data/test_dict" + file_name + ".h5 and ./data/train_dict" + file_name + ".h5"
    test_dict = io.load(os.path.join('data', 'test_dict_' + file_name + '.h5'))
    train_dict = io.load(os.path.join('data', 'train_dict_' + file_name + '.h5'))
    
    X_train = train_dict['X']
    y_train = train_dict['y']    

    X_test = test_dict['X']
    y_test = test_dict['y']
    n_features = X_test.shape[2]

    # this is a df
    ip3d = test_dict['ip3d'] 

    print 'Building model...'
    # -- for track grade as a normal input:
    model = Sequential()
    model.add(Masking(mask_value=-999, input_shape=(n_tracks, n_features)))
    model.add(Dropout(0.2)) # dropping out before the GRU should help us reduce dependence on any specific input variable
    # ^^ could be desirable when training on track hits in case one detector layer fails
    model.add(GRU(25, return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.summary()

    print 'Compiling model...'
    model.compile('adam', 'categorical_crossentropy', metrics=["accuracy"])

    # -- if the pre-trained model exists, load it in, otherwise start from scratch
    try:
        _weights_location = os.path.join('weights', 'ip3d-replacement_' + MODEL_FILE + '_' + run_name +'.h5')
        model.load_weights(_weights_location)
        print 'Loaded pre-trained model from ' + _weights_location
    except IOError:
        print 'No pre-trained model found in ' + _weights_location

    print 'Training:'
    try:
        model.fit(X_train, y_train, batch_size=1024,
            callbacks = [
                EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                ModelCheckpoint(MODEL_FILE + run_name +'-progress', monitor='val_loss', verbose=True, save_best_only=True)
            ],
        nb_epoch=300, 
        validation_split = 0.2) 
        
    except KeyboardInterrupt:
        print 'Training ended early.'

    # -- load in best network
    model.load_weights(MODEL_FILE + run_name +'-progress')

    print 'Saving weights in ' + _weights_location
    model.save_weights(_weights_location, overwrite=True)

    json_string = model.to_json()
    open(MODEL_FILE + run_name +'.json', 'w').write(json_string)

    print 'Testing...'
    yhat = model.predict(X_test, verbose = True, batch_size = 1024) 
    io.save('yhat'+ run_name +'.h5', yhat) 
     
    print 'Plotting ROC...'
    fg_bl, fg_bc = plot_ROC(y_test, yhat, ip3d, run_name, MODEL_FILE)
        

if __name__ == '__main__': 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Tag that refers to the name of the input file, i.e.: '_v47_SLAC_hits_no-10'", )
    parser.add_argument("--run_name", required=True, type=str, help="String that uniquely identifies the current run and its output")
    parser.add_argument("--ntrk", default=30, type=int, help="Maximum number of tracks per event. \
        If the event has fewer tracks, use padding; if is has more, only consider the first ntrk")
    args = parser.parse_args()

    run_id = args.input + args.run_name 
    sys.exit(main(args.input, run_id, args.ntrk))
