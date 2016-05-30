''' 
IPConv.py -- functionality for training the uni-directional
variant of the RCNN
'''

import numpy as np
import sys
import cPickle

import deepdish.io as io

from keras.layers import GRU, Highway, Dense, Dropout, MaxoutDense, Activation, Masking
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.legacy.models import Graph


# max number of tracks to consider per jet
# we zero pad if we dont have enough, and we truncate
# if we have too many
N_TRACKS = 30 
RUN_NAME = 'MyTrkSel'
SAVE_PROTOBUF = False

def build_graph(n_variables):
    '''
    Creates the Graph component of the model, i.e., this creates the 
    conv+gru component
    '''

    nb_feature_maps = 64
    ngram_filters = [1, 2, 3, 4, 5]

    graph = Graph()
    graph.add_input(name='data', input_shape= (N_TRACKS, n_variables))

    for n_gram in ngram_filters:
        sequential = Sequential()
        sequential.add(
            Convolution1D(
                nb_feature_maps, 
                n_gram, 
                activation='relu', 
                input_shape=(N_TRACKS, n_variables)
            )
        )
        
        sequential.add(GRU(25))

        graph.add_node(sequential, name = 'unit_{}'.format(n_gram), input='data')

    graph.add_node(Dropout(0.4), name='dropout', inputs=['unit_{}'.format(n) for n in ngram_filters], create_output=True)

    return graph


def plot_ROC(y_test, yhat, ip3d, MODEL_FILE):
    ''' 
    plot a ROC curve for the discriminant
    
    Args:
    -----
        y_test: the truth labels for the trst set
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

    # -- add ROC curves
    discs = {}

    add_curve(r'IP3D', 'black', calculate_roc((y[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ] == 2), 
                                              np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu)[ bl_sel & np.isfinite(np.log(ip3d.jet_ip3d_pb/ip3d.jet_ip3d_pu).values) ]), 
              discs)

    
    add_curve(MODEL_FILE, 'blue', calculate_roc( (y[ bl_sel & np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] == 2), 
                                              np.log(yhat[:,2] / yhat[:,0])[bl_sel &  np.isfinite(np.log(yhat[:,2] / yhat[:,0]))] ), 
              discs)

    print 'Pickling ROC curves'
    
    cPickle.dump(discs[MODEL_FILE], open(MODEL_FILE + RUN_NAME +'.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print 'Plotting'
    fg = ROC_plotter(discs, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, logscale=True)

    return fg

def main(MODEL_FILE):

    print "Loading hdf5's..."
    test_dict = io.load('./data/test_dict_IPConv_ntuple_'+ RUN_NAME +'.h5')
    train_dict = io.load('./data/train_dict_IPConv_ntuple_'+ RUN_NAME +'.h5')
    
    X_train = train_dict['X']
    y_train = train_dict['y']    

    X_test = test_dict['X']
    y_test = test_dict['y']
    n_features = X_test.shape[2]

    # this is a df
    ip3d = test_dict['ip3d'] 

    print 'Building model...'
    
    if (MODEL_FILE == 'CRNN'):
        graph = build_graph(n_features)

        model = Sequential()

        model.add(graph)
        # remove Maxout for tensorflow
        model.add(MaxoutDense(64, 5, input_shape=graph.nodes['dropout'].output_shape[1:]))
        model.add(Dense(64))

    elif (MODEL_FILE == 'RNN'):

        model = Sequential()
        model.add(Masking(mask_value=-999, input_shape=(N_TRACKS, n_features)))
        model.add(GRU(25))#, input_shape=(N_TRACKS, n_features))) #GRU
        model.add(Dropout(0.2)) #0.2
    
        # remove Maxout for tensorflow
        model.add(MaxoutDense(64, 5))  #, input_shape=graph.nodes['dropout'].output_shape[1:]))
        model.add(Dense(64, activation='relu'))

  
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
                ModelCheckpoint(MODEL_FILE + RUN_NAME +'-progress', monitor='val_loss', verbose=True, save_best_only=True)
            ],
        nb_epoch=100, 
        validation_split = 0.2, 
        show_accuracy=True) 
        
    except KeyboardInterrupt:
        print 'Training ended early.'

    # -- load in best network
    model.load_weights(MODEL_FILE + RUN_NAME +'-progress')
    
    if (SAVE_PROTOBUF):
        print 'Saving protobuf'
        # write out to a new directory called models
        # the actual graph file is graph.pb
        # the graph def is in the global session
        import tensorflow as tf
        import keras.backend.tensorflow_backend as tfbe

        sess = tfbe._SESSION

        saver = tf.train.Saver()
        tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb', as_text=False)    

        save_path = saver.save(sess, "./model-weights.ckpt")
        print "Model saved in file: %s" % save_path
        
        print saver.as_saver_def().filename_tensor_name
        print saver.as_saver_def().restore_op_name

        print model.get_output()

    print 'Saving weights...'
    model.save_weights('./weights/ip3d-replacement_' + MODEL_FILE + RUN_NAME +'.h5', overwrite=True)

    json_string = model.to_json()
    open(MODEL_FILE + RUN_NAME +'.json', 'w').write(json_string)

    print 'Testing...'
    yhat = model.predict(X_test, verbose = True, batch_size = 512) 
    io.save('yhat'+ RUN_NAME +'.h5', yhat) 
     
    print 'Plotting ROC...'
    fg = plot_ROC(y_test, yhat, ip3d, MODEL_FILE)
    #plt.show()
    fg.savefig('./plots/roc' + MODEL_FILE + RUN_NAME +'.pdf')
        
if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("model", help="type of architecture: CRNN or RNN")
   
    args = parser.parse_args()

    if not( (args.model == 'CRNN') or (args.model == 'RNN') ):
        sys.exit('Error: Unknown model. Pick: CRNN or RNN')
    else:
        sys.exit( main(args.model) )

