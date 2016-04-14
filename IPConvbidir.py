''' 
IPConvbidir.py -- functionality for training the bi-directional
variant of the RCNN
'''

import numpy as np
import sys
import cPickle

import deepdish.io as io

from keras.layers import containers, GRU, LSTM, Highway, Dense, Dropout, MaxoutDense, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Graph



# max number of tracks to consider per jet
# we zero pad if we dont have enough, and we truncate
# if we have too many
N_TRACKS = 30

def build_graph(n_variables):
    '''
    Creates the Graph component of the model, i.e., this creates the 
    conv+gru  with bi-directional
    '''
    nb_feature_maps = 64
    ngram_filters = [1, 2, 3, 4, 5] #, 6, 7, 8]

    graph = Graph()
    graph.add_input(name='data', input_shape= (N_TRACKS, n_variables))

    for n_gram in ngram_filters:
        graph.add_node(
            Convolution1D(
                nb_feature_maps, 
                n_gram, 
                activation='relu', 
                input_shape=(N_TRACKS, n_variables)
            ),
            name='conv_%s' % n_gram,
            input='data'
        )

        graph.add_node(
            GRU(25),
            name='gru_fwd_%s' % n_gram,
            input='conv_%s' % n_gram,
        )

        graph.add_node(
            GRU(25, go_backwards=True),
            name='gru_bwd_%s' % n_gram,
            input='conv_%s' % n_gram,
        )
        
        pass_thru = Lambda(lambda x: x)
        graph.add_node(
            pass_thru, 
            name = 'unit_{}'.format(n_gram), 
            inputs=['gru_fwd_%s' % n_gram, 'gru_bwd_%s' % n_gram]
        )

    graph.add_node(Dropout(0.4), name='dropout', inputs=['unit_{}'.format(n) for n in ngram_filters], create_output=True)

    return graph


def build_graph_noCNN(n_variables):
    '''
    Creates the Graph component of the model, i.e., this creates the 
    gru component with no CNN
    '''
    graph = Graph()
    graph.add_input(name='data', input_shape= (N_TRACKS, n_variables))
    
    n_gram = 1

    graph.add_node(
            GRU(25),
            name='gru_fwd_%s' % n_gram,
            input='data'
        )

    graph.add_node(
            GRU(25, go_backwards=True),
            name='gru_bwd_%s' % n_gram,
            input='data'
        )

    pass_thru = Lambda(lambda x: x)
    graph.add_node(
            pass_thru,
            name = 'unit_{}'.format(n_gram),
            inputs=['gru_fwd_%s' % n_gram, 'gru_bwd_%s' % n_gram]
        )

    graph.add_node(Dropout(0.4), name='dropout', input='unit_{}'.format(n_gram), create_output=True)

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
    import cPickle
    cPickle.dump(discs[MODEL_FILE], open(MODEL_FILE + '.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    print 'Plotting'
    fg = ROC_plotter(discs, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, logscale=True)

    return fg


def main(MODEL_FILE):

    test_dict = io.load('./data/test_dict_IPConv.h5')
    train_dict = io.load('./data/train_dict_IPConv.h5')

    X_train = train_dict['X']
    y_train = train_dict['y']
    n_features = X_train.shape[2]    
    
    X_test = test_dict['X']
    y_test = test_dict['y']
    ip3d = test_dict['ip3d'] # this is a df

    print 'Building model...'
    
    if (MODEL_FILE == 'CRNN'):
        graph = build_graph(n_features)

        model = Sequential()

        model.add(graph)

        model.add(Dense(64))

    elif (MODEL_FILE == 'RNN'):

        graph = build_graph_noCNN(n_features)
        
        model = Sequential()
        model.add(graph)

        model.add(Dense(64))
  
    model.add(Dropout(0.4))

    model.add(Highway(activation = 'relu'))

    model.add(Dropout(0.4)) #3
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
        nb_epoch=200, 
        validation_split = 0.2, 
        show_accuracy=True) 
        
    except KeyboardInterrupt:
        print 'Training ended early.'

    # -- load in best network
    model.load_weights(MODEL_FILE + '-progress')

    print 'Saving weights...'
    model.save_weights('./weights/ip3d-replacement_' + MODEL_FILE + '.h5', overwrite=True)
    
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





















