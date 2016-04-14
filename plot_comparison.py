import cPickle
import matplotlib.pyplot as plt
from viz import calculate_roc, ROC_plotter, add_curve

#RNN = cPickle.load( open('./GRU25/RNN.pkl', 'r'))
#CRNN = cPickle.load( open('./GRU25/CRNN.pkl' , 'r'))
#IP3D = cPickle.load( open('./GRU25/ip3d.pkl', 'r'))
bidirCRNN = cPickle.load( open('./BIDIR/CRNN.pkl' , 'r'))
num2 = cPickle.load( open('./CRNN.pkl' , 'r'))
#bidirRNN = cPickle.load( open('./BIDIR/RNN.pkl' , 'r'))

#CRNN['color'] = 'magenta'
bidirCRNN['color'] = 'purple'
#bidirRNN['color'] = 'cyan'

curves = {}
#curves['RNN'] = RNN
#curves['CRNN'] = CRNN
#curves['IP3D'] = IP3D 
curves['Bidir. CRNN'] = bidirCRNN
curves['Most Recent'] = num2
#curves['Bidir. RNN'] = bidirRNN

print 'Plotting'
fg = ROC_plotter(curves, title=r'Impact Parameter Taggers', min_eff = 0.5, max_eff=1.0, ymax = 1000, logscale=True)
fg.savefig('./plots/comparison_bidirCRNNs.pdf')
