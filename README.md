# IPNN - Convolutional and Recurrent Architectures for Impact Parameter Tagging

This repository contains the scripts used to obtain the results presented in the [Flavor Tagging Algorithm meeting on April 14th] (https://indico.cern.ch/event/521405/).

## How to Run:
### Libraries
This project uses several python libraries you might have to install to be able to replicate my results. These inlcude: [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html), [theano](http://deeplearning.net/software/theano/install.html), [tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html), [keras](https://github.com/fchollet/keras), [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), [rootpy](http://www.rootpy.org/install.html), [root_numpy](https://rootpy.github.io/root_numpy/install.html), and my own shortcut module called `pandautils` which I added to this project.

### Instructions
1.   Get access to simulation samples such as the ones located on lxplus at `/afs/cern.ch/work/m/mpaganin/public/temp/IPNN/` (CERN access required). </br> 
In this folder, `inputChallenge_mc15_13TeV_410000_0X_v2.root` (X=1,2,3,4) are ttbar ntuples with truth track information,
`inputChallenge_mc15_13TeV_301329_0X_v1.root` (X=1,2,3,4) are Z’ ntuples with truth track information

2.   Construct your feature matrix X and target array y using `python dataprocessing.py` </br>
This will produce HDF5 files containing a dictionary with X and y for both training and testing. </br>
You only have to run this once per data set; modify the script to specify the sample you want to run on. The remaining scripts will use the .h5 output.

3.   Decide what type of model you would like to train. For simple RNN and CRNN, run `python IPConv.py <model_name>`, where `<model_name>` can be either `RNN` or `CRNN`. For models containing bidirectional recurrent neural networks, run `python IPConvbidir.py <model_name>`, with the same possible options for the model name. </br>
Internally, this script trains and evaluates the performance of the net and plots the ROC curve comparison to IP3D. It also saves out the the ROC curve as a pickle.

4.   If you would like to plot several ROC curves on the same figure, you can modify and then run `python plot_comparison.py`

---

Feel free to ask questions, fork this repository, suggest improvements and run your own tests. If you would like to collaborate on this project, <a href="mailto:michela.paganini@cern.ch?Subject=IPNN" target="_top">email me</a>.
