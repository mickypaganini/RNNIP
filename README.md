# IPNN - Convolutional and Recurrent Architectures for Impact Parameter Tagging

This repository contains the scripts used to obtain the results presented in the [Flavor Tagging Algorithm meeting on April 14th] (https://indico.cern.ch/event/521405/).

## How to Run:
### Libraries
This project uses several python libraries you might have to install to be able to replicate my results. These inlcude: [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html), [theano](http://deeplearning.net/software/theano/install.html), [tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html), [keras](https://github.com/fchollet/keras), [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), [rootpy](http://www.rootpy.org/install.html), [root_numpy](https://rootpy.github.io/root_numpy/install.html), and my own shortcut module called `pandautils` which I added to this project.

### Instructions
1.   Get access to simulated ttbar samples identified by the tag `group.perf-flavtag` which contain track level information.

2.   Construct your feature matrix X and target array y using `dataprocessing.py`. Instructions on how to run it are included in the file description at the beginning of the file. </br>
This will produce HDF5 files containing a dictionary with X and y for both training and testing. </br>
The remaining scripts will use the .h5 output.

3.  To train a Recurrent Neural Network with 1 GRU layer, use `IPRNN.py`. Instructions on how to run it are included in the file description at the beginning of the file. </br>
Internally, this script trains and evaluates the performance of the net and plots the ROC curve comparison to IP3D. It also saves out the the ROC curve.

4.   If you want to plot several ROC curves on the same figure, you can modify and then run `plot_comparison.py`

---

Feel free to ask questions, fork this repository, suggest improvements and run your own tests. If you would like to collaborate on this project, <a href="mailto:michela.paganini@cern.ch?Subject=IPNN" target="_top">email me</a>.
