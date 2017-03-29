# RNNIP - Impact Parameter Tagging with Recurrent Neural Networks

This repository contains the scripts used to obtain the results presented in the [Flavor Tagging Algorithm meeting on April 14th] (https://indico.cern.ch/event/521405/) which later evolved into the [PUB Note](https://cds.cern.ch/record/2255226?ln=en) presented at Connecting The Dots 2017.

## How to Run:
### Libraries
This project uses several python libraries you might have to install to be able to replicate my results. These are listed in the `requirements.txt` and inlcude: [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html), [theano](http://deeplearning.net/software/theano/install.html), [tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html), [keras](https://github.com/fchollet/keras), [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html), [rootpy](http://www.rootpy.org/install.html), [root_numpy](https://rootpy.github.io/root_numpy/install.html), and my own shortcut module called `pandautils` which I added to this project. Install the requirements using `pip install -r requirements.txt`.

### Instructions
1.   Get access to simulated ttbar samples identified by the tag `group.perf-flavtag` which contain track level information.

2.   Construct your feature matrix X and target array y using `dataprocessing.py`. 
```
usage: dataprocessing.py [-h] --train_files TRAIN_FILES --test_files
                         TEST_FILES --output OUTPUT [--sort_by SORT_BY]
                         [--ntrk NTRK] [--inputs INPUTS]

optional arguments:
  -h, --help            show this help message and exit
  --train_files TRAIN_FILES
                        str, name or wildcard specifying the files for
                        training
  --test_files TEST_FILES
                        str, name or wildcard specifying the files for testing
  --output OUTPUT       Tag that refers to the name of the output file, i.e.:
                        '30trk_hits'
  --sort_by SORT_BY     str, name of the variable used to order tracks in an
                        event
  --ntrk NTRK           Maximum number of tracks per event. If the event has
                        fewer tracks, use padding; if is has more, only
                        consider the first ntrk
  --inputs INPUTS       one of: hits, grade
```
This will produce HDF5 files containing a dictionary with X and y for both training and testing. </br>
The remaining scripts will use the .h5 output produced here.

### Ugly hacks to support grade embedding:
The unique values for `jet_trk_ip3d_grade` that are present in the ntuples are `[-10,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11, 12,  13]` (but tracks with grade = -10 are removed). Because the Keras Embedding layer doesn't seem to be willing to accept a Masking layer as input, we have to use its argument `mask_zero` for padding and masking, meaning that we cannot use our preferred value of -999. Therefore, we first pad with -999, we separate the track grade from the other variables, then we shift everything up by 1, making the unique values equal to `[-998,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11, 12,  13,  14]`, and then we turn everything == -998 into a 0. By the way, the grade is not scaled, unlike all other input variables.

3.  To train a Recurrent Neural Network with 1 recurrent layer, use `rnnip.py`.
```
usage: rnnip.py [-h] [--embed_size EMBED_SIZE] [--input_id INPUT_ID]
                [--run_name RUN_NAME] [--normed]

optional arguments:
  -h, --help            show this help message and exit
  --embed_size EMBED_SIZE
                        output dimension of the embedding layer
  --input_id INPUT_ID   string that identifies the name of the hdf5 file to
                        loadcorresponding to --output argument in
                        dataprocessing.py
  --run_name RUN_NAME   string that identifies this specific run, used to
                        label outputs
  --normed              Pass this flag if you want to normalize the embedding
                        output

```
Internally, this script trains and evaluates the performance of the net and plots the ROC curve comparison to IP3D. It also saves out the the ROC curve.

4.   If you want to plot several ROC curves on the same figure, you can modify and then run `plot_comparison.py`

---

Feel free to ask questions, fork this repository, suggest improvements and run your own tests. If you would like to collaborate on this project, <a href="mailto:michela.paganini@cern.ch?Subject=IPNN" target="_top">email me</a>.
