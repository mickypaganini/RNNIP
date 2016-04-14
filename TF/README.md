This folder contains some of the files that are needed to re-assemble and load neural nets saved as protobuffers for prediction.

It relies on the assumption that a protobuf file called `graph.pb` and a checkpoint file called `model-weights.ckpt` were saved while training NN in Keras using the TensorFlow backend. The `IPConv.py` and `IPConvbidir.py` scripts in the previos directory contain a section that produces these files. The Jupyter Notebook `DummyNet.ipynb` does the same thing, but using dummy shallow net as an example. To run this notebook using the TensorFlow backend, use the command:
` KERAS_BACKEND=tensorflow ipython notebook DummyNet.ipynb`

`graph.pb` stores all the nodes in the NN, defining the architecture and its logic. </br>
`model-weights.ckpt` contains the trained weights of the neural network.

### Freezing the Graph

`freeze_graph.py`, a script developed by Google, allows us to merge the weights with the architecture, and stores the full graph in a new file. You can run this script using the command:
 
``` 
python freeze_graph.py --input_graph=graph.pb --input_checkpoint=model-weights.ckpt --output_node_names=Softmax --input_binary=True --output_graph=frozen_graph.pb
```
Depending on the model you train, your output node might be something other than `Softmax`. Check out the code for information on the flags.

### Loading NN and Evaluating in C++

The frozen graph can now be loaded in again and used for prediction yielding the same results as `model.predict(X_test)` in Keras. </br>

*Notes*:
* Currently, `load.cc` contains a hard-coded input event with two features, and loads in a very simple shallow net from the example contained in the Jupyter Notebook `DummyNet.ipynb`. This example has nothing to do with the convolutional and recurrent nets in this project, but it's just a quick example to show how we can use TensorFlow to embed our Keras-trained nets into a C++ work environment.

* I compile and run `load.cc` using [bazel](http://bazel.io/) inside my `tensorflow/tensorflow/load/` directory. 
