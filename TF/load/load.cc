#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/graph.h"
#include <iostream>

using namespace tensorflow;

int main(int argc, char* argv[]) {

  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "/home/ubuntu/prova.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Tensors that will contain the NN predictions
  std::vector<tensorflow::Tensor> out;

  /*
  // Check content of the graph and store node names
  std::vector<string> vNames;
  int node_count = graph_def.node_size();
  for (int i = 0; i < node_count; i++) {
    auto n = graph_def.node(i);
    vNames.push_back(n.name());
  }
  */

  // Prepare input tensor
  int NUM_EVENTS = 1;
  int NUM_VARIABLES = 2;
  tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({NUM_EVENTS, NUM_VARIABLES}));
  auto input_tensor_mapped = input_tensor.tensor<float, 2>(); // interact with input_tensor thru this version of it

  // Currently hard-coding in one event for testing
  input_tensor_mapped(0,0) = 3.24311246;
  input_tensor_mapped(0,1) = -5.56385935;
 
  // Create pair to insert into session->Run()
  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors({{"Placeholder", input_tensor}});

  // Pass the event thru the NN and calculate output
  status = session->Run(input_tensors, {"Sigmoid"}, {}, &out);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Write out the net's output
  auto out_mapped = out[0].tensor<float, 2>(); // [0] because we only have 1 output node ("Sigmoid")
  for (int i = 0; i < NUM_EVENTS; i++){
    std::cout << out_mapped(0,i) << std::endl;
  }

  // Free any resources used by the session
  session->Close();
  return 0;
}
