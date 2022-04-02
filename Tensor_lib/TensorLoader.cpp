//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "TensorLoader.h"

using namespace std;

void TensorLoader::saveDenseLayer(ofstream &modelFile, Tensor *tensor) {
    for(int dim : tensor->shape) modelFile << to_string(dim) << " ";
    modelFile << endl;
    for(double elem : tensor->data) modelFile << to_string(elem) << " ";
    modelFile << endl;
}
