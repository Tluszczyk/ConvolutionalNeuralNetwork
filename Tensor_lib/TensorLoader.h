//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_TENSORLOADER_H
#define NEURALNET_TENSORLOADER_H

#include "Tensor.h"

#include <fstream>

using namespace std;

class Tensor;

class TensorLoader {
public:
    static void saveTensor(ofstream& modelFile, Tensor* tensor);
    static Tensor loadTensor(ifstream& modelFile);
};


#endif //NEURALNET_TENSORLOADER_H
