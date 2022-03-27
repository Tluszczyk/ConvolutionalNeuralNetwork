//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_SEQUENTIAL_H
#define NEURALNET_SEQUENTIAL_H

#include <vector>

#include "Tensor.h"
#include "Layer.h"

class Sequential {
private:
    double learningRate;

    // TODO loss

    std::vector<Layer> layers;

public:
    Tensor feed(Tensor inputTensor);
    void backpropagate(Tensor gradient);
};


#endif //NEURALNET_SEQUENTIAL_H
