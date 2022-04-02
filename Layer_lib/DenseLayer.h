//
// Created by Filip Tłuszcz on 26.03.2022.
//

#ifndef NEURALNET_DENSELAYER_H
#define NEURALNET_DENSELAYER_H

#include "Layer.h"
#include "BackpropagationResult.h"

class DenseLayer : Layer {
private:
    int size, nextLayerSize{};
    double learningRate{};
    Tensor weightsTensor, biasTensor, activations;

public:
    explicit DenseLayer(int size) : size(size) {};

    void compile(double learningRate, int nextLayerSize);

    Tensor feed(Tensor inputTensor) override;
    BackpropagationResult backpropagate(Tensor gradient);
};


#endif //NEURALNET_DENSELAYER_H
