//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"

Tensor DenseLayer::feed(Tensor inputTensor) {
    return (this->weightsTensor ^ inputTensor.transpose({1,0})).reshape({nextLayerSize}) + this->biasTensor;
}

Tensor DenseLayer::backpropagate(Tensor gradient) {
    return Tensor{{},{}};
}

void DenseLayer::compile(double learningRate, int nextLayerSize) {
    this->learningRate = learningRate;
    this->nextLayerSize = nextLayerSize;

    this->weightsTensor = Tensor::createRandom({this->size, nextLayerSize});
    this->biasTensor = Tensor::createRandom({nextLayerSize});
}
