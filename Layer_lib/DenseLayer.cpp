//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"


Tensor DenseLayer::feed(Tensor inputTensor) {
    this->activations = ((this->weightsTensor ^ inputTensor.transpose({1,0})).reshape({nextLayerSize}) + this->biasTensor);
    return this->activations;
}

BackpropagationResult DenseLayer::backpropagate(Tensor nextActivationChanges) {
    Tensor weightChanges = this->activations.transpose({1,0}) ^ nextActivationChanges;
    const Tensor& biasChanges = nextActivationChanges;
    Tensor activationChanges = (this->weightsTensor ^ nextActivationChanges.transpose({1,0})).reshape({nextLayerSize});
    return BackpropagationResult(activationChanges, weightChanges, biasChanges);
}

void DenseLayer::compile(double learningRate, int nextLayerSize) {
    this->learningRate = learningRate;
    this->nextLayerSize = nextLayerSize;

    this->weightsTensor = Tensor::createRandom({this->size, nextLayerSize});
    this->biasTensor = Tensor::createRandom({nextLayerSize});
}
