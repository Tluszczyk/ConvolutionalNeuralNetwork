//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"


Tensor DenseLayer::feed(Tensor inputTensor) {
    this->activations = (((this->weightsTensor ^ inputTensor.transpose({1,0})).reshape({nextLayerSize}) + this->biasTensor));
    return this->activations;
}

Tensor DenseLayer::backpropagate(Tensor nextActivationChanges) {
    weightChanges = weightChanges + this->activations.transpose({1,0}) ^ nextActivationChanges;
    biasChanges = biasChanges + nextActivationChanges;
    backPropagationsCarriedOut ++;
    Tensor activationChanges = (this->weightsTensor ^ nextActivationChanges.transpose({1,0})).reshape({nextLayerSize});
    return activationChanges;
}

void DenseLayer::compile(double learningRate1, int nextLayerSize) {
    this->learningRate = learningRate1;
    this->nextLayerSize = nextLayerSize;

    this->weightsTensor = Tensor::createRandom({this->size, nextLayerSize});
    this->biasTensor = Tensor::createRandom({nextLayerSize});
}

LayerType DenseLayer::GET_LAYER_TYPE() const {
    return Dense;
}
