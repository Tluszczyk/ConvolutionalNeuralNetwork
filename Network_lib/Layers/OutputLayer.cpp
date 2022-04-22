//
// Created by kubkm on 21.04.2022.
//

#include "OutputLayer.h"

Tensor OutputLayer::feed(Tensor inputTensor) {
    this->activations = inputTensor;
    return activations;
}

Tensor OutputLayer::backpropagate(const Tensor &gradient) {
    return gradient;
}

LayerType OutputLayer::GET_LAYER_TYPE() const {
    return Output;
}

