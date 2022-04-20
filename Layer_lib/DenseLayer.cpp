//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"

#include <iostream>


Tensor DenseLayer::feed(Tensor inputTensor) {
    this->activations = inputTensor;
    this->futureActivationsBeforeFunction = (this->weightsTensor ^ inputTensor.transpose({1,0})).reshape(nextLayerShape) + this->biasTensor;
    return futureActivationsBeforeFunction.map(this->activationFunction);
}

Tensor DenseLayer::backpropagate(const Tensor& nextActivationChanges) {
    Tensor firstPart = (nextActivationChanges * this->futureActivationsBeforeFunction.map(this->activationFunctionPrime));
    weightChanges = weightChanges + (firstPart.transpose({1,0}) ^ this->activations);
    biasChanges = biasChanges + firstPart;
    backPropagationsCarriedOut ++;
    Tensor activationChanges = (firstPart ^ this->weightsTensor).reshape(this->shape);
    return activationChanges;
}

void DenseLayer::compile(double learningRate1, const vector<int>& nextLayerShape) {
    this->learningRate = learningRate1;
    this->nextLayerShape = nextLayerShape;

    vector<int> weightShape;

    copy(this->shape.begin(), this->shape.end(), back_inserter(weightShape));
    copy(this->nextLayerShape.begin(), this->nextLayerShape.end(), back_inserter(weightShape));

    this->weightsTensor = Tensor::createRandom(weightShape);
    this->biasTensor = Tensor::createRandom(nextLayerShape);

    this->weightChanges = Tensor(weightShape);
    this->biasChanges = Tensor(nextLayerShape);
}

void DenseLayer::applyChanges() {
    if (backPropagationsCarriedOut == 0){
        return;
    }
    weightsTensor = weightsTensor - (weightChanges * (learningRate / backPropagationsCarriedOut));
    biasTensor = biasTensor - (biasChanges * (learningRate / backPropagationsCarriedOut));
    backPropagationsCarriedOut = 0;
    weightChanges = Tensor(weightChanges.getShape());
    biasChanges = Tensor(biasChanges.getShape());
}

LayerType DenseLayer::GET_LAYER_TYPE() const {
    return Dense;
}
