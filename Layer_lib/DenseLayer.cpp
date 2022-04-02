//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"


Tensor DenseLayer::feed(Tensor inputTensor) {
    this->activations = ((this->weightsTensor ^ inputTensor.transpose({1,0})).reshape(nextLayerShape) + this->biasTensor).map(this->activationFunction);
    return this->activations;
}

Tensor DenseLayer::backpropagate(Tensor nextActivationChanges) {
    weightChanges = weightChanges + this->activations.transpose({1,0}) ^ nextActivationChanges;
    biasChanges = biasChanges + nextActivationChanges;
    backPropagationsCarriedOut ++;
    Tensor activationChanges = (this->weightsTensor ^ nextActivationChanges.transpose({1,0})).reshape(nextLayerShape);
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
}

LayerType DenseLayer::GET_LAYER_TYPE() const {
    return Dense;
}
