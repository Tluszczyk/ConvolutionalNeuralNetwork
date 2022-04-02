//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "DenseLayer.h"
#include "Tensor.h"

Tensor DenseLayer::feed(Tensor inputTensor) {
    return ((this->weightsTensor ^ inputTensor.transpose({1,0})).reshape(nextLayerShape) + this->biasTensor).map(this->activationFunction);
}

Tensor DenseLayer::backpropagate(Tensor gradient) {
    return Tensor{{},{}};
}

void DenseLayer::compile(double learningRate, vector<int> nextLayerShape) {
    this->learningRate = learningRate;
    this->nextLayerShape = nextLayerShape;

    vector<int> weightShape;

    copy(this->shape.begin(), this->shape.end(), back_inserter(weightShape));
    copy(this->nextLayerShape.begin(), this->nextLayerShape.end(), back_inserter(weightShape));

    this->weightsTensor = Tensor::createRandom(weightShape);
    this->biasTensor = Tensor::createRandom(nextLayerShape);
}