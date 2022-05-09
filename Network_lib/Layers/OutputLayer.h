//
// Created by kubkm on 21.04.2022.
//

#ifndef NEURALNET_OUTPUTLAYER_H
#define NEURALNET_OUTPUTLAYER_H


#include "Layer.h"

class OutputLayer : public Layer {

public:

    explicit OutputLayer(vector<int> shape, const string &activationFunctionName, string layerName) : Layer(std::move(shape), activationFunctionName, std::move(layerName)) {
    };

    Tensor feed(Tensor inputTensor) override;

    Tensor backpropagate(const Tensor& gradient) override;

    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;

    Tensor getWeights(){ return Tensor();}

};


#endif //NEURALNET_OUTPUTLAYER_H
