//
// Created by Filip Tłuszcz on 26.03.2022.
//

#ifndef NEURALNET_DENSELAYER_H
#define NEURALNET_DENSELAYER_H

#include "Layer.h"
#include "LayerLoader.h"
#include "ActivationFunctionsProvider.h"

#include <string>

using namespace std;

class DenseLayer : public Layer {
    friend class LayerLoader;

private:
    double learningRate{};
    Tensor weightsTensor, biasTensor;

public:
    explicit DenseLayer(int size, const string& activationFunctionName="id", const string& layerName="Dense Layer 69")
        : Layer({size}, activationFunctionName, layerName) {

        this->activationFunction = ActivationFunctionsProvider::fromName[activationFunctionName];
        this->activationFunctionPrime = ActivationFunctionsProvider::fromName[activationFunctionName + "Prime"];
    };

    virtual void compile(double learningRate, vector<int> nextLayerShape) override;

    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(Tensor gradient);

    [[nodiscard]] LayerType GET_LAYER_TYPE() const override { return LayerType::Dense; };
};


#endif //NEURALNET_DENSELAYER_H
