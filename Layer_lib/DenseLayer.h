//
// Created by Filip Tłuszcz on 26.03.2022.
//

#ifndef NEURALNET_DENSELAYER_H
#define NEURALNET_DENSELAYER_H

#include "Layer.h"
#include "ActivationFunctionsProvider.h"

#include <utility>

class DenseLayer : public Layer {
private:
    Tensor activations;
    Tensor weightChanges, biasChanges;
    int backPropagationsCarriedOut;

public:
    explicit DenseLayer(vector<int> shape, const string &activationFunctionName, string layerName) : Layer(std::move(shape), activationFunctionName, std::move(layerName)) {
        backPropagationsCarriedOut = 0;

        this->activationFunction = ActivationFunctionsProvider::fromName[activationFunctionName];
        this->activationFunctionPrime = ActivationFunctionsProvider::fromName[activationFunctionName + "Prime"];
    };

    void compile(double learningRate, const vector<int>& nextLayerShape);
    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;

    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(const Tensor& nextActivationChanges);

    double learningRate{};
    Tensor weightsTensor;
    Tensor biasTensor;
};


#endif //NEURALNET_DENSELAYER_H
