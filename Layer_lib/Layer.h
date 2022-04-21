//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_LAYER_H
#define NEURALNET_LAYER_H

#include "Tensor.h"
#include "LayerType.h"
#include "ActivationFunctionsProvider.h"

#include <string>
#include <utility>


using namespace std;


class Layer {
    friend class ModelLoader;
    friend class LayerLoader;

protected:
    vector<int> shape, nextLayerShape{};

    function<double(double)> activationFunction;
    function<double(double)> activationFunctionPrime;

    Tensor activations;

    [[nodiscard]] virtual LayerType GET_LAYER_TYPE() const = 0;
    string layerName;

    explicit Layer(vector<int> shape, const string& activationFunctionName="id", string layerName="Layer")
        : shape(std::move(shape)), layerName(std::move(layerName)), activationFunction(ActivationFunctionsProvider::fromName[activationFunctionName]), activationFunctionPrime(ActivationFunctionsProvider::derivativeFromName[activationFunctionName]) {};

public:
    [[nodiscard]] vector<int> getShape() const { return this->shape; }

    virtual Tensor feed(Tensor inputTensor) = 0;

    virtual void compile(double learningRate, const vector<int>& nextLayerSize) {}

    virtual ~Layer() = default;

    virtual Tensor backpropagate(const Tensor& gradient) {return gradient;};
};

#endif //NEURALNET_LAYER_H
