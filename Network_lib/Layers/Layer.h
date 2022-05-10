//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_LAYER_H
#define NEURALNET_LAYER_H

#include "Tensor.h"
#include "LayerType.h"
#include "string"
#include "ActivationFunctionsProvider.h"

#include <string>
#include <utility>


using namespace std;


class Layer {
    friend class ModelLoader;
    friend class LayerLoader;

protected:
    vector<int> shape, nextLayerShape{};

    const string activationFunctionName;

    function<double(double)> activationFunction;
    function<double(double)> activationFunctionPrime;

    Tensor activations;

    [[nodiscard]] virtual LayerType GET_LAYER_TYPE() const = 0;
    string layerName;

    explicit Layer(vector<int> shape, const string& activationFunctionName="id", string layerName="Layer") :
        shape(std::move(shape)),
        layerName(std::move(layerName)),
        activationFunctionName(activationFunctionName),
        activationFunction(ActivationFunctionsProvider::fromName[activationFunctionName]),
        activationFunctionPrime(ActivationFunctionsProvider::derivativeFromName[activationFunctionName]) {};

public:
    [[nodiscard]] vector<int> getShape() const { return this->shape; }

    virtual Tensor feed(Tensor inputTensor) = 0;

    virtual void compile(double learningRate, const vector<int>& nextLayerSize) {}

    const string &getActivationFunctionName() const { return activationFunctionName; };
    const string &getName() const { return layerName; }

    virtual ~Layer() = default;

    virtual Tensor backpropagate(const Tensor& gradient) {return gradient;};

    virtual ~Layer() = default;
};

#endif //NEURALNET_LAYER_H
