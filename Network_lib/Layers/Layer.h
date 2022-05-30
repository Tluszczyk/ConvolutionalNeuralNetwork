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


class Layer {
    friend class ModelLoader;
    friend class LayerLoader;

protected:
    std::vector<int> shape, nextLayerShape{};

    const std::string activationFunctionName;

    std::function<double(double)> activationFunction;
    std::function<double(double)> activationFunctionPrime;

    Tensor activations;

    [[nodiscard]] virtual LayerType GET_LAYER_TYPE() const = 0;
    std::string layerName;

    explicit Layer(std::vector<int> shape, const std::string& activationFunctionName="id", std::string layerName="Layer") :
        shape(std::move(shape)),
        layerName(std::move(layerName)),
        activationFunctionName(activationFunctionName),
        activationFunction(ActivationFunctionsProvider::fromName[activationFunctionName]),
        activationFunctionPrime(ActivationFunctionsProvider::derivativeFromName[activationFunctionName]) {};

public:
    [[nodiscard]] std::vector<int> getShape() const { return this->shape; }

    virtual Tensor feed(Tensor inputTensor) = 0;

    virtual void compile(double learningRate, const std::vector<int>& nextLayerSize) {}

    const std::string &getActivationFunctionName() const { return activationFunctionName; };
    const std::string &getName() const { return layerName; }

    virtual Tensor backpropagate(const Tensor& gradient) {return gradient;};

    virtual ~Layer() = default;
};

#endif //NEURALNET_LAYER_H
