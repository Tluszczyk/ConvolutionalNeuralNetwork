//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_LAYER_H
#define NEURALNET_LAYER_H

#include "Tensor.h"
#include "LayerType.h"

#include <string>
#include <utility>

using namespace std;


class Layer {
    friend class ModelLoader;
    friend class LayerLoader;

protected:
    int size, nextLayerSize{};

    function<double(double)> activationFunction;
    function<double(double)> activationFunctionPrime;

    [[nodiscard]] virtual LayerType GET_LAYER_TYPE() const = 0;
    string layerName;

    explicit Layer(int size, const string& activationFunctionName="id", string layerName="Layer")
        : size(size), layerName(std::move(layerName)) {};

public:
    [[nodiscard]] int getShape() const { return this->size; }

    virtual Tensor feed(Tensor inputTensor) = 0;

    virtual void compile(double learningRate, int nextLayerSize) {}
};

#endif //NEURALNET_LAYER_H
