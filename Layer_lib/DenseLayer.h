//
// Created by Filip Tłuszcz on 26.03.2022.
//

#ifndef NEURALNET_DENSELAYER_H
#define NEURALNET_DENSELAYER_H

#include "Layer.h"

#include <utility>

class DenseLayer : Layer {
private:
    double learningRate{};
    Tensor weightsTensor, biasTensor, activations;
    Tensor weightChanges, biasChanges;
    int backPropagationsCarriedOut;

public:
    explicit DenseLayer(int size, const string &activationFunctionName, string layerName) : Layer(size, activationFunctionName, std::move(layerName)) {
        backPropagationsCarriedOut = 0;
    };

    void compile(double learningRate1, int nextLayerSize);
    LayerType GET_LAYER_TYPE() const;

    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(Tensor gradient);
};


#endif //NEURALNET_DENSELAYER_H
