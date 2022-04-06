//
// Created by Filip Tłuszcz on 26.03.2022.
//

#ifndef NEURALNET_DENSELAYER_H
#define NEURALNET_DENSELAYER_H

#include "Layer.h"

#include <utility>

class DenseLayer : public Layer {
private:
    Tensor activations, futureActivationsBeforeFunction;
    Tensor weightChanges, biasChanges;
    int backPropagationsCarriedOut;

public:
    explicit DenseLayer(vector<int> shape, const string &activationFunctionName, string layerName) : Layer(std::move(shape), activationFunctionName, std::move(layerName)) {
        backPropagationsCarriedOut = 0;
    };

    void compile(double learningRate1, const vector<int>& nextLayerShape) override;
    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;
    [[nodiscard]] const Tensor &getActivations() const { return activations; };

    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(const Tensor& gradient);

    double learningRate{};
    Tensor weightsTensor;
    Tensor biasTensor;

    void changeWeightsTensor(Tensor newWeights){
        this->weightsTensor = std::move(newWeights);
    }

    void changeBiasTensor(Tensor newBiases){
        this->biasTensor = std::move(newBiases);
    }
};


#endif //NEURALNET_DENSELAYER_H
