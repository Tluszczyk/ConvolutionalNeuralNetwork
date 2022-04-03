//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_SEQUENTIAL_H
#define NEURALNET_SEQUENTIAL_H

#include <utility>
#include <functional>
#include <vector>

#include "Tensor.h"
#include "Layer.h"
#include "ModelLoader.h"

using namespace std;

class Sequential {
    friend class ModelLoader;

private:
    const string MODEL_NAME;

    double learningRate{};

    function<double(double, double)> lossFunction;
    function<double(double, double)> lossFunctionPrime;

    // TODO loss

    std::vector<Layer*> layers;

public:
    explicit Sequential(vector<Layer*> layers, string MODEL_NAME="Simple model") : layers(std::move(layers)), MODEL_NAME(std::move(MODEL_NAME)) {};

    Tensor feed(Tensor inputTensor);
    void backpropagate(Tensor gradient);

    void compile(double learningRate=.7, const string& lossFunctionName="MSE");

    Tensor calculateLoss(const Tensor& expected);

    ~Sequential() {
        for (auto &layer : layers) delete layer;
        layers.clear();
    }
};


#endif //NEURALNET_SEQUENTIAL_H
