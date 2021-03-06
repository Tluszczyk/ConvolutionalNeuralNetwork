//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_SEQUENTIAL_H
#define NEURALNET_SEQUENTIAL_H

#include <utility>
#include <functional>
#include <vector>
#include <cmath>

#include "Tensor.h"
#include "Layers/Layer.h"
#include "ModelLoader.h"

class Sequential {
    friend class ModelLoader;

private:
    const string MODEL_NAME;

    double learningRate{};

    function<double(double, double)> lossFunction;
    function<double(double, double)> lossFunctionPrime;

    void applyChanges();
    void backpropagate(const Tensor& gradient);

public:
    std::vector<Layer*> layers;
    explicit Sequential(vector<Layer*> layers, string MODEL_NAME="Simple model") : layers(std::move(layers)), MODEL_NAME(std::move(MODEL_NAME)) {};
    void addLayer(Layer* layer);

    void compile(double learningRate=.7, const string& lossFunctionName="MSE");

    Tensor calculateLoss(const Tensor& expected);

    Tensor feed(Tensor inputTensor);
    double analyzeBatch(vector<Tensor> &batch, vector<Tensor> &expectedResults);


    ~Sequential() {
        for (auto &layer : layers) delete layer;
        layers.clear();
    }
};


#endif //NEURALNET_SEQUENTIAL_H