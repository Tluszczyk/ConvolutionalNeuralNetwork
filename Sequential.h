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

using namespace std;

class Sequential {
private:
    double learningRate;

    function<double(double)> activationFunction;
    function<double(double)> activationFunctionPrime;

    function<double(double, double)> lossFunction;
    function<double(double, double)> lossFunctionPrime;

    // TODO loss

    std::vector<Layer> layers;

public:
    explicit Sequential(vector<Layer> layers) : layers(std::move(layers)) {};

    Tensor feed(Tensor inputTensor);
    void backpropagate(Tensor gradient);

    void compile(double learningRate, const string& activationFunctionName, const string& lossFunctionName);
};


#endif //NEURALNET_SEQUENTIAL_H
