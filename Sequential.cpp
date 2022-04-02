//
// Created by Filip Tłuszcz on 25.03.2022.
//

#include "Sequential.h"
#include "ActivationFunctionsProvider.h"
#include "LossFunctionsProvider.h"

Tensor Sequential::feed(Tensor inputTensor) {
    /* feed a single tensor to a model,
     * and get the activations of the last layer
     * */
    return Tensor({2}, {1,2});
}

void Sequential::backpropagate(Tensor gradient) {
    /* adjust all the adjustable parameters
     * */

}

void Sequential::compile(double learningRate, const string &activationFunctionName, const string &lossFunctionName) {
    this->learningRate = learningRate;

    this->activationFunction = ActivationFunctionsProvider::fromName[activationFunctionName];
    this->activationFunctionPrime = ActivationFunctionsProvider::fromName[activationFunctionName + "Prime"];

    this->lossFunction = LossFunctionsProvider::fromName[lossFunctionName];
}
