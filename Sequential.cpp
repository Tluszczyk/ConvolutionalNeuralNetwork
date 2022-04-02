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

    for(auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++)
        if( layerIt+1 != layers.end() ) inputTensor = (*layerIt)->feed(inputTensor);

    return inputTensor;
}

void Sequential::backpropagate(Tensor gradient) {
    /* adjust all the adjustable parameters
     * */

}

void Sequential::compile(double learningRate, const string &lossFunctionName) {
    this->learningRate = learningRate;

    this->lossFunction = LossFunctionsProvider::fromName[lossFunctionName];

    for(int i=0; i<layers.size(); i++)
        layers[i]->compile(learningRate, i<layers.size()-1 ? layers[i+1]->getShape() : vector<int>{0});
}
