//
// Created by Filip Tłuszcz on 25.03.2022.
//

#include "Sequential.h"
#include "LossFunctionsProvider.h"
#include "Layers/DenseLayer.h"

using namespace std;

Tensor Sequential::feed(Tensor inputTensor) {
    /* feed a single tensor to a model,
     * and get the activations of the last layer
     * */

    for(auto layerIt = layers.begin(); layerIt != layers.end(); layerIt++)
        inputTensor = (*layerIt)->feed(inputTensor);

    return inputTensor;
}

void Sequential::backpropagate(const Tensor& costDerivative) {
    /* adjust all the adjustable parameters
     * */
    Tensor changes = costDerivative.copy();
    for(auto layerIt = layers.end() - 2; layerIt + 1 != layers.begin(); layerIt--)
        changes = ((DenseLayer*)(*layerIt))->backpropagate(changes);
}

void Sequential::compile(double learningRate, const string &lossFunctionName) {
    this->learningRate = learningRate;

    this->lossFunction = LossFunctionsProvider::fromName[lossFunctionName];

    for(int i=0; i<layers.size(); i++)
        layers[i]->compile(learningRate, i<layers.size()-1 ? layers[i+1]->getShape() : vector<int>{0});
}

void Sequential::applyChanges(){
    for(auto layerIt = layers.begin(); layerIt != layers.end() - 1; layerIt++){
        ((DenseLayer*)(*layerIt))->applyChanges();
    }
}

Tensor Sequential::calculateLoss(const Tensor& expected) {

    Tensor lastLayerActivations = ((DenseLayer*)layers.back())->getActivations();

    if( expected.getShape() != lastLayerActivations.getShape() ) throw range_error("Model's output and expected output's shapes don't match");

    // TODO
    return Tensor({}, {});
}

Tensor getLossDerivative(const Tensor& expected, const Tensor& actual){
    return (actual - expected) * 2;
}

void Sequential::analyzeBatch(vector<Tensor> batch, vector<Tensor> expectedResults) {
    for (int i = 0; i < batch.size(); i++){
        Tensor result = feed(batch[i]);
        Tensor costDerivative = getLossDerivative(expectedResults[i], result);
        backpropagate(costDerivative);
    }

    applyChanges();
}