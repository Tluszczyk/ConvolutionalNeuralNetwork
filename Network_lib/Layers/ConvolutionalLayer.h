//
// Created by kubkm on 30.05.2022.
//

#ifndef NEURALNET_CONVOLUTIONALLAYER_H
#define NEURALNET_CONVOLUTIONALLAYER_H


#include "Layer.h"

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(const vector<int> &shape, const string &activationFunctionName, const string &layerName,
                       int noOfFilters, const vector<int> &filerShape);

    void compile(double learningRate1, const vector<int>& nextLayerShape) override;
    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;
    [[nodiscard]] const Tensor &getActivations() const { return activations; };

    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(const Tensor& gradient) override;
    virtual void applyChanges();
private:
    vector<Tensor> filters;
    vector<double> biases;
    int stride;
    int pad;

    int noOfFilters;

    Tensor previousInput;

    Tensor db;
    vector<Tensor> dw;
    Tensor dx;

    int backPropagationsCarriedOut;
    double learningRate{};

    vector<int> filerShape;


    Tensor addPadding(const Tensor &input);
};


#endif //NEURALNET_CONVOLUTIONALLAYER_H
