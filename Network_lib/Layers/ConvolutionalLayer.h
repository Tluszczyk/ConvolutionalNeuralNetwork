//
// Created by kubkm on 30.05.2022.
//

#ifndef NEURALNET_CONVOLUTIONALLAYER_H
#define NEURALNET_CONVOLUTIONALLAYER_H


#include "Layer.h"

class ConvolutionalLayer : public Layer {
public:
    Tensor feed(Tensor inputTensor) override;
    Tensor backpropagate(const Tensor &gradient) override;
private:
    vector<Tensor> filters;
    vector<float> biases;
    int stride;
    int pad;

    int noOfFilters;

    Tensor previousInput;

    Tensor db;
    Tensor dw;
    Tensor dx;


    Tensor addPadding(const Tensor &input);
};


#endif //NEURALNET_CONVOLUTIONALLAYER_H
