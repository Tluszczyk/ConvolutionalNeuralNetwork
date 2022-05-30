//
// Created by kubkm on 30.05.2022.
//

#ifndef NEURALNET_CONVOLUTIONALLAYER_H
#define NEURALNET_CONVOLUTIONALLAYER_H


#include "Layer.h"

class ConvolutionalLayer : public Layer {
public:
    Tensor feed(Tensor inputTensor) override;
private:
    vector<Tensor> filters;
    Tensor biases;
    int stride;
    int pad;


    Tensor addPadding(const Tensor &input);
};


#endif //NEURALNET_CONVOLUTIONALLAYER_H
