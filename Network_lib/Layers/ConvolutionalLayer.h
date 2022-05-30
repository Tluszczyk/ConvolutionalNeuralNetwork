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
    Tensor filterWeights;
    Tensor biases;
    int stride;
    int pad;


};


#endif //NEURALNET_CONVOLUTIONALLAYER_H
