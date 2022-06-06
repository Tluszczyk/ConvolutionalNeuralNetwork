//
// Created by kubkm on 30.05.2022.
//

#ifndef NEURALNET_CONVOLUTIONALLAYER_H
#define NEURALNET_CONVOLUTIONALLAYER_H


#include "Layer.h"

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(
            vector<int> shape,
            const string &activationFunctionName,
            string layerName,
            int filterCount,
            int filterSize,
            int paddingSize=-1,
            int stride=-1);

    Tensor feed(Tensor inputTensor) override;

    Tensor TEST_addPadding(const Tensor &input);

protected:
    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;

private:
    int filterCount;
    int filterSize;
    vector<Tensor> filters;
    Tensor biases;
    int stride;
    int pad;

    Tensor addPadding(const Tensor &input);
};


#endif //NEURALNET_CONVOLUTIONALLAYER_H
