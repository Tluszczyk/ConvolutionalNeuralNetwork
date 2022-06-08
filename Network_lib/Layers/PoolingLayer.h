//
// Created by Filip Tłuszcz on 31.05.2022.
//

#ifndef NEURALNET_POOLINGLAYER_H
#define NEURALNET_POOLINGLAYER_H

#include "Layer.h"

class PoolingLayer : public Layer {
public:
    PoolingLayer(
            std::vector<int> shape,
            const string &activationFunctionName,
            std::string layerName,
            int poolingSize,
            std::string poolingType = "max");

    Tensor feed(Tensor inputTensor) override;

protected:
    [[nodiscard]] LayerType GET_LAYER_TYPE() const override;

private:
    int poolingSize;
    std::string poolingType;
    function<double(const Tensor&)> poolingFuntion;
};


#endif //NEURALNET_POOLINGLAYER_H
