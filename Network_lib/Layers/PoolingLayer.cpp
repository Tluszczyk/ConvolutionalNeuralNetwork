//
// Created by Filip Tłuszcz on 31.05.2022.
//

#include "PoolingLayer.h"
#include "PoolingFunctionProvider.h"

#include <utility>

PoolingLayer::PoolingLayer(
        std::vector<int> shape,
        const string &activationFunctionName,
        std::string layerName,
        int poolingSize,
        std::string poolingType
) : Layer(std::move(shape), activationFunctionName, std::move(layerName)),
    poolingSize(poolingSize),
    poolingType(std::move(poolingType)),
    poolingFuntion(PoolingFunctionProvider::fromName[poolingType])
{}

Tensor PoolingLayer::feed(Tensor inputTensor) {
    return Tensor();
}

LayerType PoolingLayer::GET_LAYER_TYPE() const { return Pooling; }