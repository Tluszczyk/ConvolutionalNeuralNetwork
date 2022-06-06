//
// Created by kubkm on 30.05.2022.
//

#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(
        vector<int> shape,
        const string &activationFunctionName,
        string layerName,
        int filterCount,
        int filterSize,
        int paddingSize,
        int stride
        ) : Layer(std::move(shape), activationFunctionName, std::move(layerName))
{
    this->filterCount = filterCount;
    this->filterSize = filterSize;
    this->pad = paddingSize == -1 ? filterSize/2 : paddingSize;
    this->stride = stride;
}

Tensor ConvolutionalLayer::feed(Tensor inputTensor) {
    vector<Tensor> filteringResults;
    filteringResults.reserve(filterCount);
    for (int i = 0 ; i < filterCount; i++){
        filteringResults.push_back(inputTensor.convolve(filters[i]));
    }
    return Tensor::joinTensors(filteringResults);
}

Tensor ConvolutionalLayer::addPadding(const Tensor& input) {

    vector<int> oldShape = input.getShape();
    if (oldShape.size() < 2 or oldShape.size() > 3){
        throw range_error("addPadding: Expected 2 or 3 dimensional input, instead got " +
        std::to_string(oldShape.size()) + " dimensional input");
    }
    vector<int> newShape;
    newShape.reserve(oldShape.size());
    for(int i=0; i<oldShape.size(); i++)
        newShape.push_back(i < 2 ? oldShape[i] + 2*pad : oldShape[i]);

    int old2Size = oldShape[0] * oldShape[1];

    int newSize = 1;
    for(int i : newShape) newSize *= i;

    vector<double> resultingData;
    resultingData.reserve(newSize);
    int thirdDimSize = newShape.size() > 2 ? newShape[2] : 1;
    for(int d = 0; d < thirdDimSize; d++) {
        for (int i = 0; i < pad; i++) {
            for (int k = 0; k < newShape[0]; k++) {
                resultingData.push_back(0);
            }
        }

        for (int i = 0; i < oldShape[1]; i++){
            for (int k = 0; k < pad; k++) {
                resultingData.push_back(0);
            }
            for (int k = 0; k < oldShape[0]; k++){
                resultingData.push_back(input.getData()[d * old2Size + i * oldShape[0] + k]);
            }
            for (int k = 0; k < pad; k++) {
                resultingData.push_back(0);
            }
        }

        for (int i = 0; i < pad; i++) {
            for (int k = 0; k < newShape[0]; k++) {
                resultingData.push_back(0);
            }
        }
    }

    return Tensor(newShape, resultingData);
}

LayerType ConvolutionalLayer::GET_LAYER_TYPE() const { return Convolutional; }

Tensor ConvolutionalLayer::TEST_addPadding(const Tensor &input) {
    return this->addPadding(input);
}
