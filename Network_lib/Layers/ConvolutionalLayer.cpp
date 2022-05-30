//
// Created by kubkm on 30.05.2022.
//

#include "ConvolutionalLayer.h"


Tensor ConvolutionalLayer::feed(Tensor inputTensor) {

}

Tensor ConvolutionalLayer::addPadding(const Tensor& input) {
    vector<int> oldShape = input.getShape();
    if (oldShape.size() != 2){
        throw range_error("addPadding: Expected 2 dimensional input, instead got " +
        std::to_string(oldShape.size()) + " dimensional input");
    }
    vector<int> newShape = {oldShape[0] + 2 * pad, oldShape[1] + 2 * pad};
    vector<double> resultingData;
    resultingData.reserve(newShape[0] * newShape[1]);
    for (int i = 0; i < pad; i++) {
        for (int k = 0; k < oldShape[0]; k++) {
            resultingData.push_back(0);
        }
    }

    for (int i = 0; i < oldShape[1]; i++){
        for (int k = 0; k < pad; k++) {
            resultingData.push_back(0);
        }
        for (int k = 0; k < oldShape[0]; k++){
            resultingData.push_back(input.getData()[i * oldShape[1] + k]);
        }
        for (int k = 0; k < pad; k++) {
            resultingData.push_back(0);
        }
    }

    for (int i = 0; i < pad; i++) {
        for (int k = 0; k < oldShape[0]; k++) {
            resultingData.push_back(0);
        }
    }

    return Tensor(newShape, resultingData);
}
