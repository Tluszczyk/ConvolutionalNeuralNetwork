//
// Created by kubkm on 30.05.2022.
//

#include "ConvolutionalLayer.h"


ConvolutionalLayer::ConvolutionalLayer(const vector<int> &shape, const string &activationFunctionName,
                                       const string &layerName, int noOfFilters,
                                       const vector<int> &filterShape) : Layer(shape, activationFunctionName, layerName),
                                                                         noOfFilters(noOfFilters),
                                                                         filerShape(filterShape) {
    backPropagationsCarriedOut = 0;
    pad = filterShape[0]/2;
}


Tensor ConvolutionalLayer::feed(Tensor inputTensor) {
    previousInput = inputTensor;
    vector<Tensor> filteringResults;
    filteringResults.reserve(noOfFilters);
    for (int i = 0 ; i < noOfFilters; i++){
        filteringResults.push_back(inputTensor.convolve(filters[i]) + biases[i]);
    }
    return Tensor::joinTensors(filteringResults);
}

Tensor ConvolutionalLayer::backpropagate(const Tensor &gradient) {
    //double
    //float db = TODO make backprog for bias (super easy tbh)
    vector<Tensor> gradientsF = Tensor::divideTensor(gradient);
    vector<Tensor> dwF;
    dwF.reserve(gradientsF.size());
    Tensor gradientPadded = gradient; //TODO pad gradient
    vector<Tensor> dxC = Tensor::divideTensor(dx);
    for (int f = 0; f < gradientsF.size(); f++){
        dw[f] = dw[f] + previousInput.convolve(gradientsF[f]);

        vector<Tensor> filtersC = Tensor::divideTensor(filters[f]);
        for (int c = 0; c < filtersC.size(); c++){
            dxC[c] = dxC[c] + gradientPadded.convolve(filtersC[c].reversed());
        }
    }
    dx = Tensor::joinTensors(dxC);
    return dx;
}


void ConvolutionalLayer::compile(double learningRate1, const vector<int>& nextLayerShape) {
    this->learningRate = learningRate1;
    this->nextLayerShape = nextLayerShape;

    this->filters = vector<Tensor>();
    this->dw = vector<Tensor>();
    for (int i = 0; i < noOfFilters; i++) {
        filters.push_back(Tensor::createRandom(filerShape, 1.0 / noOfFilters));
        dw.emplace_back(filerShape);
        biases.push_back(0);
    }
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

void ConvolutionalLayer::applyChanges() {
    if (backPropagationsCarriedOut == 0){
        return;
    }
    for (int f = 0; f < noOfFilters; f++){
        filters[f] = filters[f] - dw[f];
        dw[f] = Tensor(dw[f].getShape());
        //biases[f] -= db[f]; TODO: biases
    }
    backPropagationsCarriedOut = 0;
}

LayerType ConvolutionalLayer::GET_LAYER_TYPE() const {
    return Convolutional;
}


