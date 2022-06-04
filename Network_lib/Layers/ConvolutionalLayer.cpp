//
// Created by kubkm on 30.05.2022.
//

#include "ConvolutionalLayer.h"


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
    for (auto & i : gradientsF){
        dwF.push_back(previousInput.convolve(i));
    }
    dw = dw + Tensor::joinTensors(dwF);
    Tensor gradientPadded = gradient; //TODO pad gradient
    vector<Tensor> dxC = Tensor::divideTensor(dx);
    for (int f = 0; f < gradientsF.size(); f++){
        vector<Tensor> filtersC = Tensor::divideTensor(filters[f]);
        for (int c = 0; c < filtersC.size(); c++){
            dxC[c] = dxC[c] + gradientPadded.convolve(filtersC[c].reversed());
        }
    }
    dx = Tensor::joinTensors(dxC);

}
