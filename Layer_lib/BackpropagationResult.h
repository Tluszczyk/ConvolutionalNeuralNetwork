//
// Created by kubkm on 02.04.2022.
//

#include <utility>

#include "Tensor.h"

#ifndef NEURALNET_BACKPROPAGATIONRESULT_H
#define NEURALNET_BACKPROPAGATIONRESULT_H

#endif //NEURALNET_BACKPROPAGATIONRESULT_H


class BackpropagationResult {
private:

protected:

public:
    Tensor activationChanges;
    Tensor weightChanges;
    Tensor biasChanges;

    explicit BackpropagationResult(Tensor activationChanges, Tensor weightChanges, Tensor biasChanges) : activationChanges(std::move(activationChanges)), biasChanges(std::move(biasChanges)), weightChanges(std::move(weightChanges)) {};

};