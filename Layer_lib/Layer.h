//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_LAYER_H
#define NEURALNET_LAYER_H

#include "Tensor.h"

class Layer {
private:

protected:
    virtual Tensor feed(Tensor inputTensor) = 0;
};


#endif //NEURALNET_LAYER_H
