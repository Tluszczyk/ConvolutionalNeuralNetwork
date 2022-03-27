#include <iostream>

#include "Tensor.h"
#include "DenseLayer.h"

int main() {

    Tensor input({2}, {.5, .9});

    DenseLayer denseLayer(2);
    denseLayer.compile(0.9, 3);

    Tensor output = denseLayer.feed(input);

    std::cout << output;

    return 0;
}
