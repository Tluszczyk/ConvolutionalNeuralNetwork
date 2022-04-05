//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "gtest/gtest.h"
#include "Tensor.h"
#include "DenseLayer.h"

TEST(FlowOperationSuite, FeedForward) {
    Tensor input({2}, {.5, .9});

    DenseLayer denseLayer({2}, "relu", "WHY");
    denseLayer.compile(0.9, {3});

    Tensor output = denseLayer.feed(input);

    std::cout << output;
}

TEST(FlowOperationSuite, Backpropagation) {
    Tensor input({2}, {.5, .9});

    DenseLayer denseLayer({2}, "relu", "WHY");
    denseLayer.compile(1.0, {3});
    denseLayer.changeWeightsTensor(Tensor({2, 3}, {0.2, -0.3, 0.4, -0.5, 0.6, -0.7}));
    denseLayer.changeBiasTensor(Tensor({3}, {-0.5, 0, 0.5}));
    Tensor output = denseLayer.feed(input);

    Tensor nextLayerChanges({3}, {0.4, 0.3, -0.15});
    Tensor changes = denseLayer.backpropagate(nextLayerChanges);
    std::cout << output <<endl;
    std::cout << changes <<endl;
    std::cout << nextLayerChanges <<endl;
}