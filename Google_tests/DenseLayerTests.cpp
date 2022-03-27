//
// Created by Filip Tłuszcz on 26.03.2022.
//

#include "gtest/gtest.h"
#include "Tensor.h"
#include "DenseLayer.h"

TEST(FlowOperationSuite, FeedForward) {
    Tensor input({2}, {.5, .9});

    DenseLayer denseLayer(2);
    denseLayer.compile(0.9, 3);

    Tensor output = denseLayer.feed(input);

    std::cout << output;
}