//
// Created by kubkm on 30.05.2022.
//
#include "gtest/gtest.h"
#include "Tensor.h"
#include "Layers/ConvolutionalLayer.h"


TEST(UtilityOpertionsSuite, Pad2D) {
    Tensor input({3, 2}, {1, 1, 1, 2, 2, 2});

    auto* conv = new ConvolutionalLayer({3, 2}, "sig", "Conv2Test", 1, {3,3});

    Tensor padded = conv->TEST_addPadding(input);

    Tensor correctlyPadded({5, 4}, {
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 2, 2, 2, 0,
        0, 0, 0, 0, 0
    });

    ASSERT_EQ(padded, correctlyPadded);
}

TEST(UtilityOpertionsSuite, Pad3D) {
    Tensor input({3, 3, 3}, {
        1, 1, 1,
        2, 2, 2,
        3, 3, 3,

        1, 1, 1,
        2, 2, 2,
        3, 3, 3,

        1, 1, 1,
        2, 2, 2,
        3, 3, 3,
    });

    auto* conv = new ConvolutionalLayer({3, 3, 3}, "sig", "Conv3Test", 1, {3,3});

    Tensor padded = conv->TEST_addPadding(input);

    Tensor correctlyPadded({5, 5, 3}, {
        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 2, 2, 2, 0,
        0, 3, 3, 3, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 2, 2, 2, 0,
        0, 3, 3, 3, 0,
        0, 0, 0, 0, 0,

        0, 0, 0, 0, 0,
        0, 1, 1, 1, 0,
        0, 2, 2, 2, 0,
        0, 3, 3, 3, 0,
        0, 0, 0, 0, 0,
    });

    ASSERT_EQ(padded, correctlyPadded);
}
