//
// Created by kubkm on 30.05.2022.
//
#include "gtest/gtest.h"
#include "Tensor.h"
#include "Layers/ConvolutionalLayer.h"

TEST(UtilityOpertionsSuite, Pad) {
    Tensor input({3, 2}, {1, 1, 1, 2, 2, 2});

    std::cout<<input<<endl;
}

