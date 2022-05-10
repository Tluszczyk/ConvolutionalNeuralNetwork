//
// Created by kubkm on 12.04.2022.
//

#include "gtest/gtest.h"

#include "../Sequential.h"
#include "DenseLayer.h"


TEST(FlowOperationSuite, ApplyBatch) {
    vector<Layer*> layers{};
    Layer *layer = new DenseLayer({1, 1}, "relu", "the only one");
    layers.push_back(layer);
    Sequential sequential(layers);

    sequential.compile();

    vector<Tensor> X = {Tensor({2}, {0,1}), Tensor({2}, {1,0}),
                        Tensor({2}, {1,1}), Tensor({2}, {0,0})};

    vector<Tensor> Y = {Tensor({1}, {1}), Tensor({1}, {1}),
                        Tensor({1}, {0}), Tensor({1}, {0})};

    sequential.analyzeBatch(X, Y);

    Tensor result = sequential.feed(Tensor({2}, {0,1}));

    cout<<result<<endl;
}

