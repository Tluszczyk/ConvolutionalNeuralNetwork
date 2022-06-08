

//
// Created by kubkm on 12.04.2022.
//

#include "gtest/gtest.h"

#include "Sequential.h"
#include "Layers/DenseLayer.h"
#include "Layers/OutputLayer.h"

#include "ModelLoader.h"

#include <filesystem>


TEST(LearningSuite, XOR){
    auto *model2 = new Sequential({
        new DenseLayer({2}, "sig", "Dense input"),
        new DenseLayer({400}, "sig", "Idk"),
        new DenseLayer({10}, "sig", "Idk"),
        new OutputLayer({2}, "id", "Dense output"),
    });

    model2->compile(0.5);

    vector<Tensor> X1 = {Tensor({2}, {0,1})};
    vector<Tensor> X2 = {Tensor({2}, {1,0})};
    vector<Tensor> X3 = {Tensor({2}, {1,1})};
    vector<Tensor> X4 = {Tensor({2}, {0,0})};

    vector<Tensor> Y1 = {Tensor({2}, {1,0})};
    vector<Tensor> Y2 = {Tensor({2}, {1,0})};
    vector<Tensor> Y3 = {Tensor({2}, {0,1})};
    vector<Tensor> Y4 = {Tensor({2}, {0,1})};

    vector<Tensor> Xs[4] = {X1, X2, X3, X4};
    vector<Tensor> Ys[4] = {Y1, Y2, Y3, Y4};

    for (int i = 0; i< 50000; i++) {
        int tmp = rand()%4;
        cout << model2->analyzeBatch(Xs[tmp], Ys[tmp]) << endl;
    }

    Tensor result = model2->feed(Tensor({2}, {0,1}));
    double res1 = result[{0}];
    double res2 = result[{1}];
    ASSERT_GE(res1, 0.9);
    ASSERT_LE(res2, 0.1);

    result = model2->feed(Tensor({2}, {1,0}));
    res1 = result[{0}];
    res2 = result[{1}];
    ASSERT_GE(res1, 0.9);
    ASSERT_LE(res2, 0.1);

    result = model2->feed(Tensor({2}, {1,1}));
    res1 = result[{0}];
    res2 = result[{1}];
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    Tensor result2 = model2->feed(Tensor({2}, {0,0}));
    res1 = result[{0}];
    res2 = result[{1}];
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    delete model2;
}

TEST(LearningSuite, AND_OR){
    auto *model2 = new Sequential({
        new DenseLayer({2}, "sig", "Dense input"),
        new DenseLayer({4}, "sig", "Idk"),
        //new DenseLayer({4}, "sig", "Idk"),
        new OutputLayer({2}, "id", "Dense output"),
    });

    model2->compile(0.8);

    vector<Tensor> X1 = {Tensor({2}, {0,1})};
    vector<Tensor> X2 = {Tensor({2}, {1,0})};
    vector<Tensor> X3 = {Tensor({2}, {1,1})};
    vector<Tensor> X4 = {Tensor({2}, {0,0})};

    vector<Tensor> Y1 = {Tensor({2}, {0,1})};
    vector<Tensor> Y2 = {Tensor({2}, {0,1})};
    vector<Tensor> Y3 = {Tensor({2}, {1,1})};
    vector<Tensor> Y4 = {Tensor({2}, {0,0})};

    vector<Tensor> Xs[4] = {X1, X2, X3, X4};
    vector<Tensor> Ys[4] = {Y1, Y2, Y3, Y4};

    for (int i = 0; i< 5000; i++) {
        int tmp = rand()%4;
        model2->analyzeBatch(Xs[tmp], Ys[tmp]);
    }

    Tensor result = model2->feed(Tensor({2}, {0,1}));
    double res1 = result[{0, 0}];
    double res2 = result[{1, 0}];
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    result = model2->feed(Tensor({2}, {1,0}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    result = model2->feed(Tensor({2}, {1,1}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_GE(res1, 0.9);
    ASSERT_GE(res2, 0.9);

    result = model2->feed(Tensor({2}, {0,0}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_LE(res1, 0.1);
    ASSERT_LE(res2, 0.1);

    delete model2;
}

TEST(ModelFileSuite, Loading){
    Sequential* xor_model = ModelLoader::loadFromFile("../../Google_tests/xor_model.mdl");

    Tensor result = xor_model->feed(Tensor({2}, {0,1}));
    double res1 = result[{0}];
    double res2 = result[{1}];
    std::cout << result << endl;
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    result = xor_model->feed(Tensor({2}, {1,0}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_LE(res1, 0.1);
    ASSERT_GE(res2, 0.9);

    result = xor_model->feed(Tensor({2}, {1,1}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_GE(res1, 0.9);
    ASSERT_LE(res2, 0.1);

    Tensor result2 = xor_model->feed(Tensor({2}, {0,0}));
    res1 = result[{0, 0}];
    res2 = result[{1, 0}];
    ASSERT_GE(res1, 0.9);
    ASSERT_LE(res2, 0.1);

    delete xor_model;
}

