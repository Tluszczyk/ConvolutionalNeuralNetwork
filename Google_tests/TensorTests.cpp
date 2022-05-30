//
// Created by Filip Tłuszcz on 25.03.2022.
//

#include "gtest/gtest.h"
#include "Tensor.h"
#include "ActivationFunctionsProvider.h"

#include <functional>

TEST(LinearOperationsSuite, AddSub) {
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({2,3}, {6,5,4,3,2,1});

    Tensor aPlusBCorrect{{2,3}, {7,7,7,7,7,7}};
    Tensor aMinusBCorrect{{2,3}, {-5,-3,-1,1,3,5}};

    ASSERT_EQ(a+b, aPlusBCorrect);
    ASSERT_EQ(a-b, aMinusBCorrect);
}

TEST(LinearOperationsSuite, MatMulTimes) {
    Tensor a({2,3}, {1,2,3,4,5,6});
    Tensor b({2,3}, {6,5,4,3,2,1});
    Tensor c({3,2}, {6,5,4,3,2,1});

    Tensor aTimesBCorrect{{2,3}, {6,10,12,12,10,6}};
    Tensor aMulCCorrect{{3,3}, {12,  9,  6, 30, 23, 16, 48, 37, 26}};

    ASSERT_EQ(a*b, aTimesBCorrect);
    ASSERT_EQ(a^c, aMulCCorrect);
}

TEST(LinearOperationsSuite, MatMulVec) {
    Tensor vec = Tensor({2}, {.5, .6}).transpose({1,0});
    Tensor mat({2,3}, {.1, .5, .8, .2, .5, .9});

    Tensor matMulVecCorrect = Tensor({3}, {.35, .52, .79});

    ASSERT_EQ((mat^vec).reshape({3}), matMulVecCorrect);
}

TEST(LogicalOperationsSuite, ApplyFunction) {
    Tensor a({7}, {-2, -1, 0, 1, 2, 3, 4});

    Tensor b = a.map((ActivationFunctionsProvider::relu));
    Tensor correct({7}, {0, 0, 0, 1, 2, 3, 4});

    ASSERT_EQ(b, correct);
}

TEST(LogicalOperationsSuite, Subscript) {
    Tensor a({3,3,3}, {
        1,2,3,
        4,5,6,
        7,8,9,

        10,11,12,
        13,14,15,
        16,17,18,

        19,20,21,
        22,23,24,
        25,26,27
    });

    for(int x=0; x<3; x++)
        for(int y=0; y<3; y++)
            for(int z=0; z<3; z++)
                ASSERT_EQ((a[{x,y,z}]), 9*z + 3*y + x + 1);
}

TEST(LogicalOperationsSuite, Transposition) {
    Tensor a({4}, {1,2,3,4});
    Tensor b({2,3}, {1,2,3,4,5,6});
    Tensor c({3,3}, {1,2,3,4,5,6,7,8,9});

    Tensor d = a.copy();

    ASSERT_EQ(a.transpose({1, 0}), Tensor({1, 4}, {1,2,3,4}));
    ASSERT_EQ(a, d);
    ASSERT_EQ(b.transpose({1, 0}), Tensor({3, 2}, {1,3,5,2,4,6}));
    ASSERT_EQ(c.transpose({1, 0}), Tensor({3, 3}, {1,4,7,2,5,8,3,6,9}));
}

TEST(LogicalOperationsSuite, Copying) {
    Tensor a({4}, {1,2,3,4});
    Tensor b = a.copy();
    Tensor c({4}, {1,2,3,4});
    ASSERT_EQ(a, b);
    a[{3}] = 0;
    ASSERT_EQ(b, c);
}

TEST(LogicalOperationsSuite, Reshaping) {

}

TEST(LogicalOperationsSuite, Random) {
    Tensor a = Tensor::createRandom({3, 3, 3}, 1);
//    std::cout << a << std::endl;
}

TEST(LogicalOperationsSuite, Map) {
    Tensor c({3,3}, {1,2,3,4,5,6,7,8,9});
    function<double(double)> op = [](double d)  -> double { return 2*d; };

    ASSERT_EQ(c.map(op), Tensor({3,3}, {2,4,6,8,10,12,14,16,18}));
}

TEST(PresentationSuite, Stringifying) {
    Tensor a({4}, {1,2,3,4});
    Tensor b({2,3}, {1,2,3,4,5,6});
    Tensor c({3,3}, {1,2,3,4,5,6,7,8,9});

    ASSERT_EQ(a.to_string(), "{1,2,3,4}");
    ASSERT_EQ(b.to_string(), "{{1,2},{3,4},{5,6}}");
    ASSERT_EQ(c.to_string(), "{{1,2,3},{4,5,6},{7,8,9}}");
}