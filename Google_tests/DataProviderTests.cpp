//
// Created by Filip Tłuszcz on 09.05.2022.
//

#include "gtest/gtest.h"

#include <DataProvider.h>

TEST(dataTestSuite, XOR) {
    auto [X, y] = DataProvider::getXorData();

    for(int i=0; i<X.size(); i++) {
        int a = X[i][{0}];
        int b = X[i][{1}];
        int res0 = y[i][{0}];
        int res1 = y[i][{1}];

        ASSERT_EQ(!a^b, res0);
        ASSERT_EQ(a^b, res1);
    }
}