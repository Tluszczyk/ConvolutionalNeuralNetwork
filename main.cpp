#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

#include "DataManager_lib/DataPrep.h"
#include "DataManager_lib/DataProvider.h"

using namespace std;

int main() {
    auto [X, y] = DataProvider::getXorData();

    for(int i=0; i<X.size(); i++) {
        double a = X[i][{0}];
        double b = X[i][{1}];
        double res0 = y[i][{0}];
        double res1 = y[i][{1}];

        cout << "a=" << a << ", b=" << b << ", a!^b=" << res0 << ", a^b=" << res1 << endl;
    }

    return 0;
}
