#include <iostream>

#include "Tensor.h"
#include "DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"

using namespace std;

int main() {
    auto *model = new Sequential({
        new DenseLayer(2, "sig", "Dense input"),
        new DenseLayer(4, "sig", "Dense hidden 1"),
        new DenseLayer(4, "sig", "Dense hidden 2"),
        new DenseLayer(2, "sig", "Dense output"),
    });

    model->compile(.7, "MSE");

    Tensor t({2}, {.5, .6});

    cout << model->feed(t) << endl;

    delete model;

    return 0;
}
