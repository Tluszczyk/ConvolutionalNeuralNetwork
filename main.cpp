#include <iostream>

#include "Tensor.h"
#include "DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"

using namespace std;

int main() {
    auto *model = new Sequential({
        new DenseLayer({2}, "sig", "Dense input"),
        new DenseLayer({2}, "sig", "Dense output")
    });
    model->compile(.7, "MSE");

    Tensor t({2}, {.5, .6});

    cout << model->feed(t) << endl;

//    ModelLoader::saveToFile(*model, "model1.mdl");

    delete model;

    return 0;
}
