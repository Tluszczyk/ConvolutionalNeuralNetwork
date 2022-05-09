#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

using namespace std;

int main(int argc, char* argv[]) {

    Sequential* xor_model = ModelLoader::loadFromFile("/Users/tluszczyk/dev/AGH/VI sem/ZSC/NeuralNet/xor_model.mdl");

    Tensor result = xor_model->feed(Tensor({2}, {0,1}));
    cout << result << endl;

    result = xor_model->feed(Tensor({2}, {1,0}));
    cout << result << endl;

    result = xor_model->feed(Tensor({2}, {1,1}));
    cout << result << endl;

    result = xor_model->feed(Tensor({2}, {0,0}));
    cout << result << endl;

    delete xor_model;

    return 0;
}
