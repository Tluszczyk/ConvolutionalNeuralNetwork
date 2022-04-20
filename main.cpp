#include <iostream>

#include "Tensor.h"
#include "DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"

using namespace std;

int main() {
//    auto *model = new Sequential({
//        new DenseLayer({2}, "sig", "Dense input"),
//        new DenseLayer({2}, "sig", "Dense output")
//    });
//    model->compile(.7, "MSE");
//
//    Tensor t({2}, {.5, .6});
//
//    cout << model->feed(t) << endl;
//
////    ModelLoader::saveToFile(*model, "model1.mdl");
//
//    delete model;
//
//    auto *model2 = new Sequential({
//                                         new DenseLayer({2}, "id", "Dense input"),
//                                         new DenseLayer({4}, "id", "Idk"),
//                                         new DenseLayer({4}, "id", "Idk"),
//                                         new DenseLayer({1}, "id", "Dense output"),
//                                 });
//
//    model2->compile(0.1);
//
//    Tensor result0 = model2->feed(Tensor({2}, {0,1}));
//    cout<<result0<<endl;
//
//    Tensor result1 = model2->feed(Tensor({2}, {0,0}));
//    cout<<result1<<endl;
//
//    vector<Tensor> X = {Tensor({2}, {0,1}), Tensor({2}, {1,0}),
//                        Tensor({2}, {1,1}), Tensor({2}, {0,0})};
//
//    vector<Tensor> Y = {Tensor({1}, {1}), Tensor({1}, {1}),
//                        Tensor({1}, {1}), Tensor({1}, {0})};
//
//    for (int i = 0; i< 1000; i++)
//        model2->analyzeBatch(X, Y);
//
//    Tensor result = model2->feed(Tensor({2}, {0,1}));
//    cout<<result<<endl;
//
//    result = model2->feed(Tensor({2}, {1,1}));
//    cout<<result<<endl;
//
//    Tensor result2 = model2->feed(Tensor({2}, {0,0}));
//    cout<<result2<<endl;
//
//    delete model2;

    Tensor input({2}, {.5, .9});

    DenseLayer denseLayer({2}, "relu", "WHY");
    denseLayer.compile(1.0, {3});
    denseLayer.changeWeightsTensor(Tensor({2, 3}, {0.2, -0.3, 0.4, -0.5, 0.6, -0.7}));
    denseLayer.changeBiasTensor(Tensor({3}, {-0.5, 0, 0.5}));
    Tensor output = denseLayer.feed(input);

    Tensor nextLayerChanges({3}, {0.4, 0.3, -0.15});
    Tensor changes = denseLayer.backpropagate(nextLayerChanges);
    std::cout << output <<endl;
    std::cout << changes <<endl;
    std::cout << nextLayerChanges <<endl;

    return 0;
}
