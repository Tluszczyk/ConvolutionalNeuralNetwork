#include <iostream>

#include "Tensor.h"
#include "DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "OutputLayer.h"

using namespace std;

int main() {
//    auto *model = new Sequential({
//                                         new DenseLayer({2}, "sig", "Dense input"),
//                                         new DenseLayer({4}, "sig", "Dense hidden 1"),
//                                         new DenseLayer({4}, "sig", "Dense hidden 2"),
//                                         new DenseLayer({2}, "sig", "Dense output"),
//                                 });
//    model->compile(.7, "MSE");
//
//    Tensor t({2}, {.5, .6});
//
//    cout << model->feed(t) << endl;
//
//    //ModelLoader::saveToFile(*model, "model1.mdl");
//
//    delete model;
//
    auto *model2 = new Sequential({
                                         new DenseLayer({2}, "id", "Dense input"),
                                         // new DenseLayer({4}, "id", "Idk"),
                                         new DenseLayer({4}, "id", "Idk"),
                                         new OutputLayer({2}, "id", "Dense output"),
                                 });

    model2->compile(0.01);

    //((DenseLayer*)(model2->layers[0]))->changeWeightsTensor(Tensor({2, 2}, {0.5, 0.5, 0.5, 0.5}));
    //((DenseLayer*)(model2->layers[0]))->changeBiasTensor(Tensor({2}, {0, 0}));

    Tensor result0 = model2->feed(Tensor({2}, {0, 1}));
    cout<<result0<<endl;

    result0 = model2->feed(Tensor({2}, {1,1}));
    cout<<result0<<endl;

    Tensor result1 = model2->feed(Tensor({2}, {0,0}));
    cout<<result1<<endl;

    vector<Tensor> X = {Tensor({2}, {0,1}), Tensor({2}, {1,0}),
                        Tensor({2}, {1,1}), Tensor({2}, {0,0})};


    vector<Tensor> X1 = {Tensor({2}, {0,1})};
    vector<Tensor> X2 = {Tensor({2}, {1,0})};
    vector<Tensor> X3 = {Tensor({2}, {1,1})};
    vector<Tensor> X4 = {Tensor({2}, {0,0})};



    vector<Tensor> Y = {Tensor({2}, {1, 0}), Tensor({2}, {1, 0}),
                        Tensor({2}, {1, 1}), Tensor({2}, {0, 0})};

    vector<Tensor> Y1 = {Tensor({2}, {1,0})};
    vector<Tensor> Y2 = {Tensor({2}, {2,4})};
    vector<Tensor> Y3 = {Tensor({2}, {4,2})};
    vector<Tensor> Y4 = {Tensor({2}, {2,0})};

    vector<Tensor> Xs[4] = {X1, X2, X3, X4};
    vector<Tensor> Ys[4] = {Y1, Y2, Y3, Y4};



    for (int i = 0; i< 10000; i++) {
        int tmp = rand()%4;
        model2->analyzeBatch(Xs[tmp], Ys[tmp]);
        //model2->analyzeBatch(X2, Y2);
        //model2->analyzeBatch(X3, Y3);
        //model2->analyzeBatch(X4, Y4);
    }

    Tensor result = model2->feed(Tensor({2}, {0,1}));
    cout<<result<<endl;

    result = model2->feed(Tensor({2}, {1,1}));
    cout<<result<<endl;

    Tensor result2 = model2->feed(Tensor({2}, {0,0}));
    cout<<result2<<endl;

    delete model2;

//    Tensor input({2}, {.5, .9});
//
//    DenseLayer denseLayer({2}, "relu", "WHY");
//    denseLayer.compile(1.0, {3});
//    //denseLayer.changeWeightsTensor(Tensor({2, 3}, {0.2, -0.3, 0.4, -0.5, 0.6, -0.7}));
//    //denseLayer.changeBiasTensor(Tensor({3}, {-0.5, 0, 0.5}));
//    Tensor output = denseLayer.feed(input);
//
//    Tensor nextLayerChanges({3}, {0.4, 0.3, -0.15});
//    Tensor changes = denseLayer.backpropagate(nextLayerChanges);
//    denseLayer.applyChanges();
//
//    cout<<
//
//    std::cout << output <<endl;
//    std::cout << changes <<endl;
//    std::cout << nextLayerChanges <<endl;

    return 0;
}
