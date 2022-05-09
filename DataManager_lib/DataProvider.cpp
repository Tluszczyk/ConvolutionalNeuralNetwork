//
// Created by Filip Tłuszcz on 09.05.2022.
//

#include "DataProvider.h"
#include <random>

std::pair<std::vector<Tensor>, std::vector<Tensor> > DataProvider::getXorData() {
    std::vector<Tensor> inputs, outputs;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> binDist(0,1);

    for(int i=0; i<10000; i++) {
        double a = binDist(rng);
        double b = binDist(rng);

        double axb = (int)(a+b)%2;
        double anxb = (int)(axb+1)%2;

        inputs.push_back(Tensor({2}, {a,b}));
        outputs.push_back(Tensor({2}, {anxb, axb}));
    }
    return std::pair(inputs, outputs);
}