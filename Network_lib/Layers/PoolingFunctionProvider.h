//
// Created by Filip Tłuszcz on 06.06.2022.
//

#ifndef NEURALNET_POOLINGFUNCTIONPROVIDER_H
#define NEURALNET_POOLINGFUNCTIONPROVIDER_H


#include <map>
#include <string>
#include <functional>

#include "Tensor.h"

using namespace std;

class PoolingFunctionProvider {
public:
    static double max(const Tensor &in);
    static double avg(const Tensor &in);
    static double min(const Tensor &in);

    static map<string, function<double(const Tensor&)>> fromName;
};


#endif //NEURALNET_POOLINGFUNCTIONPROVIDER_H