//
// Created by Filip Tłuszcz on 31.03.2022.
//

#ifndef NEURALNET_LOSSFUNCTIONSPROVIDER_H
#define NEURALNET_LOSSFUNCTIONSPROVIDER_H

#include "Tensor.h"

using namespace std;

class LossFunctionsProvider {
public:
    static double MSE(double, double);

    static double MSEPrime(double, double);

    static double (*fromName(const string& name))(double, double);
};


#endif //NEURALNET_LOSSFUNCTIONSPROVIDER_H
