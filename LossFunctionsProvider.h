//
// Created by Filip Tłuszcz on 31.03.2022.
//

#ifndef NEURALNET_LOSSFUNCTIONSPROVIDER_H
#define NEURALNET_LOSSFUNCTIONSPROVIDER_H

#include <map>
#include <string>
#include <functional>

using namespace std;

class LossFunctionsProvider {
public:
    static double MSE(double, double);

    static double MSEPrime(double, double);

    static map<string, function<double(double, double)>> fromName;
};


#endif //NEURALNET_LOSSFUNCTIONSPROVIDER_H
