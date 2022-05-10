//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_ACTIVATIONFUNCTIONSPROVIDER_H
#define NEURALNET_ACTIVATIONFUNCTIONSPROVIDER_H

#include <map>
#include <string>
#include <functional>

using namespace std;

class ActivationFunctionsProvider {
public:
    static double id(double x);
    static double bin(double x);
    static double sig(double x);
    static double tanh(double x);
    static double relu(double x);
    static double softplus(double x);

    static double idPrime(double x);
    static double binPrime(double x);
    static double sigPrime(double x);
    static double tanhPrime(double x);
    static double reluPrime(double x);
    static double softplusPrime(double x);

    static map<string, function<double(double)>> fromName;
    static map<string, function<double(double)>> derivativeFromName;
};

#endif //NEURALNET_ACTIVATIONFUNCTIONSPROVIDER_H
