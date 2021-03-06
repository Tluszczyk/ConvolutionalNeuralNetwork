//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include <cmath>
#include <functional>
#include "ActivationFunctionsProvider.h"

double ActivationFunctionsProvider::id(double x) { return x; }
double ActivationFunctionsProvider::bin(double x) { return (x<0 ? 0:1); }
double ActivationFunctionsProvider::sig(double x) { return 1./(1.+exp(-x)); }
double ActivationFunctionsProvider::tanh(double x) { return (exp(x)-exp(-x))/(exp(x)+exp(-x)); }
double ActivationFunctionsProvider::relu(double x) { return (x<0 ? 0:x); }
double ActivationFunctionsProvider::softplus(double x) { return log(1+exp(x)); }

double ActivationFunctionsProvider::idPrime(double x) { return 1; }
double ActivationFunctionsProvider::binPrime(double x) { return 0; }
double ActivationFunctionsProvider::sigPrime(double x) { return exp(-x) / pow(1. + exp(-x), 2); }
double ActivationFunctionsProvider::tanhPrime(double x) { return 1 - pow((exp(x) - exp(-x)) / (exp(x) + exp(-x)), 2); }
double ActivationFunctionsProvider::reluPrime(double x) { return (x < 0 ? 0 : 1); }
double ActivationFunctionsProvider::softplusPrime(double x) { return 1. / (1. + exp(-x)); }

map<string, function<double(double)>> ActivationFunctionsProvider::fromName = {
        {"id", ActivationFunctionsProvider::id},
        {"bin", ActivationFunctionsProvider::bin},
        {"sig", ActivationFunctionsProvider::sig},
        {"tanh", ActivationFunctionsProvider::tanh},
        {"relu", ActivationFunctionsProvider::relu},
        {"softplus", ActivationFunctionsProvider::softplus},

        {"idPrime", ActivationFunctionsProvider::idPrime},
        {"binPrime", ActivationFunctionsProvider::binPrime},
        {"sigPrime", ActivationFunctionsProvider::sigPrime},
        {"tanhPrime", ActivationFunctionsProvider::tanhPrime},
        {"reluPrime", ActivationFunctionsProvider::reluPrime},
        {"softplusPrime", ActivationFunctionsProvider::softplusPrime},
};

map<string, function<double(double)>> ActivationFunctionsProvider::derivativeFromName = {
        {"id", ActivationFunctionsProvider::idPrime},
        {"bin", ActivationFunctionsProvider::binPrime},
        {"sig", ActivationFunctionsProvider::sigPrime},
        {"tanh", ActivationFunctionsProvider::tanhPrime},
        {"relu", ActivationFunctionsProvider::reluPrime},
        {"softplus", ActivationFunctionsProvider::softplusPrime},
};