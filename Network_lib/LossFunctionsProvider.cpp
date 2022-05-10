//
// Created by Filip Tłuszcz on 31.03.2022.
//

#include "LossFunctionsProvider.h"


#include <cmath>
#include <string>
using namespace std;

double LossFunctionsProvider::MSE(double netOutput, double trueOutput) {
    return pow( netOutput - trueOutput, 2);
}

double LossFunctionsProvider::MSEPrime(double netOutput, double trueOutput) {
    return 2*( netOutput - trueOutput);
}

map<string, function<double(double, double)>> LossFunctionsProvider::fromName = {
        {"MSE", LossFunctionsProvider::MSE},

        {"MSEPrime", LossFunctionsProvider::MSEPrime}
};