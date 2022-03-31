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

double (*LossFunctionsProvider::fromName(const string& name))(double, double) {
    if( name == "MSE" ) return LossFunctionsProvider::MSE;
    if( name == "MSEPrime" ) return LossFunctionsProvider::MSEPrime;

    return nullptr;
}