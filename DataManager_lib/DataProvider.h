//
// Created by Filip Tłuszcz on 09.05.2022.
//

#ifndef NEURALNET_DATAPROVIDER_H
#define NEURALNET_DATAPROVIDER_H

#include <vector>

#include "Tensor.h"

typedef std::pair<std::vector<Tensor>, std::vector<Tensor>> dataset;
typedef std::pair<std::vector<Tensor>, std::vector<std::string>> labeled_dataset;

class DataProvider {
public:
    static dataset getXorData();
    static dataset getMnistHRDData(const string& mnistPath);
};


#endif //NEURALNET_DATAPROVIDER_H
