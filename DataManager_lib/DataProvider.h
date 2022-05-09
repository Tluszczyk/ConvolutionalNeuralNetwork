//
// Created by Filip Tłuszcz on 09.05.2022.
//

#ifndef NEURALNET_DATAPROVIDER_H
#define NEURALNET_DATAPROVIDER_H

#include <vector>

#include "Tensor.h"

class DataProvider {
public:
    static std::pair<std::vector<Tensor>, std::vector<Tensor>> getXorData();
};


#endif //NEURALNET_DATAPROVIDER_H
