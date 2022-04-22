//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_LAYERTYPE_H
#define NEURALNET_LAYERTYPE_H

#include <string>

using namespace std;

enum LayerType { Dense, Output };

inline string LayerTypeToString(const LayerType &layerType) {
    switch(layerType) {
        case Dense: return "DENSE_LAYER_TYPE";
        case Output: return "OUTPUT_LAYER_TYPE";
        default:    return "";
    }
}

#endif //NEURALNET_LAYERTYPE_H