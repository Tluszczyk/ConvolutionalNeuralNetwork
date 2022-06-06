//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_LAYERTYPE_H
#define NEURALNET_LAYERTYPE_H

#include <string>

using namespace std;

enum LayerType { Dense, Output, Convolutional, Pooling, None };

inline string LayerTypeToString(const LayerType &layerType) {
    switch(layerType) {
        case Dense: return "DENSE_LAYER_TYPE";
        case Output: return "OUTPUT_LAYER_TYPE";
        case Convolutional: return "CONVOLUTIONAL_LAYER_TYPE";
        case Pooling: return "POOLING_LAYER_TYPE";
        default:    return "";
    }
}

inline LayerType StringToLayerType(const string &layerType) {
    if( layerType == "DENSE_LAYER_TYPE" ) return Dense;
    if( layerType == "OUTPUT_LAYER_TYPE" ) return Output;
    if( layerType == "CONVOLUTIONAL_LAYER_TYPE" ) return Convolutional;
    if( layerType == "POOLING_LAYER_TYPE" ) return Pooling;
    return None;
}

#endif //NEURALNET_LAYERTYPE_H