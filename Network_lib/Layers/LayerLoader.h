//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_LAYERLOADER_H
#define NEURALNET_LAYERLOADER_H

#include <fstream>
#include "DenseLayer.h"

using namespace std;

class DenseLayer;

class LayerLoader {
public:
    static void saveDenseLayer(ofstream& modelFile, DenseLayer* layer);
};


#endif //NEURALNET_LAYERLOADER_H
