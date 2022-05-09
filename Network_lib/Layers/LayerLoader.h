//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_LAYERLOADER_H
#define NEURALNET_LAYERLOADER_H

#include <fstream>
#include "DenseLayer.h"
#include "OutputLayer.h"

using namespace std;

class DenseLayer;

class LayerLoader {
private:
    static void saveLayer(ofstream& modelFile, Layer* layer);
public:
    static void saveDenseLayer(ofstream& modelFile, DenseLayer* layer);
    static void saveOutputLayer(ofstream& modelFile, OutputLayer* layer);

    static DenseLayer* loadDenseLayer(ifstream& modelFile);
    static OutputLayer* loadOutputLayer(ifstream& modelFile);
};


#endif //NEURALNET_LAYERLOADER_H
