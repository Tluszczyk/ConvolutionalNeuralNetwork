//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "LayerLoader.h"
#include "TensorLoader.h"

using namespace std;

void LayerLoader::saveDenseLayer(ofstream &modelFile, DenseLayer *denseLayer) {
    // saving shape
    for(int shapeElem : denseLayer->shape)
        modelFile << to_string(shapeElem) << " ";
    modelFile << endl;

    // saving next layer's shape
    for(int nextShapeElem : denseLayer->nextLayerShape)
        modelFile << to_string(nextShapeElem) << " ";
    modelFile << endl;

    modelFile << to_string(denseLayer->learningRate) << endl;

    TensorLoader::saveDenseLayer(modelFile, &(denseLayer->weightsTensor));
    TensorLoader::saveDenseLayer(modelFile, &(denseLayer->biasTensor));
}
