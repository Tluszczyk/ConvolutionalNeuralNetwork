//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "ModelLoader.h"
#include "Layers/LayerLoader.h"
#include "Layers/LayerType.h"
#include "Layers/DenseLayer.h"

#include <string>
#include <fstream>

using namespace std;

void ModelLoader::saveToFile(const Sequential& model, const string& filename) {
    ofstream modelFile("../" + filename);

    modelFile << model.MODEL_NAME << endl;
    modelFile << model.layers.size() << endl;

    for(Layer* layer : model.layers) {

        switch ( layer->GET_LAYER_TYPE() ) {
            case Dense:
                LayerLoader::saveDenseLayer(modelFile, (DenseLayer*) layer);
                break;

            case Output:
                LayerLoader::saveOutputLayer(modelFile, (OutputLayer*) layer);
                break;

            default:
                break;
        }
    }

    modelFile.close();
}

Sequential* ModelLoader::loadFromFile(const string& filename) {
    ifstream modelFile(filename);

    if (!modelFile.is_open()) throw std::runtime_error("Could not open the file");

    string modelName, layerCountS;

    getline(modelFile, modelName);
    getline(modelFile, layerCountS);

    int layerCount = std::stoi(layerCountS);

    auto *model = new Sequential({}, modelName);

    for (int layerIt = 0; layerIt < layerCount; layerIt++) {
        string layerType;
        getline(modelFile, layerType);

        switch (StringToLayerType(layerType)) {
            case Dense:
                model->addLayer(LayerLoader::loadDenseLayer(modelFile));
                break;

            case Output:
                model->addLayer(LayerLoader::loadOutputLayer(modelFile));
                break;

            default:
                break;
        }
    }

    modelFile.close();
    return model;
}