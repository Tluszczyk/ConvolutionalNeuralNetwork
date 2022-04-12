//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "ModelLoader.h"
#include "LayerLoader.h"
#include "LayerType.h"
#include "DenseLayer.h"

#include <string>
#include <fstream>

using namespace std;

//void ModelLoader::saveToFile(const Sequential& model, const string& filename) {
//    ofstream modelFile("../" + filename);
//
//    modelFile << model.MODEL_NAME << endl;
//
//    for(Layer* layer : model.layers) {
//        modelFile << LayerTypeToString(layer->GET_LAYER_TYPE()) << endl;
//        modelFile << layer->layerName << endl;
//
//        switch ( layer->GET_LAYER_TYPE() ) {
//            case Dense:
//                LayerLoader::saveDenseLayer(modelFile, (DenseLayer*) layer);
//        }
//    }
//
//    modelFile.close();
//}

//Sequential ModelLoader::loadFromFile(const string& filename) {
//
//}