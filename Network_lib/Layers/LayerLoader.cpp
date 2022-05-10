//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "LayerLoader.h"
#include "TensorLoader.h"

#include "../StringTools_lib/StringTools.h"

using namespace std;

void LayerLoader::saveLayer(ofstream &modelFile, Layer *layer) {
    // saving layer's type
    modelFile << LayerTypeToString(layer->GET_LAYER_TYPE()) << endl;

    // saving layer's name
    modelFile << layer->getName() << endl;

    // saving activation function
    modelFile << layer->getActivationFunctionName() << endl;

    // saving shape
    for(int shapeElem : layer->shape)
        modelFile << to_string(shapeElem) << " ";
    modelFile << endl;
}

void LayerLoader::saveDenseLayer(ofstream &modelFile, DenseLayer *layer) {
    LayerLoader::saveLayer(modelFile, layer);

    // saving next layer's shape
    for(int nextShapeElem : layer->nextLayerShape)
        modelFile << to_string(nextShapeElem) << " ";
    modelFile << endl;

    // saving learning rate
    modelFile << to_string(layer->learningRate) << endl;

    TensorLoader::saveTensor(modelFile, &(layer->weightsTensor));
    TensorLoader::saveTensor(modelFile, &(layer->biasTensor));
}

void LayerLoader::saveOutputLayer(ofstream &modelFile, OutputLayer *layer) {
    LayerLoader::saveLayer(modelFile, layer);
}

DenseLayer* LayerLoader::loadDenseLayer(ifstream& modelFile) {
    string shapeS, nextLayerShapeS, activationFunctionName, layerName, learningRateS, weightsTensorS, biasTensorS;

    getline(modelFile, layerName);
    getline(modelFile, activationFunctionName);
    getline(modelFile, shapeS);
    getline(modelFile, nextLayerShapeS);
    getline(modelFile, learningRateS);

    Tensor weightsTensor = TensorLoader::loadTensor(modelFile);
    Tensor biasTensor = TensorLoader::loadTensor(modelFile);

    vector<int> shape;
    vector<string> shapesS = StringTools::split(StringTools::trim(shapeS), " ");
    std::transform(
            shapesS.begin(),
            shapesS.end(),
            std::back_inserter(shape),
            [](string s){return std::stoi(s);}
    );

    double learningRate = stod(learningRateS);

    vector<int> nextLayerShape;
    vector<string> nextLayerShapesS = StringTools::split(StringTools::trim(nextLayerShapeS), " ");
    std::transform(
            nextLayerShapesS.begin(),
            nextLayerShapesS.end(),
            std::back_inserter(nextLayerShape),
            [](string s){return std::stoi(s);}
    );

    auto *result = new DenseLayer(shape, activationFunctionName, layerName);
    result->compile(learningRate, nextLayerShape);

    result->changeWeightsTensor(weightsTensor);
    result->changeBiasTensor(biasTensor);

    return result;
}

OutputLayer* LayerLoader::loadOutputLayer(ifstream& modelFile) {
    string shapeS, nextLayerShapeS, activationFunctionName, layerName, learningRateS, weightsTensorS, biasTensorS;

    getline(modelFile, layerName);
    getline(modelFile, activationFunctionName);
    getline(modelFile, shapeS);

    vector<int> shape;
    size_t pos = 0;
    while((pos = shapeS.find(' ')) != string::npos) {
        shape.push_back(stoi(shapeS.substr(0, pos)));
        shapeS.erase(0, pos+1);
    }

    auto *result = new OutputLayer(shape, activationFunctionName, layerName);

    return result;
}