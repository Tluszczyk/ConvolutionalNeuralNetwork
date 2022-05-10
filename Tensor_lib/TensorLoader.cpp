//
// Created by Filip Tłuszcz on 01.04.2022.
//

#include "TensorLoader.h"

#include "../StringTools_lib/StringTools.h"

using namespace std;

void TensorLoader::saveTensor(ofstream &modelFile, Tensor *tensor) {
    for(int dim : tensor->shape) modelFile << to_string(dim) << " ";
    modelFile << endl;
    for(double elem : tensor->data) modelFile << to_string(elem) << " ";
    modelFile << endl;
}

Tensor TensorLoader::loadTensor(ifstream &modelFile) {
    string shapeS, dataS;

    getline(modelFile, shapeS);
    getline(modelFile, dataS);

    vector<string> shapesS = StringTools::split(StringTools::trim(shapeS), " ");
    vector<int> shape;

    vector<string> datasS = StringTools::split(StringTools::trim(dataS), " ");
    vector<double> data;

    transform(shapesS.begin(), shapesS.end(), std::back_inserter(shape), [](string s){return std::stoi(s);});
    transform(datasS.begin(), datasS.end(), std::back_inserter(data), [](string s){return std::stod(s);});

    return Tensor(shape, data);
}
