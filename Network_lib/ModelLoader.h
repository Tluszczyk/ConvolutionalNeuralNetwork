//
// Created by Filip Tłuszcz on 01.04.2022.
//

#ifndef NEURALNET_MODELLOADER_H
#define NEURALNET_MODELLOADER_H

#include <string>
#include <Layers/Layer.h>


#include "Sequential.h"

using namespace std;

class Sequential;

class ModelLoader {
public:
    static void saveToFile(const Sequential& model, const string& filename);
    static Sequential* loadFromFile(const string& filename);
};


#endif //NEURALNET_MODELLOADER_H
