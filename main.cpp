#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

#include "DataManager_lib/DataPrep.h"
#include "DataManager_lib/DataProvider.h"

//#include <SDL.h>
//#include <SDL_main.h>
//#include "SDL2.framework/Headers/SDL.h"
//#include "SDL2.framework/Headers/SDL_main.h"
#include "Simulation.h"

#include <random>

using namespace std;

int main() {
    auto[images, labels] = DataProvider::getMnistHRDData("/Users/tluszczyk/Desktop/mnist");

    for (auto &image : images) image = image.reshape({image.getShape()[0] * image.getShape()[1]});

    std::vector<Tensor> oneHotLabels;
    std::transform(
            labels.begin(),
            labels.end(),
            std::back_inserter(oneHotLabels),
            [](Tensor label) {
                std::vector<double> data(10, 0);
                data[label.getData()[0]] = 1;
                return Tensor({10}, data);
            }
    );

    Sequential *model = new Sequential({
                                               new DenseLayer({784}, "sig", "input layer"),
                                               new DenseLayer({16}, "sig", "first hidden layer"),
                                               new DenseLayer({16}, "sig", "second hidden layer"),
                                               new OutputLayer({10}, "sig", "output output"),
                                       }, "Mnist HRD Model");

    model->compile();

    std::vector<int> indices(images.size()), sampledIndices;
    std::iota(indices.begin(), indices.end(), 0);

    int batchSize = 10, epochSize = 10, testCases = 6000;
    int err = 0;
    for (int i = 0; i < testCases; i++) {
        std::vector<Tensor> batchImages, batchLabels;

//        std::sample(
//                indices.begin(), indices.end(),
//                std::back_inserter(sampledIndices),
//                batchSize, std::mt19937{std::random_device{}()}
//        );

//        for(int si : sampledIndices) {
//            batchImages.push_back(images[si]);
//            batchLabels.push_back(oneHotLabels[si]);
//        }

        for (int si = 0; si < batchSize; si++) {
            batchImages.push_back(images[(i * batchSize + si) % images.size()]);
            batchLabels.push_back(oneHotLabels[(i * batchSize + si) % images.size()]);
        }

        for (int epoch = 0; epoch < epochSize; epoch++)
            model->analyzeBatch(batchImages, batchLabels);

        std::vector<double> output = model->feed(batchImages[0]).getData();
        auto indexIt = std::max_element(output.begin(), output.end());

        if (std::distance(output.begin(), indexIt) != labels[(i * batchSize) % images.size()].getData()[0])
            err++;
        if (i % 10 == 1) std::cout << "accuracy: " << 100 * (double) (i + 1 - err) / i << "%" << endl;
    }

    ModelLoader::saveToFile(*model, "mnist_hrd_model.mdl");
    delete model;
}


//void runSimulationStep(){
//
//}

//int main(int argv, char** args) {
//
//
//    Simulation sim;
//
//    sim.init();
//    sim.run();
//
//    return 0;
//}
