#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

#include "DataManager_lib/DataPrep.h"
#include "DataManager_lib/DataProvider.h"

//#include <windows.h>
#include <random>
#include <filesystem>
#include <thread>

bool running = true;

void input_thread(){
    cout << "Enter any key to end training and save model to file." << endl;
    char c;
    scanf("%c", &c);
    running = false;
    cout << "Quiting learning process, wait for current epoch to finish."<< endl;
}


int main() {
    std::cout << "Enter F if you want to read from file, or N if you want to create a new model." << endl;
    char c;
    scanf("%c", &c);

    Sequential *model;

    if (c == 'F'){
        model = ModelLoader::loadFromFile("mnist_hrd_model_1.mdl");
    } else if (c == 'N'){
        model = new Sequential({
                                                   new DenseLayer({784}, "id", "input layer"),
                                                   new DenseLayer({400}, "relu", "first hidden"),
                                                   new DenseLayer({100}, "relu", "second hidden"),
                                                   new OutputLayer({10}, "softmax", "output output"),
                                           }, "Mnist HRD Model");

        model->compile(0.1);
    }

    scanf("%c", &c);

    std::cout << "Loading MNiST data..." << endl;
    auto [images, labels] = DataProvider::getMnistHRDData("../data");
    cout << "Loaded MNiST data" << endl;

    for(auto &image : images) image = image.reshape({image.getShape()[0]*image.getShape()[1]});
    for(auto &image : images) image = image*(1./255.);

    std::vector<Tensor> oneHotLabels;
    std::transform(
            labels.begin(),
            labels.end(),
            std::back_inserter(oneHotLabels),
            [](Tensor label){
                std::vector<double> data(10, 0);
                data[label.getData()[0]] = 1;
                return Tensor({10}, data);
            }
    );

    std::vector<int> indices(images.size()), sampledIndices;
    std::iota(indices.begin(), indices.end(), 0);

    int batchSize = 5, epochSize=10, testCases = 6000;
    int err=0;

    std::thread quitter_thread(input_thread);


    for(int i=0; i<testCases && running; i++) {
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

        for(int si=0; si<batchSize; si++) {
            int ind = (i*batchSize+si)%images.size();
            batchImages.push_back(images[ind]);
            batchLabels.push_back(oneHotLabels[ind]);
        }

        cout << "new epoch: " << i << endl;
        double loss;
        for(int epoch=0; epoch<epochSize; epoch++) {
            loss = model->analyzeBatch(batchImages, batchLabels);
            cout << "loss is " << loss << endl;
        }

        std::vector<double> output = model->feed(batchImages[0]).getData();
        auto indexIt = std::max_element(output.begin(), output.end());

        if(std::distance(output.begin(), indexIt) != labels[(i*batchSize)%images.size()].getData()[0])
            err++;

        printf("label: %d, predicted: %d, ", (int)labels[(i*batchSize)%images.size()].getData()[0], std::distance(output.begin(), indexIt));

        printf("accuracy: %5f%%\n", 100.*(double)(i+1-err)/(i+1));
    }

    quitter_thread.join();

    ModelLoader::saveToFile(*model, "mnist_hrd_model_2.mdl");
    cout << "Saved model to file." << endl;
    delete model;

    return 0;
}
