#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

#include "DataManager_lib/DataPrep.h"
#include "DataManager_lib/DataProvider.h"

#include <random>

using namespace std;

//int main() {
//    auto [images, labels] = DataProvider::getMnistHRDData("/Users/tluszczyk/Desktop/mnist");
//
//    Sequential *model = ModelLoader::loadFromFile("/Users/tluszczyk/dev/AGH/VI sem/ZSC/NeuralNet/mnist_hrd_model.mdl");
//
//    auto indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
//
//    for(auto i : indices) {
//        Tensor image = images[i];
//        Tensor label = labels[i];
//
//        cout << "test " << i << endl;
//
//        for(int y=0; y<image.getShape()[1]; y++) {
//            for(int x=0; x<image.getShape()[0]; x++) {
//                if( image[{x,y}] < 50 )         cout << " ";
//                else if( image[{x,y}] < 100 )   cout << ".";
//                else if( image[{x,y}] < 150 )   cout << "+";
//                else if( image[{x,y}] < 200 )   cout << "*";
//                else                            cout << "#";
//            }
//            cout << endl;
//        }
//        cout << "label: " << label[{0}] << endl;
//
//        vector<double> output = model->feed(image.reshape({image.getShape()[0]*image.getShape()[1]})).getData();
//
//        auto indexIt = std::max_element(output.begin(), output.end());
//
//        cout << "prediction: " << distance(output.begin(), indexIt) << endl;
//    }
//
//    delete model;
//
//    return 0;
//}

int main() {
    cout << "Loading MNiST data..." << endl;
    auto [images, labels] = DataProvider::getMnistHRDData("/Users/tluszczyk/Desktop/mnist");
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

    Sequential *model = new Sequential({
        new DenseLayer({784}, "sig", "input layer"),
//        new DenseLayer({64}, "sig", "first hidden"),
//        new DenseLayer({16}, "sig", "second hidden"),
        new OutputLayer({10}, "sig", "output output"),
    }, "Mnist HRD Model");

    model->compile(2147483645);

    std::vector<int> indices(images.size()), sampledIndices;
    std::iota(indices.begin(), indices.end(), 0);

    int batchSize = 1, epochSize=10, testCases = 6000;
    int err=0;
    for(int i=0; i<testCases; i++) {
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

        cout << "new epoch:" << endl;
        for(int epoch=0; epoch<epochSize; epoch++)
            cout << "loss is " << model->analyzeBatch(batchImages, batchLabels) << endl;

        std::vector<double> output = model->feed(batchImages[0]).getData();
        auto indexIt = std::max_element(output.begin(), output.end());

        if(std::distance(output.begin(), indexIt) != labels[(i*batchSize)%images.size()].getData()[0])
            err++;

        printf("label: %d, predicted: %d, ", (int)labels[(i*batchSize)%images.size()].getData()[0], std::distance(output.begin(), indexIt));

        printf("accuracy: %5f%%\n", 100.*(double)(i+1-err)/(i+1));
    }

    ModelLoader::saveToFile(*model, "mnist_hrd_model.mdl");
    delete model;

    return 0;
}
