//
// Created by Filip Tłuszcz on 09.05.2022.
//

#include "DataProvider.h"

#include <random>
#include <fstream>

dataset DataProvider::getXorData() {
    std::vector<Tensor> inputs, outputs;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> binDist(0,1);

    for(int i=0; i<10000; i++) {
        double a = binDist(rng);
        double b = binDist(rng);

        double axb = (int)(a+b)%2;
        double anxb = (int)(axb+1)%2;

        inputs.push_back(Tensor({2}, {a,b}));
        outputs.push_back(Tensor({2}, {anxb, axb}));
    }
    return std::pair(inputs, outputs);
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<Tensor> loadMnistHRDImages(const string& imagePath) {
    std::vector<Tensor> images;

    ifstream mnistFile(imagePath, ios::binary);
    if(mnistFile.is_open()) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        mnistFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);

        mnistFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        mnistFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        mnistFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        for(int i=0;i<number_of_images;++i)
        {
            std::vector<double> data;
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    mnistFile.read((char*)&temp,sizeof(temp));
                    data.push_back((int)temp);
                }
            }
            images.push_back(Tensor{{n_cols, n_rows}, data});
        }
        return images;
    } else throw std::runtime_error("Cannot open mnist file");
}

std::vector<Tensor> loadMnistHRDLabels(const string& labelPath) {
    std::vector<Tensor> labels;

    ifstream mnistFile(labelPath, ios::binary);
    if(mnistFile.is_open()) {
        int magic_number=0;
        int number_of_labels=0;
        int n_rows=0;
        int n_cols=0;
        mnistFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);

        mnistFile.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= reverseInt(number_of_labels);

        for(int i=0;i<number_of_labels;++i)
        {
            unsigned char temp=0;
            mnistFile.read((char*)&temp,sizeof(temp));

            labels.push_back(Tensor{{1}, {(double)temp}});
        }
        return labels;
    } else throw std::runtime_error("Cannot open mnist file");
}

dataset DataProvider::getMnistHRDData(const string& mnistPath) {
    std::string imagePath   = mnistPath + "/train-images-idx3-ubyte";
    std::string labelPath = mnistPath + "/train-labels-idx1-ubyte";

    return std::pair(
            loadMnistHRDImages(imagePath),
            loadMnistHRDLabels(labelPath)
    );
}
