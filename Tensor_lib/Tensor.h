//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_TENSOR_H
#define NEURALNET_TENSOR_H

#include "TensorLoader.h"

#include <vector>
#include <ostream>
#include <functional>

using namespace std;

class Tensor {
    friend class TensorLoader;

private:
    vector<int> shape;
    vector<double> data;

    friend ostream& operator<<(ostream &strm, const Tensor &t) { return strm << t.to_string(); }

public:
    Tensor();
    explicit Tensor(const vector<int>& shape, const vector<double>& data = {});

    Tensor operator+(const Tensor& that) const;
    Tensor operator-(const Tensor& that) const;
    Tensor operator*(const Tensor& that) const;
    Tensor operator^(const Tensor& that) const;

    Tensor operator+(double that) const;
    Tensor operator-(double that) const;
    Tensor operator*(double that) const;

    double &operator[](vector<int> coords) const;

    bool operator==(const Tensor& that) const;

    static Tensor createRandom(const vector<int>& shape);
    static Tensor ZERO() { return Tensor{{1},{0}}; };

    Tensor map(const function<double(double)> &op);

    [[nodiscard]] Tensor transpose(const vector<int>& transposition) const;
    [[nodiscard]] Tensor reshape(vector<int> newShape) const;
    [[nodiscard]] Tensor copy() const;


    [[nodiscard]] vector<int> getShape() const { return shape; };
    [[nodiscard]] const vector<double> &getData() const { return data; };

    [[nodiscard]] string to_string() const;


};


#endif //NEURALNET_TENSOR_H
