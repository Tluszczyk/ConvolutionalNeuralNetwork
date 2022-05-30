//
// Created by Filip Tłuszcz on 25.03.2022.
//

#ifndef NEURALNET_TENSOR_H
#define NEURALNET_TENSOR_H

#include "TensorLoader.h"

#include <vector>
#include <ostream>
#include <functional>

class Tensor {
    friend class TensorLoader;

private:
    std::vector<int> shape;
    std::vector<double> data;

    friend std::ostream& operator<<(std::ostream &strm, const Tensor &t) { return strm << t.to_string(); }

public:
    Tensor();
    explicit Tensor(const std::vector<int>& shape, const std::vector<double>& data = {});

    Tensor operator+(const Tensor& that) const;
    Tensor operator-(const Tensor& that) const;
    Tensor operator*(const Tensor& that) const;
    Tensor operator^(const Tensor& that) const;

    Tensor operator+(double that) const;
    Tensor operator-(double that) const;
    Tensor operator*(double that) const;

    Tensor convolve(const Tensor& filter);

    double &operator[](std::vector<int> coords) const;

    bool operator==(const Tensor& that) const;

    static Tensor createRandom(const std::vector<int>& shape, double variance);
    static Tensor ZERO() { return Tensor{{1},{0}}; };

    Tensor map(const std::function<double(double)> &op);

    double max_abs();

    [[nodiscard]] Tensor transpose(const std::vector<int>& transposition) const;
    [[nodiscard]] Tensor reshape(std::vector<int> newShape) const;
    [[nodiscard]] Tensor copy() const;


    [[nodiscard]] std::vector<int> getShape() const { return shape; };
    [[nodiscard]] const std::vector<double> &getData() const { return data; };

    [[nodiscard]] std::string to_string() const;


};


#endif //NEURALNET_TENSOR_H
