//
// Created by Filip Tłuszcz on 25.03.2022.
//

#include "Tensor.h"

#include <functional>
#include <cfloat>
#include <string>
#include <map>
#include <random>
#include <numeric>

Tensor::Tensor() {
    this->shape = {0};
    this->data = {};
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<double>& data) {
    int dataSize = 1;
    for(int dim : shape) dataSize *= dim;

    this->shape = shape;
    this->data = data;
}

Tensor applySameSizeTensorOperator(const Tensor& a, const Tensor& b, const std::function<double(double, double)>& op) {
    if( a.getShape() != b.getShape() ) throw std::range_error("Tensors' sizes don't match");

    std::vector<double> resultData;
    resultData.reserve(a.getData().size());

    for(int i=0; i<a.getData().size(); i++)
        resultData.push_back(op(a.getData()[i],b.getData()[i]));

    return Tensor(a.getShape(), resultData);
}

Tensor Tensor::operator+(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, std::plus<>());
}

Tensor Tensor::operator-(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, std::minus<>());
}

Tensor Tensor::operator*(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, std::multiplies<>());
}

void multiplyMatrices(const Tensor& a, const Tensor& b, int startInd, std::vector<double> &resultData) {
    if( a.getShape().size() != 2 || b.getShape().size() != 2 ) throw std::domain_error("Matrices must be 2D");

    int resultMatrixWidth = b.getShape()[0];
    int resultMatrixHeight = a.getShape()[1];

    for(int w=0; w<resultMatrixWidth; w++) {
        for(int h=0; h<resultMatrixHeight; h++) {
            double val = 0;
            for(int it=0; it<a.getShape()[0]; it++) {
                double left = a.getData()[startInd + a.getShape()[0]*h + it];
                double upper = b.getData()[startInd + w + it*b.getShape()[0]];
                val += left * upper ;
            }

            resultData[startInd + w + h*resultMatrixWidth] = val;
        }
    }
}

Tensor Tensor::operator^(const Tensor& that) const {
    if( this->shape[0] != that.getShape()[1] ) throw std::range_error(
            "First tensor's first dimension must be equal to "
            "second tensor's second dimension to preform tensor multiplication on them"
    );

    for(int i=2; i<shape.size(); i++)
        if(shape[i] != that.getShape()[i])
            throw std::range_error("Both tensors's sizes above 0th and 1st must match");

    int dataSize = shape[1] * that.getShape()[0];
    for(int i=2; i<shape.size(); i++) dataSize *= shape[i];

    std::vector<double> resultData(dataSize, 0);

    int matrixSize = this->shape[1] * that.getShape()[0];

    for(int i=0; i<this->data.size(); i += matrixSize)
        multiplyMatrices(*this, that, i, resultData);


    std::vector<int> resultShape(this->shape);
    resultShape[0] = that.getShape()[0];

    return Tensor(resultShape, resultData);
}

bool Tensor::operator==(const Tensor &that) const {
    return this->shape == that.getShape() && this->data == that.getData();
}

double &Tensor::operator[](std::vector<int> coords) const {
    if( coords.size() != shape.size() ) throw std::range_error("coords don't match tensor's shape");
    int index = 0, currSize = 1;

    for(int i=0; i<shape.size(); i++) {
        if( coords[i] >= shape[i] ) throw std::range_error("tensor index out of range");
        index += coords[i] * currSize;
        currSize *= shape[i];
    }

    return (double&) data[index];
}

void iterateAndTransposeThroughTensor( const Tensor& a, const Tensor& b, const std::vector<int>& transposition, int dimIt=0, std::vector<int> *coord=nullptr){

    if( dimIt == a.getShape().size()-1 )
        coord = new std::vector<int>(a.getShape().size(), 0);

    if( dimIt == -1 ) {
        std::vector<int> newCoord;
        std::transform(transposition.begin(), transposition.end(), std::back_inserter(newCoord),
                       [&](int i) { return (*coord)[i]; });

        b[newCoord] = a[*coord];

    } else {
        for(int i=0; i<a.getShape()[dimIt]; i++) {
            (*coord)[dimIt] = i;
            iterateAndTransposeThroughTensor(a, b, transposition, dimIt-1, coord);
        }
    }
}

void transposeTensorWrapper( const Tensor& a, const Tensor& b, const std::vector<int>& transposition) {
    return iterateAndTransposeThroughTensor(a, b, transposition, a.getShape().size()-1);
}

Tensor Tensor::transpose(const std::vector<int>& transposition) const {
    std::vector<int> range(transposition.size());
    std::iota(range.begin(), range.end(), 0);

    if( !std::is_permutation(transposition.begin(), transposition.end(), range.begin()) )
        throw std::domain_error("given vector is not a transposition");

    Tensor transposedTensor = *this;
    if( transposition.size() > transposedTensor.getShape().size() ) {
        std::vector<int> transposedTensorShape = transposedTensor.getShape();
        transposedTensorShape.resize(transposition.size(), 1);
        transposedTensor = transposedTensor.reshape(transposedTensorShape);
    }

    std::vector<int> newShape;
    std::transform(transposition.begin(), transposition.end(), std::back_inserter(newShape), [&](int i){return transposedTensor.getShape()[i];});

    Tensor result(transposedTensor);
    result = result.reshape(newShape);

    transposeTensorWrapper(transposedTensor, result, transposition);

    return result;
}

Tensor Tensor::reshape(const std::vector<int>& newShape) const {
    int newShapeSize = 1;
    for(int newShapeDim : newShape) newShapeSize *= newShapeDim;

    int oldShapeSize = 1;
    for(int oldShapeDim : shape) oldShapeSize *= oldShapeDim;

    if( newShapeSize != oldShapeSize ) throw std::range_error("invalid shape");

    return Tensor(newShape, data);
}

std::string iterateAndStringifyThroughTensor( const Tensor& t, int dimIt, std::vector<int> *coord=nullptr ){
    std::string result;

    if( dimIt == t.getShape().size()-1 )
        coord = new std::vector<int>(t.getShape().size(), 0);

    if( dimIt == -1 ) {
        std::string number = std::to_string(t[*coord]);
        number.erase(number.find_last_not_of('0') + 1, std::string::npos);
        if( number.back() == '.' ) number.erase(number.size()-1);
        result += number;

    } else {
        result += "{";
        for(int i=0; i<t.getShape()[dimIt]; i++) {
            (*coord)[dimIt] = i;
            result += iterateAndStringifyThroughTensor(t, dimIt-1, coord);

            if( i != t.getShape()[dimIt]-1 )
                result += ",";
        }
        result += "}";
    }

    return result;
}

std::string stringifyTensorWrapper(const Tensor& t) {
    return iterateAndStringifyThroughTensor(t, t.getShape().size()-1);
}

std::string Tensor::to_string() const {
    return stringifyTensorWrapper(*this);
}

Tensor Tensor::createRandom(const std::vector<int> &shape) {
    std::random_device randomDevice;
    std::default_random_engine defaultRandomEngine(randomDevice());
    std::uniform_real_distribution<double> uniformRealDistribution(0,std::nextafter(1, DBL_MAX));

    int dataSize = 1;
    for(int dim : shape) dataSize *= dim;

    std::vector<double> data;
    std::generate_n(std::back_inserter(data), dataSize, [&](){return uniformRealDistribution(defaultRandomEngine);});

    return Tensor(shape, data);
}
