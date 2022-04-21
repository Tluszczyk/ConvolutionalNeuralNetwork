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
#include <utility>

Tensor::Tensor() {
    this->shape = {0};
    this->data = {};
}

Tensor::Tensor(const vector<int>& shape, const vector<double>& data) {
    int dataSize = 1;
    for(int dim : shape) dataSize *= dim;

    this->shape = shape;
    if (shape.size() < 2){
        this->shape.push_back(1);
    }
    if (data.empty()){
        this->data = vector<double>(dataSize, 0);
    } else {
        if (data.size() != dataSize){
            throw length_error("Length of provided data doesn't match provided shape.");
        }
        this->data = data;
    }
}

string vecToString(const vector<int>& vec) {
    string res = "{ ";
    for(int i=0; i<vec.size(); i++) res += to_string(vec[i]) + (i<vec.size()-1 ? ", " : "");
    return res + " }";
}

Tensor applySameSizeTensorOperator(const Tensor& a, const Tensor& b, const function<double(double, double)>& op) {
    if( a.getShape() != b.getShape() ) throw range_error("Tensors' shapes " + vecToString(a.getShape()) + ", " + vecToString(b.getShape()) + " don't match");

    vector<double> resultData;
    resultData.reserve(a.getData().size());

    for(int i=0; i<a.getData().size(); i++)
        resultData.push_back(op(a.getData()[i],b.getData()[i]));

    return Tensor(a.getShape(), resultData);
}

Tensor Tensor::operator+(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, plus<>());
}

Tensor Tensor::operator-(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, minus<>());
}

Tensor Tensor::operator*(const Tensor& that) const {
    return applySameSizeTensorOperator(*this, that, multiplies<>());
}

Tensor applyTensorNumberOperator(const Tensor& a, double k, const function<double(double, double)>& op) {
    vector<double> resultData;
    resultData.reserve(a.getData().size());

    for(double i : a.getData())
        resultData.push_back(op(i, k));

    return Tensor(a.getShape(), resultData);
}

Tensor Tensor::operator+(double that) const {
    return applyTensorNumberOperator(*this, that, plus<>());
}

Tensor Tensor::operator-(double that) const {
    return applyTensorNumberOperator(*this, that, minus<>());
}

Tensor Tensor::operator*(double that) const {
    return applyTensorNumberOperator(*this, that, multiplies<>());
}

void multiplyMatrices(const Tensor& a, const Tensor& b, int level, vector<double> &result) {
    if( a.getShape().size() != 2 || b.getShape().size() != 2 ) throw domain_error("Matrices must be 2D");

    int resultMatrixWidth = b.getShape()[0];
    int resultMatrixHeight = a.getShape()[1];

    int leftStartIndex = level * a.getShape()[0] * a.getShape()[1];
    int upperStartIndex = level * b.getShape()[0] * b.getShape()[1];
    int resultStartIndex = level * resultMatrixWidth * resultMatrixHeight;

    for(int j=0; j < resultMatrixWidth; j++) {
        for(int p=0; p < a.getShape()[0]; p++) {
            for(int i=0; i < resultMatrixHeight; i++) {
                double left = a.getData()[leftStartIndex + a.getShape()[0] * i + p];
                double upper = b.getData()[upperStartIndex + j + p * b.getShape()[0]];
                result[resultStartIndex + j + i * resultMatrixWidth] += left * upper;
            }
        }
    }
}


Tensor Tensor::operator^(const Tensor& that) const {
    if( this->shape[0] != that.getShape()[1] ) throw range_error(
                "First tensor's first dimension must be equal to "
                "second tensor's second dimension to preform tensor multiplication on them"
        );

    for(int i=2; i<shape.size(); i++)
        if(shape[i] != that.getShape()[i])
            throw range_error("Both tensors's sizes above 0th and 1st must match");

    int dataSize = shape[1] * that.getShape()[0];
    for(int i=2; i<shape.size(); i++) dataSize *= shape[i];

    vector<double> resultData(dataSize, 0);

    for(int i=0; i<dataSize/(shape[1] * that.getShape()[0]); i++)
        multiplyMatrices(*this, that, i, resultData);

    vector<int> resultShape(this->shape);
    resultShape[0] = that.getShape()[0];

    return Tensor(resultShape, resultData);
}

bool Tensor::operator==(const Tensor &that) const {
    return this->shape == that.getShape() && this->data == that.getData();
}

double &Tensor::operator[](vector<int> coords) const {
    if( coords.size() != shape.size() ) throw range_error("coords don't match tensor's shape");
    int index = 0, currSize = 1;

    for(int i=0; i<shape.size(); i++) {
        if( coords[i] >= shape[i] ) throw range_error("tensor index out of range");
        index += coords[i] * currSize;
        currSize *= shape[i];
    }

    return (double&) data[index];
}

void iterateAndTransposeThroughTensor( const Tensor& a, const Tensor& b, const vector<int>& transposition, int dimIt=0, vector<int> *coord=nullptr){

    if( dimIt == a.getShape().size()-1 )
        coord = new vector<int>(a.getShape().size(), 0);

    if( dimIt == -1 ) {
        vector<int> newCoord;
        transform(transposition.begin(), transposition.end(), back_inserter(newCoord),
                       [&](int i) { return (*coord)[i]; });

        b[newCoord] = a[*coord];

    } else {
        for(int i=0; i<a.getShape()[dimIt]; i++) {
            (*coord)[dimIt] = i;
            iterateAndTransposeThroughTensor(a, b, transposition, dimIt-1, coord);
        }
    }
}

void transposeTensorWrapper( const Tensor& a, const Tensor& b, const vector<int>& transposition) {
    return iterateAndTransposeThroughTensor(a, b, transposition, a.getShape().size()-1);
}

Tensor Tensor::transpose(const vector<int>& transposition) const {
    vector<int> range(transposition.size());
    iota(range.begin(), range.end(), 0);

    if( !is_permutation(transposition.begin(), transposition.end(), range.begin()) )
        throw domain_error("given vector is not a transposition");

    Tensor transposedTensor = *this;
    if( transposition.size() > transposedTensor.getShape().size() ) {
        vector<int> transposedTensorShape = transposedTensor.getShape();
        transposedTensorShape.resize(transposition.size(), 1);
        transposedTensor = transposedTensor.reshape(transposedTensorShape);
    }

    vector<int> newShape;
    transform(transposition.begin(), transposition.end(), back_inserter(newShape), [&](int i){return transposedTensor.getShape()[i];});

    Tensor result(transposedTensor);
    result = result.reshape(newShape);

    transposeTensorWrapper(transposedTensor, result, transposition);

    return result;
}

Tensor Tensor::reshape(vector<int> newShape) const {
    int newShapeSize = 1;
    for(int newShapeDim : newShape) newShapeSize *= newShapeDim;

    int oldShapeSize = 1;
    for(int oldShapeDim : shape) oldShapeSize *= oldShapeDim;

    if( newShapeSize != oldShapeSize ) throw range_error("Invalid shape, expected shape size " + std::to_string(oldShapeSize)
    + " got shape size " + std::to_string(newShapeSize));

    if (newShape.size() < 2){
        newShape.push_back(1);
    }

    return Tensor(newShape, data);
}

Tensor Tensor::copy() const {
    return Tensor(shape, data);
}

string iterateAndStringifyThroughTensor( const Tensor& t, int dimIt, vector<int> *coord=nullptr ){
    string result;

    if( dimIt == -1 ) {
        string number = to_string(t[*coord]);
        number.erase(number.find_last_not_of('0') + 1, string::npos);
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

string stringifyTensorWrapper(const Tensor& t) {
    return iterateAndStringifyThroughTensor(t, t.getShape().size()-1, new vector<int>(t.getShape().size(), 0));
}

string Tensor::to_string() const {
    return stringifyTensorWrapper(*this);
}

Tensor Tensor::createRandom(const vector<int> &shape) {
    random_device randomDevice;
    default_random_engine defaultRandomEngine(randomDevice());
    uniform_real_distribution<double> uniformRealDistribution(0,nextafter(1, DBL_MAX));

    int dataSize = 1;
    for(int dim : shape) dataSize *= dim;

    vector<double> data;
    generate_n(back_inserter(data), dataSize, [&](){return uniformRealDistribution(defaultRandomEngine);});

    return Tensor(shape, data);
}

Tensor Tensor::map(const function<double(double)> &op) {
    vector<double> resultData;
    transform(this->data.begin(), this->data.end(), back_inserter(resultData), op);
    return Tensor(this->shape, resultData);
}
