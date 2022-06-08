//
// Created by Filip Tłuszcz on 06.06.2022.
//

#include "PoolingFunctionProvider.h"

#include <numeric>

double PoolingFunctionProvider::max(const Tensor &in) { return *std::max_element(in.getData().begin(), in.getData().end()); }

double PoolingFunctionProvider::avg(const Tensor &in) { return std::accumulate(in.getData().begin(), in.getData().end(), 0.) / in.getData().size(); }

double PoolingFunctionProvider::min(const Tensor &in) { return *std::min_element(in.getData().begin(), in.getData().end()); }

map<string, function<double(const Tensor&)>> PoolingFunctionProvider::fromName = {
        {"max", PoolingFunctionProvider::max},
        {"avg", PoolingFunctionProvider::avg},
        {"min", PoolingFunctionProvider::min},
};