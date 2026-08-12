#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>

double relu(double x);

std::vector<double> softmax(
    const std::vector<double>& values
);

#endif