#include "Activations.h"
#include <cmath>
#include <algorithm>


double relu(double x) {

    return std::max(0.0, x);
}


std::vector<double> softmax(
    const std::vector<double>& values
) {

    double maxValue = values[0];

    for (double value : values) {
        maxValue =
            std::max(maxValue, value);
    }

    std::vector<double> probabilities;

    double sum = 0.0;

    for (double value : values) {

        double e =
            std::exp(value - maxValue);

        probabilities.push_back(e);

        sum += e;
    }

    for (double& probability : probabilities) {
        probability /= sum;
    }

    return probabilities;
}

//dont forget the softmax function in neuron
