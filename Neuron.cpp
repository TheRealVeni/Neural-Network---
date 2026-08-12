#include "Neuron.h"
#include <random>

Neuron::Neuron(int inputCount) {

    weights.resize(inputCount);
    weightGradients.resize(inputCount);

    static std::random_device rd;
    static std::mt19937 generator(rd());

    std::uniform_real_distribution<double> distribution(
        -0.1,
        0.1
    );

    for (int i = 0; i < inputCount; i++) {
        weights[i] = distribution(generator);
        weightGradients[i] = 0.0;
    }

    bias = 0.0;
    biasGradient = 0.0;
    lastOutput = 0.0;
}


double Neuron::forward(
    const std::vector<double>& inputs
) {

    lastInputs = inputs;

    double sum = bias;

    for (int i = 0; i < inputs.size(); i++) {
        sum += inputs[i] * weights[i];
    }

    lastOutput = sum;

    return sum;
}


std::vector<double> Neuron::backward(
    double gradient
) {

    std::vector<double> inputGradients(
        weights.size(),
        0.0
    );

    for (int i = 0; i < weights.size(); i++) {

        weightGradients[i] =
            gradient * lastInputs[i];

        inputGradients[i] =
            gradient * weights[i];
    }

    biasGradient = gradient;

    return inputGradients;
}


void Neuron::update(double learningRate) {

    for (int i = 0; i < weights.size(); i++) {

        weights[i] -=
            learningRate * weightGradients[i];
    }

    bias -=
        learningRate * biasGradient;
}