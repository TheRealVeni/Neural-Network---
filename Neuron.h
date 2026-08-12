#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    std::vector<double> weights;
    std::vector<double> weightGradients;

    double bias;
    double biasGradient;

    std::vector<double> lastInputs;
    double lastOutput;

    Neuron(int inputCount);

    double forward(const std::vector<double>& inputs);

    std::vector<double> backward(double gradient);

    void update(double learningRate);
};

#endif