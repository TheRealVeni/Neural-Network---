#ifndef LAYER_H
#define LAYER_H

//what defines a layer? how do we propagate through it?

#include <vector>
#include "Neuron.h"

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int inputCount, int neuronCount);

    std::vector<double> forward(
        const std::vector<double>& inputs,
        bool useRelu
    );

    std::vector<double> backward(
        const std::vector<double>& gradients,
        bool useRelu
    );

    void update(double learningRate);
};

#endif
