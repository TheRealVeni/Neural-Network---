#include "Layer.h"
#include "Activations.h"

Layer::Layer(
    int inputCount,
    int neuronCount
) {

    for (int i = 0; i < neuronCount; i++) {
        neurons.emplace_back(inputCount);
    }
}


std::vector<double> Layer::forward(
    const std::vector<double>& inputs,
    bool useRelu
) {

    std::vector<double> outputs;

    for (Neuron& neuron : neurons) {

        double value =
            neuron.forward(inputs);

        if (useRelu) {
            value = relu(value);
        }

        outputs.push_back(value);
    }

    return outputs;
}


std::vector<double> Layer::backward(
    const std::vector<double>& gradients,
    bool useRelu
) {

    std::vector<double> inputGradients(
        neurons[0].weights.size(),
        0.0
    );

    for (int i = 0; i < neurons.size(); i++) {

        double gradient = gradients[i];

        if (useRelu &&
            neurons[i].lastOutput <= 0) {

            gradient = 0;
        }

        std::vector<double> neuronGradients =
            neurons[i].backward(gradient);

        for (int j = 0;
             j < inputGradients.size();
             j++) {

            inputGradients[j] +=
                neuronGradients[j];
        }
    }

    return inputGradients;
}


void Layer::update(double learningRate) {

    for (Neuron& neuron : neurons) {
        neuron.update(learningRate);
    }
}