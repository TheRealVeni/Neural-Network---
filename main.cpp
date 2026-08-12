#include <iostream>
#include <vector>
#include "Layer.h"
#include "Activations.h"
#include "Loss.h"
#include "MNIST.h"
//why not use namespace std; idk tbh, probably look at the constructor for the neurons

int main() {

    // ============================================
    // Load MNIST
    // ============================================

    std::cout << "Loading MNIST...\n";

    std::vector<std::vector<double>> images =
        loadImages(
            "data/train-images-idx3-ubyte"
        );

    std::vector<int> labels =
        loadLabels(
            "data/train-labels-idx1-ubyte"
        );

    std::cout << "Loaded "
              << images.size()
              << " images.\n";


    // ============================================
    // Create network
    // ============================================

    Layer layer1(784, 128);
    Layer layer2(128, 64);
    Layer layer3(64, 10);


    // ============================================
    // Training settings
    // ============================================

    double learningRate = 0.01;

    int epochs = 5;


    // ============================================
    // Training
    // ============================================

    for (int epoch = 0;
         epoch < epochs;
         epoch++) {

        double totalLoss = 0.0;

        int correct = 0;


        for (int i = 0;
             i < images.size();
             i++) {

            // ------------------------------------
            // Forward
            // ------------------------------------

            std::vector<double> output1 =
                layer1.forward(
                    images[i],
                    true
                );

            std::vector<double> output2 =
                layer2.forward(
                    output1,
                    true
                );

            std::vector<double> logits =
                layer3.forward(
                    output2,
                    false
                );


            // ------------------------------------
            // Softmax
            // ------------------------------------

            std::vector<double> probabilities =
                softmax(logits);


            // ------------------------------------
            // Loss
            // ------------------------------------

            double loss =
                crossEntropyLoss(
                    probabilities,
                    labels[i]
                );

            totalLoss += loss;


            // ------------------------------------
            // Prediction
            // ------------------------------------

            int prediction = 0;

            for (int j = 1; j < 10; j++) {

                if (probabilities[j] >
                    probabilities[prediction]) {

                    prediction = j;
                }
            }

            if (prediction == labels[i]) {
                correct++;
            }


            // ------------------------------------
            // Backpropagation
            // ------------------------------------

            std::vector<double> outputGradients =
                probabilities;

            outputGradients[labels[i]] -= 1.0;


            std::vector<double> hidden2Gradients =
                layer3.backward(
                    outputGradients,
                    false
                );


            std::vector<double> hidden1Gradients =
                layer2.backward(
                    hidden2Gradients,
                    true
                );


            layer1.backward(
                hidden1Gradients,
                true
            );


            // ------------------------------------
            // Gradient descent
            // ------------------------------------

            layer1.update(learningRate);
            layer2.update(learningRate);
            layer3.update(learningRate);


            // ------------------------------------
            // Progress
            // ------------------------------------

            if ((i + 1) % 1000 == 0) {

                std::cout
                    << "Epoch "
                    << epoch + 1
                    << " | Image "
                    << i + 1
                    << " / "
                    << images.size()
                    << '\n';
            }
        }


        // ========================================
        // Epoch statistics
        // ========================================

        double averageLoss =
            totalLoss / images.size();

        double accuracy =
            static_cast<double>(correct)
            / images.size();


        std::cout << "\n";
        std::cout << "Epoch "
                  << epoch + 1
                  << " finished\n";

        std::cout << "Loss: "
                  << averageLoss
                  << '\n';

        std::cout << "Accuracy: "
                  << accuracy * 100
                  << "%\n";

        std::cout << "-------------------------\n";
    }


    return 0;
}
