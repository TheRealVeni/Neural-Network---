#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <string>


std::vector<std::vector<double>> loadImages(
    const std::string& filename
);


std::vector<int> loadLabels(
    const std::string& filename
);

#endif