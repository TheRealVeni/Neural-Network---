#include "MNIST.h"
#include <fstream>
#include <cstdint>
#include <stdexcept>


uint32_t readInt(std::ifstream& file) {

    uint32_t value;

    file.read(
        reinterpret_cast<char*>(&value),
        sizeof(value)
    );

    return
        ((value & 0x000000FF) << 24) |
        ((value & 0x0000FF00) << 8)  |
        ((value & 0x00FF0000) >> 8)  |
        ((value & 0xFF000000) >> 24);
}


std::vector<std::vector<double>> loadImages(
    const std::string& filename
) {

    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open image file: " + filename
        );
    }

    uint32_t magic = readInt(file);
    uint32_t count = readInt(file);
    uint32_t rows  = readInt(file);
    uint32_t cols  = readInt(file);

    if (magic != 2051) {
        throw std::runtime_error(
            "Invalid MNIST image file."
        );
    }

    std::vector<std::vector<double>> images(
        count,
        std::vector<double>(rows * cols)
    );

    for (uint32_t i = 0; i < count; i++) {

        for (uint32_t j = 0;
             j < rows * cols;
             j++) {

            unsigned char pixel;

            file.read(
                reinterpret_cast<char*>(&pixel),
                1
            );

            images[i][j] =
                pixel / 255.0;
        }
    }

    return images;
}


std::vector<int> loadLabels(
    const std::string& filename
) {

    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open label file: " + filename
        );
    }

    uint32_t magic = readInt(file);
    uint32_t count = readInt(file);

    if (magic != 2049) {
        throw std::runtime_error(
            "Invalid MNIST label file."
        );
    }

    std::vector<int> labels(count);

    for (uint32_t i = 0; i < count; i++) {

        unsigned char label;

        file.read(
            reinterpret_cast<char*>(&label),
            1
        );

        labels[i] = label;
    }

    return labels;
}