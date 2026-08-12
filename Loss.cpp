#include "Loss.h"
#include <cmath>


double crossEntropyLoss(
    const std::vector<double>& probabilities,
    int correctLabel
) {

    return -std::log(
        probabilities[correctLabel]
    );
}