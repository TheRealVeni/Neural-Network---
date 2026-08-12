#ifndef LOSS_H
#define LOSS_H

#include <vector>

double crossEntropyLoss(
    const std::vector<double>& probabilities,
    int correctLabel
);

#endif