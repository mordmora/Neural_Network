#ifndef LOSSES_CPP
#define LOSSES_CPP

#include<vector>
#include<cmath>
#include<math.h>


double BCELoss(std::vector<double> true_label, std::vector<double> pred_prob){

/**
 * @brief Calculates the Binary Cross-Entropy Loss between true labels and predicted probabilities
 * 
 * The Binary Cross-Entropy Loss is defined as:
 * BCE = -1/N * Σ(y * log(p) + (1-y) * log(1-p))
 * where y is true label, p is predicted probability, and N is the number of samples
 * 
 * @param true_label Vector containing ground truth binary labels (0 or 1)
 * @param pred_prob Vector containing predicted probabilities in range [0,1]
 * @return double The calculated BCE loss value
 * 
 * @note Both input vectors must have the same size
 * @note Predicted probabilities should be in range (0,1) to avoid log(0)
 */

    double sum = 0;

    for(int i = 0; i < pred_prob.size(); i++){
        sum += true_label[i] * log(pred_prob[i]) + (1 - true_label[i]) * log(1 - pred_prob[i]);
    }
    int size = true_label.size();
    double loss = -(1.0/size) * sum;
    return loss;
}

std::vector<double> BCELossDerivative(std::vector<double> true_label, std::vector<double> pred_prob){
    std::vector<double> dev = {
        (pred_prob[0] - true_label[0]) / ((pred_prob[0] * (1 - pred_prob[0])))};

    return dev;
}

#endif