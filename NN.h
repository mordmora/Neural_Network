#ifndef NN_CPP
#define NN_CPP

#include<vector>
#include<memory>
#include<iostream>
#include "layer.h"
#include "losses.h"

class NN {
    public:
    std::vector<std::unique_ptr<Layer>> layers;

    void add(Layer* layer){
        layers.emplace_back(layer);
    }

    std::vector<double> forward_propagation(std::vector<double> input){
        std::vector<double> data = input;
        for(const auto& layer : layers){
            data = layer->forward(data);
        }
        return data;
    }

    std::vector<double> predict (std::vector<double> input){
        return forward_propagation(input);
    }

    void back_propagation(std::vector<double> error, double learning_rate){
        std::vector<double> data = error;
        for(auto it = layers.rbegin(); it != layers.rend(); ++it){
            data = (*it)->backward(data, learning_rate);
        }
    }

    void fit(const std::vector<std::vector<double>>&X, std::vector<std::vector<double>>&Y, int epochs, double learning_rate){
        for(int epoch = 0; epoch < epochs; epoch++){
            double total_loss = 0;
            for(size_t i = 0; i < X.size(); i++){
                std::vector<double> out = forward_propagation(X[i]);
                double loss = BCELoss(Y[i], out);
                total_loss += loss;
                std::vector<double> loss_derivative = BCELossDerivative(Y[i], out);
                back_propagation(loss_derivative, learning_rate);
            }
            std::cout << "Epoch: " << epoch << " Loss: " << total_loss << std::endl;
        }
    }
};
#endif