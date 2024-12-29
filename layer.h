#ifndef LAYER_CPP
#define LAYER_CPP

#include<vector>
#include "utils.h"
#include "activation.h"

class Layer{
    public:
        std::vector<double> input;
        std::vector<double> output;
        virtual std::vector<double> forward(const std::vector<double> input_data) = 0;
        virtual std::vector<double> backward(std::vector<double> error, double learning_rate) = 0;
};

class Sigmoid : public Layer {
    public:
        std::vector<double> forward(const std::vector<double> input_data) override {
            input = input_data;
            output = vectSigmoid(input_data);
            return output;
        }

        std::vector<double> backward(std::vector<double> error, double learning_rate) override {
            std::vector<double> derivative = vectSigmoidDerivative(input);
            std::vector<double> grad_input;
            for(int i = 0; i < derivative.size(); i++){
                grad_input.push_back(error[i] * derivative[i]);
            }
            return grad_input;
        }
};

class Relu : public Layer {
    public:
        std::vector<double> forward(const std::vector<double> input_data) override {
            input = input_data;
            output = vectRelu(input_data);
            return output;
        }

        std::vector<double> backward(std::vector<double> error, double learning_rate) override {
            std::vector<double> derivative = vectReluDerivative(input);
            std::vector<double> grad_input;
            for(int i = 0; i < derivative.size(); i++){
                grad_input.push_back(error[i] * derivative[i]);
            }
            return grad_input;
        }
};

class LeakyRelu : public Layer {
    public:
        double alpha = 0.01;

        std::vector<double> forward(const std::vector<double> input_data) override {
            input = input_data;
            output = vectLeakyRelu(input_data, alpha);
            return output;
        }

        std::vector<double> backward(std::vector<double> error, double learning_rate) override {
            std::vector<double> derivative = vectLeakyReluDerivative(input, alpha);
            std::vector<double> grad_input;
            for(int i = 0; i < derivative.size(); i++){
                grad_input.push_back(error[i] * derivative[i]);
            }
            return grad_input;
        }
};

class Tanh : public Layer {
    public:
        std::vector<double> forward(const std::vector<double> input_data) override {
            input = input_data;
            output = vectTanh(input_data);
            return output;
        }

        std::vector<double> backward(std::vector<double> error, double learning_rate) override {
            std::vector<double> derivative = vectTanhDerivative(input);
            std::vector<double> grad_input;
            for(int i = 0; i < derivative.size(); i++){
                grad_input.push_back(error[i] * derivative[i]);
            }
            return grad_input;
        }
};


class Linear : public Layer {
    public:
        int input_neurons;
        int output_neurons;
        std::vector<std::vector<double>> weights;
        std::vector<double> bias;

        Linear(int input_neurons, int output_neurons){
            this->input_neurons = input_neurons;
            this->output_neurons = output_neurons;
            weights = uniformWeightInitializer(output_neurons, input_neurons);
            bias = biasInit(output_neurons);
        }

        std::vector<double> forward(const std::vector<double> intput_data) override {
            input = intput_data;
            output.clear();
            for(int i = 0; i < output_neurons; i++){
                output.push_back(dotProduct(weights[i], input) + bias[i]);
            }

            return output;
        }

        std::vector<double> backward(std::vector<double> error, double learning_rate) override {
            std::vector<double> input_error;                    //dE/dX
            std::vector<std::vector<double>> weight_error;      //dE/dW
            std::vector<double> bias_error;                     //dE/dB
            std::vector<std::vector<double>> weight_transpose;
            input_error.clear();
            weight_error.clear();
            bias_error.clear();
            weight_transpose.clear();

            weight_transpose = transpose(weights);
            bias_error = error;

            for(int i = 0; i < weight_transpose.size(); i++){
                input_error.push_back(dotProduct(weight_transpose[i], error));
            }
            for(int i = 0; i < error.size(); i++){
                std::vector<double> row;
                for(int j = 0; j < input.size(); j++){
                    row.push_back(error[i] * input[j]);
                }
                weight_error.push_back(row);
            }

            std::vector<double> delta_bias = scalarVectorMultiplication(bias_error, learning_rate);
            bias = subtract(bias, delta_bias);

            for(int i = 0; i < weight_error.size(); i++){
                std::vector<double> delta_weight = scalarVectorMultiplication(weight_error[i], learning_rate);
                weights[i] = subtract(weights[i], delta_weight);
            }

            return input_error;
        }



};

#endif