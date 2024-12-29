#include <iostream>
#include <vector>
#include "NN.h"

int main()
{
    // Initialize the neural network
    NN neural_network;

    // Add layers dynamically
    neural_network.add(new Linear(2, 3));
    neural_network.add(new Relu());
    neural_network.add(new Linear(3, 3));
    neural_network.add(new Relu());
    neural_network.add(new Linear(3, 1));
    neural_network.add(new Sigmoid());

    // Example input data
    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    // Train the network
    neural_network.fit(X, y, 10000, 0.01);

    // Test the network using XOR example
    std::vector<double> input = {0, 0};
    std::vector<double> output_prob = neural_network.predict(input);
    std::vector<double> output = {0};
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {0, 1};
    output_prob = neural_network.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {1, 0};
    output_prob = neural_network.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 1 << std::endl;
    std::cout << "----------------------" << std::endl;

    input = {1, 1};
    output_prob = neural_network.predict(input);
    if (output_prob[0] > 0.5)
    {
        output = {1};
    }
    else
    {
        output = {0};
    }
    std::cout << "Input: " << input[0] << ", " << input[1] << std::endl;
    std::cout << "Output Probability: " << output_prob[0] << std::endl;
    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Expected Output: " << 0 << std::endl;
    std::cout << "----------------------" << std::endl;

    return 0;
}

// Output example (100% accuracy)
/*  Input: 0, 0
    Output Probability: 4.68587e-5
    Output: 0
    Expected Output: 0
    ----------------------
    Input: 0, 1
    Output Probability: 0.998833
    Output: 1
    Expected Output: 1
    ----------------------
    Input: 1, 0
    Output Probability: 0.999439
    Output: 1
    Expected Output: 1
    ----------------------
    Input: 1, 1
    Output Probability: 0.00265182
    Output: 0
    Expected Output: 0
    ----------------------
*/