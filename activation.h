#ifndef ACTIVATION_CPP
#define ACTIVATION_CPP


#include<cmath>
#include<vector>

    double sigmoid(double x) {

        /**
         * @brief Calculates the sigmoid (logistic) activation function
         *
         * The sigmoid function is defined as f(x) = 1/(1 + e^(-x))
         * This function squashes the input into a range between 0 and 1,
         * making it useful for binary classification and neural network outputs.
         *
         * @param x The input value to the sigmoid function
         * @return double The sigmoid of x, which will be between 0 and 1
         */

        return 1 / (1 + exp(-x));
    }

    double sigmoidDerivative(double x) {

        /**
         * @brief Calculates the derivative of the sigmoid function
         *
         * The derivative of the sigmoid function is f'(x) = f(x) * (1 - f(x))
         * This function is used in backpropagation to calculate gradients
         * and update weights in neural networks.
         *
         * @param x The input value to the sigmoid function
         * @return double The derivative of the sigmoid function at x
         */

        return sigmoid(x) * (1 - sigmoid(x));
    }

    std::vector<double> vectSigmoid(const std::vector<double> x)
    {
        /**
         * A vectorized version of the sigmoid function.
         * @param x the input vector
         * @return a vector where each element is the sigmoid of the corresponding element in x
         */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(sigmoid(i));
        return result;
    }

    std::vector<double> vectSigmoidDerivative(const std::vector<double> x)
    {
        /**
         * A vectorized version of the derivative of the sigmoid function.
         * @param x the input vector
         * @return a vector where each element is the derivative of the sigmoid of the corresponding element in x
         */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(sigmoidDerivative(i));
        return result;
    }


    double relu(double x) {

        /**
         * @brief Calculates the ReLU (Rectified Linear Unit) activation function
         *
         * The ReLU function is defined as f(x) = max(0, x)
         * This function returns x if x is positive, and 0 otherwise.
         * ReLU is a popular activation function for deep neural networks.
         *
         * @param x The input value to the ReLU function
         * @return double The ReLU of x, which will be x if x > 0, or 0 otherwise
         */

        return x > 0 ? x : 0;
    }

    double reluDerivative(double x) {

        /**
         * @brief Calculates the derivative of the ReLU function
         *
         * The derivative of the ReLU function is f'(x) = 1 if x > 0, or 0 otherwise
         * This function is used in backpropagation to calculate gradients
         * and update weights in neural networks.
         *
         * @param x The input value to the ReLU function
         * @return double The derivative of the ReLU function at x
         */

        return x >= 0 ? 1 : 0;
    }

    std::vector<double> vectRelu(const std::vector<double> x)
    { /**
       * A vectorized version of the Rectified Linear Unit (ReLU) activation function.
       * @param x the input vector
       * @return a vector where each element is the ReLU of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(relu(i));
        return result;
    }

    std::vector<double> vectReluDerivative(const std::vector<double> x)
    { /**
       * A vectorized version of the derivative of the Rectified Linear Unit (ReLU) activation function.
       * @param x the input vector
       * @return a vector where each element is the derivative of the ReLU function of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(reluDerivative(i));
        return result;
    }

    double leakyRelu(double x, double alpha = 0.01) {

        /**
         * @brief Calculates the Leaky ReLU activation function
         *
         * The Leaky ReLU function is defined as f(x) = x if x > 0, or alpha*x otherwise
         * where alpha is a small constant (like 0.01). This function allows a small gradient
         * when x < 0, which can help with training deep neural networks.
         *
         * @param x The input value to the Leaky ReLU function
         * @param alpha The slope for x < 0, usually a small constant like 0.01
         * @return double The Leaky ReLU of x, which will be x if x > 0, or alpha*x otherwise
         */

        return x > 0 ? x : alpha * x;
    }

    double leakyReluDerivative(double x, double alpha = 0.01) {

        /**
         * @brief Calculates the derivative of the Leaky ReLU function
         *
         * The derivative of the Leaky ReLU function is f'(x) = 1 if x > 0, or alpha otherwise
         * This function is used in backpropagation to calculate gradients
         * and update weights in neural networks.
         *
         * @param x The input value to the Leaky ReLU function
         * @param alpha The slope for x < 0, usually a small constant like 0.01
         * @return double The derivative of the Leaky ReLU function at x
         */

        return x >= 0 ? 1 : alpha;
    }

    std::vector<double> vectLeakyRelu(const std::vector<double> x, double alpha = 0.01)
    { /**
       * A vectorized version of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
       * @param x the input vector
       * @param alpha the leak rate, defaults to 0.01
       * @return a vector where each element is the Leaky ReLU of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(leakyRelu(i, alpha));
        return result;
    }

    std::vector<double> vectLeakyReluDerivative(const std::vector<double> x, double alpha = 0.01)
    { /**
       * A vectorized version of the derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
       * @param x the input vector
       * @param alpha the leak rate, defaults to 0.01
       * @return a vector where each element is the derivative of the Leaky ReLU function of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(leakyReluDerivative(i, alpha));
        return result;
    }

    double tanh(double x) {

        /**
         * @brief Calculates the hyperbolic tangent of a number
         *
         * This function computes the hyperbolic tangent using the formula:
         * tanh(x) = (e^x - e^-x)/(e^x + e^-x)
         *
         * @param x The input value to calculate tanh for
         * @return double The hyperbolic tangent of x
         */

        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double tanhDerivative(double x) {

        /**
         * @brief Calculates the derivative of the hyperbolic tangent (tanh) function
         * @details The derivative of tanh(x) is 1 - tanh²(x)
         *
         * @param x The input value for which to calculate the derivative
         * @return The derivative value at point x
         */

        return 1 - pow(tanh(x), 2);
    }

    std::vector<double> vectTanh(const std::vector<double> x)
    { /**
       * A vectorized version of the Hyperbolic Tangent (tanh) activation function.
       * @param x the input vector
       * @return a vector where each element is the tanh of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(tanh(i));
        return result;
    }

    std::vector<double> vectTanhDerivative(const std::vector<double> x)
    { /**
       * A vectorized version of the derivative of the Hyperbolic Tangent (tanh) activation function.
       * @param x the input vector
       * @return a vector where each element is the derivative of the tanh function of the corresponding element in x
       */
        std::vector<double> result;
        result.reserve(x.size());
        for (double i : x)
            result.push_back(tanhDerivative(i));
        return result;
    }

#endif