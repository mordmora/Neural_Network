#ifndef UTILS_CPP
#define UTILS_CPP

#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>


    
    double dotProduct(std::vector<double>& v1, std::vector<double>& v2) {
    /**
     * @brief Calculates the dot product of two vectors
     * 
     * @param v1 First vector of doubles
     * @param v2 Second vector of doubles
     * @return double The dot product of the two vectors
     * @throws None
     * @pre Vectors must be of equal size
     */
        
        double res = 0;
        for (int i = 0; i < v1.size(); i++) {
            res += v1[i] * v2[i];
        }

        return res;
    }


    std::vector<double> scalarVectorMultiplication(std::vector<double>& v, double scalar) {

    /**
     * @brief Multiplies each element of a vector by a scalar value
     * 
     * @param v Reference to the vector to be multiplied
     * @param scalar The scalar value to multiply each element by
     * @return std::vector<double> The modified vector with all elements multiplied by the scalar
     * 
     * @note The input vector is modified in place and then returned
     */

        std::transform(v.begin(), v.end(), v.begin(),
            std::bind(std::multiplies<double>(), std::placeholders::_1, scalar));
        return v;
    }

    std::vector<double> subtract(std::vector<double>& v1, std::vector<double>& v2) {
    /**
     * @brief Performs element-wise subtraction of two vectors
     * 
     * Takes two vectors of equal size and subtracts the elements of the second vector
     * from the corresponding elements of the first vector.
     * 
     * @param v1 First vector (minuend)
     * @param v2 Second vector (subtrahend)
     * @return std::vector<double> Result vector containing the differences
     * @note Both input vectors must be of the same size
     */
        std::vector<double> output;

        std::transform(v1.begin(),
            v1.end(), v2.begin(), std::back_inserter(output), std::minus<double>());

        return output;
    }

    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>>& m) {
    /**
     * @brief Transposes a 2D matrix represented as a vector of vectors
     * 
     * This function takes a matrix (vector of vectors) and returns its transpose,
     * where rows become columns and columns become rows.
     * 
     * @param m Reference to the input matrix to be transposed
     * @return std::vector<std::vector<double>> The transposed matrix
     * 
     * @note The function assumes the input matrix is not empty and all rows have the same size
     * @note Time complexity: O(rows * cols) where rows and cols are dimensions of input matrix
     * @note Space complexity: O(rows * cols) for the new transposed matrix
     */

        std::vector<std::vector<double>> trans_vec(m[0].size(), std::vector<double>(m.size()));

        for (int i = 0; i < m.size(); i++) {
            for (int j = 0; j < m[i].size(); j++) {
                if (trans_vec[j].size() != m.size()) trans_vec[j].resize(m.size());
                trans_vec[j][i] = m[i][j];
            }
        }
        return trans_vec;
    }



    std::vector<std::vector<double>> uniformWeightInitializer(int rows, int cols){
    /**
     * @brief Initializes a 2D weight matrix with uniform random values between -1 and 1
     * 
     * This function creates a matrix of specified dimensions and fills it with random
     * values drawn from a uniform distribution in the range [-1.0, 1.0]. It uses
     * a Mersenne Twister engine (mt19937) seeded with a combination of random_device
     * and current system time to ensure good randomization.
     * 
     * @param rows The number of rows in the weight matrix
     * @param cols The number of columns in the weight matrix
     * @return std::vector<std::vector<double>> A 2D vector containing the initialized weights
     */
        std::random_device rd;
        std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = dis(gen);
            }
        }
        return weights;
    }


    std::vector<double> biasInit(int size){
        /**
         * @brief Initializes a vector of bias values
         * @param size The size of the bias vector to create
         * @return A vector of double values initialized as biases
         * 
         * Creates a vector of specified size filled with bias initialization values.
         * These values can be used as initial biases in neural network layers.
         */
        std::random_device rd;
        std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        std::vector<double> bias(size);
        for (int i = 0; i < size; i++) {
            bias[i] = dis(gen);
        }
        return bias;
    }


    
#endif