#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>

class Layer;
class NeuralNetwork;
static constexpr uint32_t SEED = 123456u;

std::mt19937 generator(SEED);

struct Activation {
    // 0: sigmoid, 1: relu
    static inline double apply(int type, double x) {
        if (type == 0) {
            return 1.0 / (1.0 + std::exp(-x));
        }
        if (type == 1) {
            return x > 0 ? x : 0;
        }
        return x;
    }

    static inline double derivative(int type, double output) {
        if (type == 0) {
            return output * (1.0 - output);
        }
        if (type == 1) {
            return output > 0 ? 1.0 : 0.0;
        }
        return 1.0;
    }
};

struct Params {
    int type = 1; // 0 = Conv, 1 = Dense
    int input_size = 0;
    int output_size = 0;
    double learning_rate = 0.01;
    double l2_lambda = 0.0;
    int activation_type = 0;
    // Conv specific
    int kernel_size = 3;
    int stride = 1;
    int padding = 0;
};

class Layer {
public:
    int type;
    int input_size;
    int output_size;
    double learning_rate;
    double l2_lambda;
    int activation_type;

    // Conv-specific params
    int kernel_size;
    int stride;
    int padding;

    // Pointers for weights, biases, and gradients
    double* weights; 
    double* biases;  
    double* delta;   

    Layer(const Params& p)
        : type(p.type),
          input_size(p.input_size),
          output_size(p.output_size),
          learning_rate(p.learning_rate),
          l2_lambda(p.l2_lambda),
          activation_type(p.activation_type),
          kernel_size(p.kernel_size),
          stride(p.stride),
          padding(p.padding),
          weights(nullptr), biases(nullptr), delta(nullptr)
    {
        std::default_random_engine generator;

        if (type == 1) { // Dense
            weights = new double[input_size * output_size];
            biases = new double[output_size];
            
            double limit = std::sqrt(6.0 / (input_size + output_size));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (int i = 0; i < input_size * output_size; ++i) weights[i] = distribution(generator);
            for (int i = 0; i < output_size; ++i) biases[i] = 0.0;
            std::cout << "Dense Layer created: " << input_size << " -> " << output_size << std::endl;

        } else if (type == 0) { // Conv
            this->output_size = ((input_size + 2 * padding - kernel_size) / stride) + 1;
            weights = new double[kernel_size]; 
            biases = new double[1]; 

            double limit = std::sqrt(6.0 / (kernel_size + this->output_size));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (int i = 0; i < kernel_size; ++i) weights[i] = distribution(generator);
            biases[0] = 0.0;
            std::cout << "Conv1D Layer created: input=" << input_size << ", kernel=" << kernel_size 
                      << ", stride=" << stride << " -> output=" << this->output_size << std::endl;
        }
        delta = new double[this->output_size * 400];
    }

    ~Layer() {
        delete[] weights;
        delete[] biases;
        delete[] delta;
    }

    void forward(double* input_buf, double* output_buf, int batch_size) {
        for (int b = 0; b < batch_size; ++b) {
            if (type == 1) { // Dense
                for (int j = 0; j < output_size; ++j) {
                    double sum = 0.0;
                    for (int i = 0; i < input_size; ++i) {
                        sum += input_buf[b * input_size + i] * weights[j * input_size + i];
                    }
                    sum += biases[j];
                    output_buf[b * output_size + j] = Activation::apply(activation_type, sum);
                }
            } else if (type == 0) { // Conv
                for (int out_idx = 0; out_idx < output_size; ++out_idx) {
                    double sum = 0.0;
                    for (int k = 0; k < kernel_size; ++k) {
                        int in_idx = out_idx * stride + k - padding;
                        if (in_idx >= 0 && in_idx < input_size) {
                            sum += input_buf[b * input_size + in_idx] * weights[k];
                        }
                    }
                    sum += biases[0];
                    output_buf[b * output_size + out_idx] = Activation::apply(activation_type, sum);
                }
            }
        }
    }
    
    void calculate_delta(double* this_layer_output, Layer* next_layer, int batch_size) {
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < output_size; ++j) {
                double error = 0.0;
                for (int k = 0; k < next_layer->output_size; ++k) {
                    error += next_layer->delta[b * next_layer->output_size + k] * next_layer->weights[k * next_layer->input_size + j];
                }
                double current_output = this_layer_output[b * output_size + j];
                delta[b * output_size + j] = error * Activation::derivative(activation_type, current_output);
            }
        }
    }

    void update_parameters(double* prev_layer_output, int batch_size) {
        if (type == 1) {
            for (int j = 0; j < output_size; ++j) {
                for (int i = 0; i < input_size; ++i) {
                    double grad = 0.0;
                    for(int b = 0; b < batch_size; ++b) {
                        grad += delta[b * output_size + j] * prev_layer_output[b * input_size + i];
                    }
                    grad = (grad / batch_size) + l2_lambda * weights[j * input_size + i];
                    weights[j * input_size + i] -= learning_rate * grad;
                }
                double bias_grad = 0.0;
                for(int b = 0; b < batch_size; ++b) bias_grad += delta[b * output_size + j];
                biases[j] -= learning_rate * (bias_grad / batch_size);
            }
        } else if (type == 0) {
            for (int k = 0; k < kernel_size; ++k) {
                double grad = 0.0;
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < output_size; ++i) {
                        int in_idx = i * stride + k - padding;
                        if (in_idx >= 0 && in_idx < input_size) {
                            grad += prev_layer_output[b * input_size + in_idx] * delta[b * output_size + i];
                        }
                    }
                }
                weights[k] -= learning_rate * (grad / batch_size);
            }
            double bias_grad = 0.0;
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < output_size; ++i) {
                    bias_grad += delta[b * output_size + i];
                }
            }
            biases[0] -= learning_rate * (bias_grad / batch_size);
        }
    }
};

class NeuralNetwork {
public:
    std::vector<Layer*> layers;
    std::vector<double*> layer_outputs;

    ~NeuralNetwork() {
        for (auto layer : layers) {
            delete layer;
        }
        for (auto out : layer_outputs) {
            delete[] out;
        }
    }

    void add_layer(const Params& p) {
        layers.push_back(new Layer(p));
    }

    double* predict(double* inputs, int num_samples, int num_features, int batch_size) {
        double* all_predictions = new double[num_samples * layers.back()->output_size];

        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - i);
            if (current_batch_size <= 0) continue;

            double* current_input = inputs + i * num_features;
            
            layers[0]->forward(current_input, layer_outputs[0], current_batch_size);
            for (size_t l = 1; l < layers.size(); ++l) {
                layers[l]->forward(layer_outputs[l-1], layer_outputs[l], current_batch_size);
            }

            for(int j = 0; j < current_batch_size * layers.back()->output_size; ++j) {
                all_predictions[i * layers.back()->output_size + j] = layer_outputs.back()[j];
            }
        }
        return all_predictions;
    }

    void train(double* inputs, double* labels, int num_samples, int num_features, int num_classes, int epochs, int batch_size) {
        std::cout << "Starting sequential training on CPU..." << std::endl;

        for (const auto& layer : layers) {
            layer_outputs.push_back(new double[batch_size * layer->output_size]);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int e = 0; e < epochs; ++e) {
            for (int i = 0; i < num_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, num_samples - i);
                if (current_batch_size <= 0) continue;

                double* batch_inputs = inputs + i * num_features;
                double* batch_labels = labels + i * num_classes;

                layers[0]->forward(batch_inputs, layer_outputs[0], current_batch_size);
                for (size_t l = 1; l < layers.size(); ++l) {
                    layers[l]->forward(layer_outputs[l-1], layer_outputs[l], current_batch_size);
                }

                Layer* output_layer = layers.back();
                for (int b = 0; b < current_batch_size; ++b) {
                    for (int j = 0; j < num_classes; ++j) {
                        output_layer->delta[b * num_classes + j] = layer_outputs.back()[b * num_classes + j] - batch_labels[b * num_classes + j];
                    }
                }

                for (int l = layers.size() - 1; l >= 0; --l) {
                    if (l < layers.size() - 1) {
                        layers[l]->calculate_delta(layer_outputs[l], layers[l+1], current_batch_size);
                    }
                    double* prev_layer_output = (l > 0) ? layer_outputs[l-1] : batch_inputs;
                    layers[l]->update_parameters(prev_layer_output, current_batch_size);
                }
            }
             if ((e + 1) % 100 == 0) {
                std::cout << "Epoch " << e + 1 << "/" << epochs << " completed." << std::endl;
             }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        std::cout << "Training finished in " << elapsed.count() << " seconds." << std::endl;

        std::cout << "Calculating final accuracy..." << std::endl;
        double* predictions = predict(inputs, num_samples, num_features, batch_size);
        
        int correct_predictions = 0;
        for (int i = 0; i < num_samples; ++i) {
            int pred_label = 0;
            double max_pred = predictions[i * num_classes];
            for (int j = 1; j < num_classes; ++j) {
                if (predictions[i * num_classes + j] > max_pred) {
                    max_pred = predictions[i * num_classes + j];
                    pred_label = j;
                }
            }

            int true_label = 0;
            double max_label = labels[i * num_classes];
             for (int j = 1; j < num_classes; ++j) {
                if (labels[i * num_classes + j] > max_label) {
                    max_label = labels[i * num_classes + j];
                    true_label = j;
                }
            }

            if (pred_label == true_label) {
                correct_predictions++;
            }
        }

        double accuracy = (double)correct_predictions / num_samples;
        std::cout << "Final Accuracy: " << accuracy * 100.0 << "%" << std::endl;

        delete[] predictions;
    }
};

std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);
    std::string line;
    if (file.good()) {
        std::getline(file, line);
    }
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                row.push_back(std::stod(item));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Could not convert string to double: " << item << std::endl;
                row.push_back(0.0);
            }
        }
        data.push_back(row);
    }
    return data;
}

double* flatten(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return nullptr;
    int rows = data.size();
    int cols = data[0].size();
    double* flat = new double[rows * cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat[i * cols + j] = data[i][j];
        }
    }
    return flat;
}

void normalize_data(double* data, int rows, int cols) {
    for (int j = 0; j < cols; ++j) { 
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < rows; ++i) {
            double val = data[i * cols + j];
            sum += val;
            sum_sq += val * val;
        }
        
        double mean = sum / rows;
        double std_dev = std::sqrt(sum_sq / rows - mean * mean);

        if (std_dev > 1e-8) { 
            for (int i = 0; i < rows; ++i) {
                data[i * cols + j] = (data[i * cols + j] - mean) / std_dev;
            }
        }
    }
}

std::vector<std::pair<double, double>> get_normalization_params(double* data, int rows, int cols) {
    std::vector<std::pair<double, double>> params;
    for (int j = 0; j < cols; ++j) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < rows; ++i) {
            double val = data[i * cols + j];
            sum += val;
            sum_sq += val * val;
        }
        double mean = sum / rows;
        double std_dev = std::sqrt(sum_sq / rows - mean * mean);
        params.push_back({mean, std_dev});
    }
    return params;
}

void apply_normalization(double* data, int rows, int cols, const std::vector<std::pair<double, double>>& params) {
    for (int j = 0; j < cols; ++j) {
        double mean = params[j].first;
        double std_dev = params[j].second;
        if (std_dev > 1e-8) {
            for (int i = 0; i < rows; ++i) {
                data[i * cols + j] = (data[i * cols + j] - mean) / std_dev;
            }
        }
    }
}

int main() {
    int num_classes = 2;
    int epochs = 200;
    double learning_rate = 0.0001;
    double l2 = 0.0001;
    int batch_size = 400;

    std::cout << "Loading data for trainging..." << std::endl;
    auto X_vec = read_csv("NoClass11.csv");
    auto y_vec = read_csv("labels.csv");

    if (X_vec.empty() || y_vec.empty()) {
        std::cerr << "Error: Could not read data files. Make sure 'NoClass11.csv' and 'labels.csv' are present." << std::endl;
        return 1;
    }

    int rows = X_vec.size();
    int columns = X_vec[0].size();

    double* X_flat = flatten(X_vec);
    double* y_flat = flatten(y_vec);
    std::cout << "Data loaded: " << rows << " samples, " << columns << " features." << std::endl;

    std::cout << "Calculating normalization params from training data..." << std::endl;
    auto norm_params = get_normalization_params(X_flat, rows, columns);

    std::cout << "Normalizing training data..." << std::endl;
    apply_normalization(X_flat, rows, columns, norm_params);

    NeuralNetwork nn;

    Params conv1_params;
    conv1_params.type = 0; // Conv
    conv1_params.input_size = columns;
    conv1_params.kernel_size = 5;
    conv1_params.stride = 1;
    conv1_params.learning_rate = learning_rate;
    conv1_params.l2_lambda = l2;
    conv1_params.activation_type = 1; // ReLU
    
    int conv1_output_size = ((conv1_params.input_size + 2 * conv1_params.padding - conv1_params.kernel_size) / conv1_params.stride) + 1;

    Params dense1_params;
    dense1_params.type = 1;
    dense1_params.input_size = conv1_output_size;
    dense1_params.output_size = 45;
    dense1_params.learning_rate = learning_rate;
    dense1_params.l2_lambda = l2;
    dense1_params.activation_type = 1; // ReLU
    
    Params dense2_params;
    dense2_params.type = 1;
    dense2_params.input_size = 45;
    dense2_params.output_size = 20;
    dense2_params.learning_rate = learning_rate;
    dense2_params.l2_lambda = l2;
    dense2_params.activation_type = 1; // ReLU
    
    Params dense_output_params;
    dense_output_params.type = 1;
    dense_output_params.input_size = 20;
    dense_output_params.output_size = num_classes;
    dense_output_params.learning_rate = learning_rate;
    dense_output_params.l2_lambda = l2;
    dense_output_params.activation_type = 0; // Sigmoid for output

    nn.add_layer(conv1_params);
    nn.add_layer(dense1_params);
    nn.add_layer(dense2_params);
    nn.add_layer(dense_output_params);

    nn.train(X_flat, y_flat, rows, columns, num_classes, epochs, batch_size);

    std::cout << "Loading data for testing..." << std::endl;
    auto X_vec_test = read_csv("NoClassTest.csv");
    auto y_vec_test = read_csv("labelsTest.csv");

    if (X_vec_test.empty() || y_vec_test.empty()) {
        std::cerr << "Error: Could not read data files. Make sure 'NoClassTest.csv' and 'labelsTest.csv' are present." << std::endl;
        return 1;
    }

    rows = X_vec_test.size();
    columns = X_vec_test[0].size();

    auto X_test_flat = flatten(X_vec_test);
    auto y_flat_test = flatten(y_vec_test);
    std::cout << "Data loaded: " << rows << " samples, " << columns << " features." << std::endl;

    std::cout << "Normalizing test data with training params..." << std::endl;
    apply_normalization(X_test_flat, rows, columns, norm_params);

    std::cout << "Calculating test accuracy..." << std::endl;
    double* test_predictions = nn.predict(X_flat, rows, columns, batch_size);
    int correct_test_predictions = 0;
    for (int i = 0; i < rows; ++i) {
        int pred_label = 0;
        double max_pred = test_predictions[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            if (test_predictions[i * num_classes + j] > max_pred) {
                max_pred = test_predictions[i * num_classes + j];
                pred_label = j;
            }
        }
        int true_label = 0;
        double max_label = y_flat[i * num_classes];
        for (int j = 1; j < num_classes; ++j) {
            if (y_flat[i * num_classes + j] > max_label) {
                max_label = y_flat[i * num_classes + j];
                true_label = j;
            }
        }
        if (pred_label == true_label) {
            correct_test_predictions++;
        }

    }
    double test_accuracy = (double)correct_test_predictions / rows;
    std::cout << "Test Accuracy: " << test_accuracy * 100.0 << "%" << std::endl;

    delete[] X_flat;
    delete[] y_flat;

    return 0;
}
