#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <chrono>
#include <numeric>
#include <tbb/global_control.h>

class Layer;
class NeuralNetwork;
static constexpr uint32_t SEED = 123456u;

std::mt19937 generator(SEED);

auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                      << e.what() << std::endl;
        }
    }
};

sycl::queue q(sycl::gpu_selector_v, exception_handler);

struct Activation {
    // 0: sigmoid, 1: relu
    static inline double apply(int type, double x) {
        if (type == 0) {
            return 1.0 / (1.0 + sycl::exp(-x));
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
    double l1_lambda = 0.0;
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
    double l1_lambda;
    double l2_lambda;
    int activation_type;

    // Conv-specific params
    int kernel_size;
    int stride;
    int padding;

    // Host pointers
    double* h_weights;
    double* h_biases;
    double* h_delta;   

    // SYCL Buffers
    sycl::buffer<double> d_weights;
    sycl::buffer<double> d_biases;
    sycl::buffer<double> d_delta; 
    sycl::buffer<double> d_input_delta;

    Layer(const Params& p)
        : type(p.type),
          input_size(p.input_size),
          output_size(p.output_size),
          learning_rate(p.learning_rate),
          l1_lambda(p.l1_lambda),
          l2_lambda(p.l2_lambda),
          activation_type(p.activation_type),
          kernel_size(p.kernel_size),
          stride(p.stride),
          padding(p.padding),
          h_weights(nullptr), h_biases(nullptr), h_delta(nullptr),
          d_weights(sycl::range<1>(1)), d_biases(sycl::range<1>(1)),
          d_delta(sycl::range<1>(1)), d_input_delta(sycl::range<1>(1))
    {
        std::default_random_engine generator;

        if (type == 1) {
            h_weights = new double[input_size * output_size];
            h_biases = new double[output_size];
            
            double limit = std::sqrt(6.0 / (input_size + output_size));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (int i = 0; i < input_size * output_size; ++i) h_weights[i] = distribution(generator);
            for (int i = 0; i < output_size; ++i) h_biases[i] = 0.0;

            d_weights = sycl::buffer<double>(h_weights, sycl::range<1>(input_size * output_size));
            d_biases = sycl::buffer<double>(h_biases, sycl::range<1>(output_size));
            std::cout << "Dense Layer created: " << input_size << " -> " << output_size << std::endl;

        } else if (type == 0) {
            this->output_size = ((input_size + 2 * padding - kernel_size) / stride) + 1;
            h_weights = new double[kernel_size];
            h_biases = new double[1];

            double limit = std::sqrt(6.0 / (kernel_size + this->output_size));
            std::uniform_real_distribution<double> distribution(-limit, limit);
            for (int i = 0; i < kernel_size; ++i) h_weights[i] = distribution(generator);
            h_biases[0] = 0.0;

            d_weights = sycl::buffer<double>(h_weights, sycl::range<1>(kernel_size));
            d_biases = sycl::buffer<double>(h_biases, sycl::range<1>(1));
            std::cout << "Conv1D Layer created: input=" << input_size << ", kernel=" << kernel_size 
                      << ", stride=" << stride << " -> output=" << this->output_size << std::endl;
        }

        h_delta = new double[this->output_size * 400];
        d_delta = sycl::buffer<double>(sycl::range<1>(this->output_size * 400));
        d_input_delta = sycl::buffer<double>(sycl::range<1>(this->input_size * 400));
    }

    ~Layer() {
        delete[] h_delta;
    }

    void forward(sycl::buffer<double>& input_buf, sycl::buffer<double>& output_buf, int batch_size) {
        if (type == 1) {
            q.submit([&](sycl::handler& h) {
                auto input_acc = input_buf.get_access<sycl::access::mode::read>(h);
                auto weights_acc = d_weights.get_access<sycl::access::mode::read>(h);
                auto biases_acc = d_biases.get_access<sycl::access::mode::read>(h);
                auto output_acc = output_buf.get_access<sycl::access::mode::write>(h);

                int local_input_size = input_size;
                int local_output_size = output_size;
                int local_activation_type = activation_type;

                h.parallel_for(sycl::range<2>(batch_size, local_output_size), [=](sycl::id<2> item) {
                    int batch_idx = item[0];
                    int neuron_idx = item[1];
                    double sum = 0.0;
                    for (int i = 0; i < local_input_size; ++i) {
                        sum += weights_acc[neuron_idx * local_input_size + i] * input_acc[batch_idx * local_input_size + i];
                    }
                    sum += biases_acc[neuron_idx];
                    output_acc[batch_idx * local_output_size + neuron_idx] = Activation::apply(local_activation_type, sum);
                });
            });
        } else if (type == 0) {
            q.submit([&](sycl::handler& h) {
                auto input_acc = input_buf.get_access<sycl::access::mode::read>(h);
                auto kernel_acc = d_weights.get_access<sycl::access::mode::read>(h);
                auto bias_acc = d_biases.get_access<sycl::access::mode::read>(h);
                auto output_acc = output_buf.get_access<sycl::access::mode::write>(h);

                int local_input_size = input_size;
                int local_output_size = output_size;
                int local_kernel_size = kernel_size;
                int local_stride = stride;
                int local_padding = padding;
                int local_activation_type = activation_type;

                h.parallel_for(sycl::range<2>(batch_size, local_output_size), [=](sycl::id<2> item) {
                    int batch_idx = item[0];
                    int out_idx = item[1];
                    double sum = 0.0;
                    for (int k = 0; k < local_kernel_size; ++k) {
                        int in_idx = out_idx * local_stride + k - local_padding;
                        if (in_idx >= 0 && in_idx < local_input_size) {
                            sum += input_acc[batch_idx * local_input_size + in_idx] * kernel_acc[k];
                        }
                    }
                    sum += bias_acc[0];
                    output_acc[batch_idx * local_output_size + out_idx] = Activation::apply(local_activation_type, sum);
                });
            });
        }
    }
    
    // Calculates the delta for a hidden layer
    void calculate_delta(sycl::buffer<double>& this_layer_output_buf, Layer* next_layer, int batch_size) {
        q.submit([&](sycl::handler& h) {
            auto this_output_acc = this_layer_output_buf.get_access<sycl::access::mode::read>(h);
            auto next_delta_acc = next_layer->d_delta.get_access<sycl::access::mode::read>(h);
            auto next_weights_acc = next_layer->d_weights.get_access<sycl::access::mode::read>(h);
            auto this_delta_acc = d_delta.get_access<sycl::access::mode::write>(h);
            
            int this_out_size = output_size;
            int next_out_size = next_layer->output_size;
            int next_in_size = next_layer->input_size;
            int local_activation_type = activation_type;

            h.parallel_for(sycl::range<2>(batch_size, this_out_size), [=](sycl::id<2> item) {
                int batch_idx = item[0];
                int neuron_idx = item[1];
                double error = 0.0;

                for (int k = 0; k < next_out_size; ++k) {
                    error += next_delta_acc[batch_idx * next_out_size + k] * next_weights_acc[k * next_in_size + neuron_idx];
                }
                double current_output = this_output_acc[batch_idx * this_out_size + neuron_idx];
                this_delta_acc[batch_idx * this_out_size + neuron_idx] = error * Activation::derivative(local_activation_type, current_output);
            });
        });
    }

    // Updates the parameters (weights/kernel and biases) of the layer
    void update_parameters(sycl::buffer<double>& prev_layer_output_buf, int batch_size) {
        if (type == 1) {
            // Update weights
            q.submit([&](sycl::handler& h) {
                auto prev_output_acc = prev_layer_output_buf.get_access<sycl::access::mode::read>(h);
                auto this_delta_acc = d_delta.get_access<sycl::access::mode::read>(h);
                auto this_weights_acc = d_weights.get_access<sycl::access::mode::read_write>(h);

                int local_input_size = input_size;
                int local_output_size = output_size;
                double local_l1 = l1_lambda;
                double local_l2 = l2_lambda;
                double local_lr = learning_rate;

                h.parallel_for(sycl::range<2>(local_output_size, local_input_size), [=](sycl::id<2> item) {
                    int neuron_idx = item[0];
                    int weight_idx = item[1];
                    double grad = 0.0;
                    for(int b = 0; b < batch_size; ++b) {
                        grad += this_delta_acc[b * local_output_size + neuron_idx] * prev_output_acc[b * local_input_size + weight_idx];
                    }
                    
                    double current_weight = this_weights_acc[neuron_idx * local_input_size + weight_idx];

                    double sign_w = (current_weight > 0.0) ? 1.0 : ((current_weight < 0.0) ? -1.0 : 0.0);
                    
                    grad = (grad / batch_size) + local_l2 * current_weight + local_l1 * sign_w;
                    
                    this_weights_acc[neuron_idx * local_input_size + weight_idx] -= local_lr * grad;
                });
            });
            // Update biases
            q.submit([&](sycl::handler& h) {
                auto this_delta_acc = d_delta.get_access<sycl::access::mode::read>(h);
                auto this_biases_acc = d_biases.get_access<sycl::access::mode::read_write>(h);

                int local_output_size = output_size;
                double local_lr = learning_rate;

                h.parallel_for(sycl::range<1>(local_output_size), [=](sycl::id<1> neuron_idx) {
                    double grad = 0.0;
                    for(int b = 0; b < batch_size; ++b) grad += this_delta_acc[b * local_output_size + neuron_idx];
                    this_biases_acc[neuron_idx] -= local_lr * (grad / batch_size);
                });
            });
        } else if (type == 0) {
             // Update kernel
            q.submit([&](sycl::handler& h) {
                auto prev_output_acc = prev_layer_output_buf.get_access<sycl::access::mode::read>(h);
                auto this_delta_acc = d_delta.get_access<sycl::access::mode::read>(h);
                auto this_kernel_acc = d_weights.get_access<sycl::access::mode::read_write>(h);

                int local_input_size = input_size;
                int local_output_size = output_size;
                int local_kernel_size = kernel_size;
                int local_stride = stride;
                int local_padding = padding;
                double local_l1 = l1_lambda;
                double local_l2 = l2_lambda;
                double local_lr = learning_rate;

                h.parallel_for(sycl::range<1>(local_kernel_size), [=](sycl::id<1> k) {
                    double grad = 0.0;
                    for (int b = 0; b < batch_size; ++b) {
                        for (int i = 0; i < local_output_size; ++i) {
                            int in_idx = i * local_stride + k - local_padding;
                            if (in_idx >= 0 && in_idx < local_input_size) {
                                grad += prev_output_acc[b * local_input_size + in_idx] * this_delta_acc[b * local_output_size + i];
                            }
                        }
                    }
                    
                    double current_kernel_weight = this_kernel_acc[k];

                    double sign_k = (current_kernel_weight > 0.0) ? 1.0 : ((current_kernel_weight < 0.0) ? -1.0 : 0.0);
                    
                    double total_grad = (grad / batch_size) + local_l2 * current_kernel_weight + local_l1 * sign_k;

                    this_kernel_acc[k] -= local_lr * total_grad;
                });
            });
            // Update bias
            sycl::buffer<double> temp_grad_buf(1);
            {
                sycl::host_accessor w(temp_grad_buf, sycl::write_only);
                w[0] = 0.0;
            }
            q.submit([&](sycl::handler& cgh) {
                auto reduction = sycl::reduction(temp_grad_buf, cgh, sycl::plus<>());
                auto delta_acc_read = d_delta.get_access<sycl::access::mode::read>(cgh);
                cgh.parallel_for(sycl::range<1>(batch_size * output_size), reduction, [=](sycl::id<1> idx, auto& sum) {
                    sum += delta_acc_read[idx];
                });
            });

            q.submit([&](sycl::handler& cgh) {
                auto temp_grad_acc = temp_grad_buf.get_access<sycl::access::mode::read>(cgh);
                auto bias_acc_write = d_biases.get_access<sycl::access::mode::read_write>(cgh);
                double local_lr = learning_rate;
                cgh.single_task([=]() {
                    bias_acc_write[0] -= local_lr * (temp_grad_acc[0] / batch_size);
                });
            });
        }
    }
};

class NeuralNetwork {
public:
    std::vector<Layer*> layers;
    std::vector<sycl::buffer<double>> layer_outputs;

    ~NeuralNetwork() {
        for (auto layer : layers) {
            delete layer;
        }
    }

    void add_layer(const Params& p) {
        layers.push_back(new Layer(p));
    }

    double* predict(double* inputs, int num_samples, int num_features, int batch_size) {
        double* all_predictions = new double[num_samples * layers.back()->output_size];
        sycl::buffer<double> d_all_inputs(inputs, sycl::range<1>(num_samples * num_features));

        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - i);
            if (current_batch_size <= 0) continue;

            sycl::buffer<double> d_batch_inputs(d_all_inputs, i * num_features, current_batch_size * num_features);
            
            layers[0]->forward(d_batch_inputs, layer_outputs[0], current_batch_size);
            for (size_t l = 1; l < layers.size(); ++l) {
                layers[l]->forward(layer_outputs[l-1], layer_outputs[l], current_batch_size);
            }

            sycl::buffer<double> final_output_buf = layer_outputs.back();
            sycl::buffer<double> host_batch_preds(all_predictions + i * layers.back()->output_size, sycl::range<1>(current_batch_size * layers.back()->output_size));
            
            q.submit([&](sycl::handler& h) {
                auto src = final_output_buf.get_access<sycl::access::mode::read>(h, sycl::range<1>(current_batch_size * layers.back()->output_size), sycl::id<1>(0));
                auto dst = host_batch_preds.get_access<sycl::access::mode::write>(h);
                h.copy(src, dst);
            }).wait_and_throw();
        }

        return all_predictions;
    }

    void train(double* inputs, double* labels, int num_samples, int num_features, int num_classes, int epochs, int batch_size) {
        std::cout << "Starting training on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        sycl::buffer<double> d_all_inputs(inputs, sycl::range<1>(num_samples * num_features));
        sycl::buffer<double> d_all_labels(labels, sycl::range<1>(num_samples * num_classes));

        layer_outputs.clear();
        for (const auto& layer : layers) {
            layer_outputs.push_back(sycl::buffer<double>(sycl::range<1>(batch_size * layer->output_size)));
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int e = 0; e < epochs; ++e) {
            for (int i = 0; i < num_samples; i += batch_size) {
                int current_batch_size = std::min(batch_size, num_samples - i);
                if (current_batch_size <= 0) continue;

                sycl::buffer<double> d_batch_inputs(d_all_inputs, i * num_features, current_batch_size * num_features);
                sycl::buffer<double> d_batch_labels(d_all_labels, i * num_classes, current_batch_size * num_classes);

                layers[0]->forward(d_batch_inputs, layer_outputs[0], current_batch_size);
                for (size_t l = 1; l < layers.size(); ++l) {
                    layers[l]->forward(layer_outputs[l-1], layer_outputs[l], current_batch_size);
                }

                Layer* output_layer = layers.back();
                q.submit([&](sycl::handler& h) {
                    auto output_acc = layer_outputs.back().get_access<sycl::access::mode::read>(h);
                    auto label_acc = d_batch_labels.get_access<sycl::access::mode::read>(h);
                    auto delta_acc = output_layer->d_delta.get_access<sycl::access::mode::write>(h);
                    h.parallel_for(sycl::range<1>(current_batch_size * num_classes), [=](sycl::id<1> idx) {
                        delta_acc[idx] = output_acc[idx] - label_acc[idx];
                    });
                });

                for (int l = layers.size() - 1; l >= 0; --l) {
                    if (l < layers.size() - 1) {
                        layers[l]->calculate_delta(layer_outputs[l], layers[l+1], current_batch_size);
                    }
                    
                    sycl::buffer<double>& prev_layer_output = (l > 0) ? layer_outputs[l-1] : d_batch_inputs;
                    layers[l]->update_parameters(prev_layer_output, current_batch_size);
                }
            }
             q.wait_and_throw();
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
    double l1 = 0.0001;
    double l2 = 0.0001;
    int batch_size = 400;

	tbb::global_control control(tbb::global_control::max_allowed_parallelism, 12);

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
    conv1_params.l1_lambda = l1; 
    conv1_params.l2_lambda = l2;
    conv1_params.activation_type = 1; // ReLU
    Params conv2_params;
    conv2_params.type = 0; // Conv
    conv2_params.input_size = columns-4;
    conv2_params.kernel_size = 5;
    conv2_params.stride = 1;
    conv2_params.learning_rate = learning_rate;
    conv2_params.l1_lambda = l1; 
    conv2_params.l2_lambda = l2;
    conv2_params.activation_type = 1;
    int conv1_output_size = ((conv1_params.input_size + 2 * conv1_params.padding - conv1_params.kernel_size) / conv1_params.stride) + 1;

    Params dense1_params;
    dense1_params.type = 1;
    dense1_params.input_size = 25;
    dense1_params.output_size = 45;
    dense1_params.learning_rate = learning_rate;
    dense1_params.l1_lambda = l1;
    dense1_params.l2_lambda = l2;
    dense1_params.activation_type = 1; // ReLU
    
    Params dense2_params;
    dense2_params.type = 1;
    dense2_params.input_size = 45;
    dense2_params.output_size = 20;
    dense2_params.learning_rate = learning_rate;
    dense2_params.l1_lambda = l1;
    dense2_params.l2_lambda = l2;
    dense2_params.activation_type = 1; // ReLU
    
    Params dense_output_params;
    dense_output_params.type = 1;
    dense_output_params.input_size = 20;
    dense_output_params.output_size = num_classes;
    dense_output_params.learning_rate = learning_rate;
    dense_output_params.l1_lambda = l1;
    dense_output_params.l2_lambda = l2;
    dense_output_params.activation_type = 0; // Sigmoid for output

    nn.add_layer(conv1_params);
	nn.add_layer(conv2_params);
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
