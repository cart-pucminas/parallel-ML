#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <array>
#include <limits>
#include <vector>
#include <iostream>
#include <string>

#include "util.h"
#include "CART.h"

struct FeatureSplit {
    double gini;
    float threshold;
    int feature_idx;
};

void split(const std::vector<std::vector<float>> &X, const std::vector<int> &index_sample, int i_feature, double threshold, std::vector<int> &i_left, std::vector<int> &i_right){
    for(int i : index_sample){
        if (X[i][i_feature]>threshold)
            i_right.push_back(i);
        else
            i_left.push_back(i);
    }
    return;
}

std::string class_to_word(int classe){
    if (classe == 0) {
        return "Normal";
    } 
    else if (classe == 1) {
        return "RPM_Spoofing";
    } 
    else if (classe == 2) {
        return "DoS";
    } 
    else if (classe == 3) {
        return "Gear_Spoofing";
    } 
    else if (classe == 4) {
        return "Fuzzy";
    } 
    else {
        return "Desconhecido";
    }
}


void print_tree(Node* node, std::vector<std::string> &headers, std::string prefix, bool is_left, bool is_root) {
    if (node == nullptr) {
        return;
    }

    std::cout << prefix;

    if (!is_root) {
        std::cout << (is_left ? "├── Sim: " : "└── Nao: ");
    }

    // Lógica de impressão do Nó
    if (node->is_leaf()) {
        std::cout << "Predicao: Classe " << node->predicted_class << " " << class_to_word(node->predicted_class) << std::endl;
    } else {
        std::cout << "[Feature " << headers[node->feature_index] << " <= " << node->threshold << "]" << std::endl;

        std::string next_prefix = prefix + (is_root ? "" : (is_left ? "│   " : "    "));

        print_tree(node->l, headers, next_prefix, true, false);  // Filho esquerdo
        print_tree(node->r, headers, next_prefix, false, false); // Filho direito
    }
}

Node* build_tree_rec(std::vector<std::vector<float>> &X, std::vector<int> &y, 
                     std::vector<int> &index, 
                     int depth, int max_depth,
                     sycl::queue &q){

    Node* node = new Node();

    int n_features = X[0].size();
    int n_instances = index.size();
    int min_sample_split = 50;
    const int n_classes = 5;

    std::array<int, 5> class_counter = {0};
    for (int i : index){
        class_counter[y[i]]++; // 0 = normal, 1 = DOS... acessar class_counter[y] acessa casa responsável pela classe de mesmo valor
    }

    int majority_class = -1;
    int max_count = -1;
    for(int c = 0; c < n_classes; c++) {
        if(class_counter[c] > max_count) {
            max_count = class_counter[c];
            majority_class = c;
        }
    }

    node->predicted_class = majority_class;

    if (depth >= max_depth || max_count == n_instances || n_instances < min_sample_split ) {
        return node;
    } 

    double final_gini = std::numeric_limits<double>::max();
    int final_feature = -1;
    float final_threshold = 0.0;
    std::vector<int> final_i_left, final_i_right;

    // Arrange matrixes for indexes
    std::vector<float> all_keys(n_features * n_instances);
    std::vector<int> all_sorted_indices(n_features * n_instances);
    std::vector<FeatureSplit> host_best_splits(n_features);

    for (int f = 0; f < n_features; f++) {
        for (int i = 0; i < n_instances; i++) {
            all_keys[f * n_instances + i] = X[index[i]][f];
            all_sorted_indices[f * n_instances + i] = index[i]; // Start with unsorted indices
        }
    }

    { // Block forces synchronization
        // Sort index matrix on device in parallel
        sycl::buffer<float, 1> keys_buf(all_keys.data(), sycl::range<1>{all_keys.size()});
        sycl::buffer<int, 1> idx_buf(all_sorted_indices.data(), sycl::range<1>{all_sorted_indices.size()});
        sycl::buffer<int, 1> y_buf(y.data(), sycl::range<1>{y.size()});

        // Resultados dos splits, reduzir em host depois
        sycl::buffer<FeatureSplit, 1> splits_buf(host_best_splits.data(), sycl::range<1>{host_best_splits.size()});

        // Precisa de criar a policy para DPL ter informacao sobre o dispositivo e como rodar
        auto policy = oneapi::dpl::execution::make_device_policy(q);

        for (int f = 0; f < n_features; f++) {
            // Get iterators for this specific feature's slice of the flat buffer// Block forces synchronization
            auto keys_begin = oneapi::dpl::begin(keys_buf) + (f * n_instances);
            auto keys_end = keys_begin + n_instances;
            auto idx_begin = oneapi::dpl::begin(idx_buf) + (f * n_instances);

            // Zip the feature values with their corresponding original indices
            auto zip_begin = oneapi::dpl::make_zip_iterator(keys_begin, idx_begin);
            auto zip_end = oneapi::dpl::make_zip_iterator(keys_end, idx_begin + n_instances);

            // Dispatch asynchronous sort for this feature
            oneapi::dpl::sort(policy, zip_begin, zip_end, [](auto a, auto b) {
                return std::get<0>(a) < std::get<0>(b);
            });
        }

        q.submit([&](sycl::handler& h) {
            sycl::accessor keys_acc{keys_buf, h, sycl::read_only};
            sycl::accessor idx_acc{idx_buf, h, sycl::read_only};
            sycl::accessor y_acc{y_buf, h, sycl::read_only};
            sycl::accessor splits_acc{splits_buf, h, sycl::write_only, sycl::no_init};

            // Uma thread por feature
            h.parallel_for(sycl::range<1>{static_cast<size_t>(n_features)}, [=](sycl::id<1> item) {
                int i_feature = item[0];
                int offset = i_feature * n_instances;

                double local_best_gini = 1.0; // Worst possible Gini
                float local_best_threshold = 0.0f;

                int left_total = 0, right_total = n_instances;
                std::array<int, 5> left_counter = {0};

                int right_counter[n_classes];
                for(int c = 0; c < n_classes; c++){
                    right_counter[c] = class_counter[c];
                }

                for (int k = 0; k < n_instances - 1; k++) {
                    int real_idx = idx_acc[offset + k];
                    int current_class = y_acc[real_idx];

                    left_counter[current_class]++;
                    right_counter[current_class]--;

                    left_total++;
                    right_total--;

                    float current_val = keys_acc[offset + k];
                    float next_val = keys_acc[offset + k + 1];

                    if (current_val == next_val) {
                        continue;
                    }

                    double gini_left = 1.0;
                    for(int c = 0; c < n_classes; c++) {
                        if (left_counter[c] > 0) {
                            double p = (double)left_counter[c] / left_total;
                            gini_left -= (p * p); // realiza -p² para cada classe 
                        }
                    }

                    double gini_right = 1.0;
                    for(int c = 0; c < n_classes; c++) {
                        if (right_counter[c] > 0) {
                            double p = (double)right_counter[c] / right_total;
                            gini_right -= (p * p); 
                        }
                    }

                    double gini = (gini_left * left_total + gini_right * right_total) / n_instances;

                    // Melhor split para essa thread encontrado
                    if (gini < local_best_gini) {
                        local_best_gini = gini;
                        local_best_threshold = (current_val + next_val) / 2.0f;
                    }
                }

                // Anotar no arranjo o melhor split da thread, para reduzir em CPU depois
                splits_acc[i_feature] = {local_best_gini, local_best_threshold, i_feature};
            });
        });
    }

    // Reduce results array to find best split
    for (int f = 0; f < n_features; f++) {
        if (host_best_splits[f].gini < final_gini) {
            final_gini = host_best_splits[f].gini;
            final_feature = host_best_splits[f].feature_idx;
            final_threshold = host_best_splits[f].threshold;
        }
    }

    if (final_feature == -1 || final_gini >= 1.0) {
        return node;
    }

    final_i_left.reserve(n_instances);
    final_i_right.reserve(n_instances);

    split(X, index, final_feature, final_threshold, final_i_left, final_i_right);

    // 8. Chamadas recursivas para construir os filhos
    node->threshold = final_threshold;
    node->feature_index = final_feature;
    node->l = build_tree_rec(X, y, final_i_left, depth+1, max_depth, q);
    node->r = build_tree_rec(X, y, final_i_right, depth+1, max_depth, q);

    return node;
}


Node* build_tree(std::vector<std::vector<float>> &X, std::vector<int> &y, std::vector<int> &index, int depth, int max_depth, bool gpu){
    sycl::device sycl_device;

    try {
        if (gpu) {
            sycl_device = sycl::device{sycl::gpu_selector_v};
        } else {
            sycl_device = sycl::device{sycl::cpu_selector_v};
        }
    } catch (const sycl::exception& e) {
        std::cerr << "Requested hardware not found: " << e.what() << "\n";
        std::cerr << "Falling back to default device.\n";
        sycl_device = sycl::device{sycl::default_selector_v}; 
    }

    sycl::queue q{sycl_device};
    return build_tree_rec(X, y, index, depth, max_depth, q);
}
