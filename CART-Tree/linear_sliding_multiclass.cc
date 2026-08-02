#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>
#include <string>
#include "util.h"
#include "CART.h"

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
 
void split(const std::vector<std::vector<float>> &X, const std::vector<int> &index_sample, int i_feature, double threshold, std::vector<int> &i_left, std::vector<int> &i_right){
    for(int i : index_sample){
        if (X[i][i_feature]>threshold)
            i_right.push_back(i);
        else
            i_left.push_back(i);
    }
    return;
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

Node* build_tree(std::vector<std::vector<float>> &X, std::vector<int> &y, std::vector<int> &index, int depth, int max_depth, bool gpu){
    
    Node* node = new Node();

    int n_features = X[0].size();
    int n_instances = index.size();
    int min_sample_split = 50;
    const int n_classes = 5;

    int class_counter[5] = {0};
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


        
    for(int i_feature = 0; i_feature<n_features; i_feature++)
    {
        std::vector<int> sorted_index = index;
        int left_total = 0, right_total = n_instances;
        
        // Ordena os índices em O(N log N) usando o valor real na matriz X
        std::sort(sorted_index.begin(), sorted_index.end(), [&](int a, int b) {
            return X[a][i_feature] < X[b][i_feature];
        });

        int left_counter[n_classes] = {0};

        int right_counter[n_classes];
        for(int c = 0; c < n_classes; c++){
            right_counter[c] = class_counter[c];
        }

        for (int k = 0; k < n_instances - 1; k++) {
            int real_idx = sorted_index[k];
            int current_class = y[real_idx];

            left_counter[current_class]++;
            right_counter[current_class]--;
            
            left_total++;
            right_total--;

            if (X[sorted_index[k]][i_feature] == X[sorted_index[k+1]][i_feature]) {
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

            if (gini < final_gini) {
                final_gini = gini;
                final_feature = i_feature;
                final_threshold = (X[sorted_index[k]][i_feature] + X[sorted_index[k+1]][i_feature]) / 2.0f;
            }
        }
    }    

    if (final_feature == -1) {
        return node;
    }

    final_i_left.reserve(n_instances);
    final_i_right.reserve(n_instances);
    
    split(X, index, final_feature, final_threshold, final_i_left, final_i_right);

    node->threshold = final_threshold;
    node->feature_index = final_feature;
    node->l = build_tree(X, y, final_i_left, depth+1, max_depth, gpu);
    node->r = build_tree(X, y, final_i_right, depth+1, max_depth, gpu);
        
    return node;
}


