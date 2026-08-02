#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>
#include <string>
#include "CART.h"

void split(const std::vector<std::vector<float>> &X, const std::vector<int> &index_sample, int i_feature, double threshold, std::vector<int> &i_left, std::vector<int> &i_right){
    for(int i : index_sample){
        if (X[i][i_feature]>threshold)
            i_right.push_back(i);
        else
            i_left.push_back(i);
    }
    return;
}

std::string class_to_word(bool classe){
    if (classe) return "Malicious";
    else return "Normal";
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

    int classe0 = 0;
    for (int i : index){
        if(y[i] == 0)
            classe0++;
    }
    
    int classe1 = n_instances - classe0;
    node->predicted_class = ( classe0 >= (n_instances/2.0) ) ? 0 : 1;

    if (depth >= max_depth || classe0 == 0 || classe0 == n_instances || n_instances < min_sample_split ) {
        return node;
    } 
    
    double final_gini = std::numeric_limits<double>::max();
    int final_feature = -1;
    float final_threshold = 0.0;
    std::vector<int> final_i_left, final_i_right;


        
    for(int i_feature = 0; i_feature<n_features; i_feature++)
    {
        std::vector<int> sorted_index = index;
        
        // Ordena os índices em O(N log N) usando o valor real na matriz X
        std::sort(sorted_index.begin(), sorted_index.end(), [&](int a, int b) {
            return X[a][i_feature] < X[b][i_feature];
        });

        int left_total = 0, right_total = n_instances;
        int left_c0 = 0, right_c0 = classe0;

        for (int k = 0; k < n_instances - 1; k++) {
            int real_idx = sorted_index[k];

            if (y[real_idx] == 0) {
                left_c0++;
                right_c0--;
            }
            left_total++;
            right_total--;

            if (X[sorted_index[k]][i_feature] == X[sorted_index[k+1]][i_feature]) {
                continue;
            }

            double p0_l = (double)left_c0 / left_total;
            double p1_l = 1.0 - p0_l;
            double gini_left = 1.0 - (p0_l * p0_l + p1_l * p1_l);
            
            double p0_r = (double)right_c0 / right_total;
            double p1_r = 1.0 - p0_r;
            double gini_right = 1.0 - (p0_r * p0_r + p1_r * p1_r);
            
            double gini = (gini_left * left_total + gini_right * right_total) / n_instances;

            if (gini < final_gini) {
                final_gini = gini;
                final_feature = i_feature;
                // Threshold é a média entre o valor atual e o próximo
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

    // 8. Chamadas recursivas para construir os filhos
    node->threshold = final_threshold;
    node->feature_index = final_feature;
    node->l = build_tree(X, y, final_i_left, depth+1, max_depth, gpu);
    node->r = build_tree(X, y, final_i_right, depth+1, max_depth, gpu);
        
    return node;
}


