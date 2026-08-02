#include <fstream>
#include <iostream>
#include "util.h"
#include "CART.h"
#include <numeric>
#include <ostream>
#include <string>
#include <vector>
#include <chrono>

/*
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
*/
 
int classify(Node* node, std::vector<float>& test_data){
    int predicted_class = 0;
    if(node->is_leaf()){
        predicted_class = node->predicted_class;
    }
    else{
        if(test_data[node->feature_index] <= node->threshold){
           return classify(node->l, test_data);
        }
        else if(test_data[node->feature_index] > node->threshold){
            return classify(node->r, test_data);
        }
    }
    return predicted_class;
}

float precision(int true_positive, int false_positive){
    return ((float)true_positive/(float)(true_positive+false_positive));
}

float recall(int true_positive, int false_negative){
    return ((float)true_positive/(float)(true_positive+false_negative));
}

float f1(float precision, float recall){
    return 2.0f * precision*recall/(precision+recall);
}

Metrics* calculate_metrics_bi(Node* tree_root, Dataset& test_dataset){
    int predicted_class = 0;
    int current_expected_class = 0;

    int true_positive=0; // positive -> malicioso(flag = 1)
    int true_negative=0; // negative -> normal(flag = 0)
    int false_positive=0;
    int false_negative=0;

    int total_entries=0;

    Metrics* metrics = new Metrics();

    for(int i = 0; i<test_dataset.X.size(); i++){
        current_expected_class = test_dataset.y[i];
        predicted_class = classify(tree_root, test_dataset.X[i]);
        if(current_expected_class == predicted_class){ 
            if(current_expected_class == 0){
                true_negative++;
            }
            else{
                true_positive++;
            }
        }
        else{
            if(current_expected_class == 0){
                false_negative++;
            }
            else{
                false_positive++;
            }
        }
    }

    metrics->precision = precision(true_positive, false_positive);
    metrics->recall = recall(true_positive, false_negative);
    metrics->f1 = f1(metrics->precision, metrics->recall);

    std::cout << "Precisão: " << metrics->precision*100 << "%" << std::endl; 
    std::cout << "Recall: " << metrics->recall*100 << "%"  << std::endl; 
    std::cout << "F1 Score: " << metrics->f1*100 << "%" << std::endl; 

    return metrics;
}

Metrics* calculate_metrics_multi(Node* tree_root, Dataset& test_dataset){
    const int n_classes = 5;

    std::vector<std::vector<int>> confusion_matrix(n_classes, std::vector<int>(n_classes, 0));

    int total_entries = test_dataset.X.size();
    int correct = 0;

    for(int i = 0; i < total_entries; i++){
        int current_expected_class = test_dataset.y[i];
        int predicted_class = classify(tree_root, test_dataset.X[i]);

        confusion_matrix[current_expected_class][predicted_class]++;

        if(current_expected_class == predicted_class) {
            correct++;
        }
    }

    double global_precision = 0.0;
    double global_recall = 0.0;
    double global_f1 = 0.0;

    std::cout << " Lembrete: \n" << std::endl;
    std::cout << " Precisão: Disse que era, e era \n" << std::endl;
    std::cout << " Recall: Disse que não era, e era\n" << std::endl;
    std::cout << " F1: Média harmônica dos dois\n" << std::endl;

    std::cout << "\n== Metricas por Classe ==" << std::endl;

    for (int c = 0; c < n_classes; c++) {
        int tp = confusion_matrix[c][c]; //total de positivos (ex: 0/0 -> esperava normal e é normal)
        int fp = 0; //falso positivo: falou que é e não é
        int fn = 0; //falso negativo: falou que não é, e é

        for (int i = 0; i < n_classes; i++) {
            if (i != c) {
                fp += confusion_matrix[i][c]; // falsos positivos acumuladros
                fn += confusion_matrix[c][i]; // falsos negativos acumulados
            }
        }

        double p = (double)tp / (tp + fp);
        double r = (double)tp / (tp + fn);
        double f = 2.0 * p * r / (p + r);

        std::cout << "Classe " << c << " (" << c << "):\t"
                  << "Precisao: " << p * 100 << "%\t"
                  << "Recall: " << r * 100 << "%\t"
                  << "F1: " << f * 100 << "%" << std::endl;

        if ((tp + fp + fn) > 0) {
            global_precision += p;
            global_recall += r;
            global_f1 += f;
        }
    }

    Metrics* metrics = new Metrics();
        metrics->precision = global_precision / n_classes;
        metrics->recall = global_recall / n_classes;
        metrics->f1 = global_f1 / n_classes;
 
    double accuracy = (double)correct / total_entries;

    std::cout << "\n--- Metricas Globais---" << std::endl;
    std::cout << "Acuracia Global: " << accuracy * 100 << "%" << std::endl;
    std::cout << "Precisao Global: " << metrics->precision * 100 << "%" << std::endl;
    std::cout << "Recall Global: " << metrics->recall * 100 << "%"  << std::endl;
    std::cout << "F1 Score Global: " << metrics->f1 * 100 << "%" << std::endl;
    std::cout << "----------------------------------------\n" << std::endl;

    return metrics;
}

int main (int argc, char *argv[]) {

    if (argc < 6) {
        std::cerr << "Plese provide train, test and result paths and nThreadSetting and multiSetting in that order\n";
        return 1;
    }

    int nThreadSetting = std::stoi(argv[4]);
    bool gpu = nThreadSetting == 0;

    bool multi = std::stoi(argv[5]) == 1;

    std::vector<std::string> headers;
    std::vector<std::string> dummy_headers;

    Dataset train_set = read_csv_and_header(argv[1], true, headers);
    populate_X_y_from_rows(train_set);

    Dataset test_set = read_csv_and_header(argv[2], false, dummy_headers);
    populate_X_y_from_rows(test_set);

    print_dataset_summary(train_set, "train_set");

    print_dataset_summary(train_set, "Train Set");
    print_dataset_summary(test_set, "Test Set");

    std::vector<int> initial_indices(train_set.X.size());
    std::iota(initial_indices.begin(), initial_indices.end(), 0);

    int max_depth = 5;
    
    auto start = std::chrono::high_resolution_clock::now();
    Node* root = build_tree(train_set.X, train_set.y, initial_indices, 0, max_depth, gpu);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << "Treinamento concluido com sucesso!" << std::endl;

    std::chrono::duration<double> total_time = end - start;
    std::cout << "Tempo de execucao de " << argv[0] << " -> " << total_time.count() << "\n";

    std::cout << "--- Estrutura da Arvore de Decisao ---" << std::endl;
    print_tree(root, headers);
    std::cout << "--------------------------------------\n" << std::endl;

    auto metrics = multi ? calculate_metrics_multi(root, test_set) : calculate_metrics_bi(root, test_set);
    // Metrics* metrics = calculate_metrics(root, test_set);

    std::ofstream results;
    results.open(argv[3], std::ios_base::app);
    results << argv[0] << ", " 
        << nThreadSetting << ", " 
        << multi << ", "
        << total_time.count() << ", " 
        << metrics->f1 << ", " 
        << metrics->precision << ", " 
        << metrics->recall << "\n";
    
    delete root;
    return 0;
}
