#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include "util.h"

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current_field;
    bool in_quotes = false;
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            fields.push_back(current_field);
            current_field.clear();
        } else {
            current_field += c;
        }
    }
    fields.push_back(current_field);
    return fields;
}

// is_train should be true for train.csv (includes Survived), false for test.csv
Dataset read_csv_and_header(const std::string& filename, bool is_train, std::vector<std::string> &headers) {
    Dataset dataset;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    // Read header
    if (std::getline(file, line)) {
        if (!line.empty() && line.find_first_not_of("\r\n") != std::string::npos) {
            headers = parse_csv_line(line);
        }
    }

    while (std::getline(file, line)) {
        if (line.empty() || line.find_first_not_of("\r\n") == std::string::npos) continue;

        std::vector<std::string> fields = parse_csv_line(line);
        Packet p;
        int field_idx = 0;
        
        try {
            p.arbitration_id = std::stof(fields[field_idx++]);
            p.dlc = std::stof(fields[field_idx++]);
            p.data_0 = std::stof(fields[field_idx++]);
            p.data_1 = std::stof(fields[field_idx++]);
            p.data_2 = std::stof(fields[field_idx++]);
            p.data_3 = std::stof(fields[field_idx++]);
            p.data_4 = std::stof(fields[field_idx++]);
            p.data_5 = std::stof(fields[field_idx++]);
            p.data_6 = std::stof(fields[field_idx++]);
            p.data_7 = std::stof(fields[field_idx++]);
            p.road_type_highway = std::stof(fields[field_idx++]);
            p.road_type_rural = std::stof(fields[field_idx++]);
            p.road_type_urban = std::stof(fields[field_idx++]);
            p.climate_dry = std::stof(fields[field_idx++]);
            p.climate_rain_fog = std::stof(fields[field_idx++]);
            p.climate_snow = std::stof(fields[field_idx++]);
            p.flag_binaria = std::stof(fields[field_idx++]);
            dataset.rows.push_back(p);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing line: " << line << " -> " << e.what() << std::endl;
        }
    }
    return dataset;
}


void populate_X_y_from_rows(Dataset& dataset) {
    dataset.X.clear();
    dataset.y.clear();
    
    size_t num_rows = dataset.rows.size();
    dataset.X.reserve(num_rows);
    dataset.y.reserve(num_rows);

    for (const auto& p : dataset.rows) {
        std::vector<float> features = {
            p.arbitration_id,
            p.dlc,
            p.data_0,
            p.data_1,
            p.data_2,
            p.data_3,
            p.data_4,
            p.data_5,
            p.data_6,
            p.data_7,
            static_cast<float>(p.road_type_highway),
            static_cast<float>(p.road_type_rural),
            static_cast<float>(p.road_type_urban),
            static_cast<float>(p.climate_dry),
            static_cast<float>(p.climate_rain_fog),
            static_cast<float>(p.climate_snow)
        };
        dataset.X.push_back(features);
        dataset.y.push_back(p.flag_binaria);
    }
}

void print_dataset_summary(const Dataset& ds, const std::string& name) {
    std::cout << "\n=== Resumo do Dataset: " << name << " ===" << std::endl;
    
    std::cout << "Total de pacotes (rows): " << ds.rows.size() << std::endl;
    
    if (!ds.X.empty()) {
        std::cout << "Total de amostras processadas em X: " << ds.X.size() << std::endl;
        std::cout << "Features por amostra (colunas): " << ds.X[0].size() << std::endl;
    } else {
        std::cout << "Matriz X: Vazia (lembre-se de chamar populate_X_y_from_rows)" << std::endl;
    }
    
    if (!ds.y.empty()) {
        // Cria um mapa dinâmico para contar qualquer quantidade de classes
        std::map<int, int> class_counts;
        
        for (int label : ds.y) {
            class_counts[label]++;
        }
        
        std::cout << "Tamanho do vetor y: " << ds.y.size() << std::endl;
        std::cout << "Distribuição de classes (y):" << std::endl;
        
        // Itera sobre o mapa e imprime a contagem de cada classe encontrada
        for (const auto& par : class_counts) {
            std::cout << "  -> Classe " << par.first << ": " << par.second << " amostras" << std::endl;
        }
        
    } else {
        std::cout << "Vetor y: Vazio" << std::endl;
    }
    
    std::cout << "=======================================\n" << std::endl;
}


