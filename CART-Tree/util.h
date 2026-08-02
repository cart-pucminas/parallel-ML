#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>

struct Packet {
    float timestamp;
    float arbitration_id;
    float dlc;
    float data_0;
    float data_1;
    float data_2;
    float data_3;
    float data_4;
    float data_5;
    float data_6;
    float data_7;
    int road_type_highway;
    int road_type_rural;
    int road_type_urban;
    int climate_dry ;
    int climate_rain_fog; 
    int climate_snow;
    int flag_binaria;
};

struct Metrics {
    float precision;
    float recall;
    float f1;
};

struct Dataset {
    std::vector<Packet> rows;
    std::vector<std::vector<float>> X; // Normalized feature matrix
    std::vector<int> y; // Labels
};

std::vector<std::string> parse_csv_line(const std::string& line);

Dataset read_csv_and_header(const std::string& filename, bool is_train, std::vector<std::string> &header);

void print_dataset_summary(const Dataset& ds, const std::string& name);

void populate_X_y_from_rows(Dataset& dataset);

#endif // UTIL_H
