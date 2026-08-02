#ifndef CART_H
#define CART_H

#include <vector>
#include <string>

struct Node {
    int feature_index;
    float threshold;
    int predicted_class;
    Node* l;
    Node* r;

    Node() : feature_index(-1), threshold(0.0f), predicted_class(-1), l(nullptr), r(nullptr) {}

    ~Node(){
        delete l;
        delete r;
    }

    bool is_leaf() {
        return (l == nullptr && r == nullptr);
    }
};

Node* build_tree(std::vector<std::vector<float>> &X, std::vector<int> &y, std::vector<int> &index, int depth, int max_depth, bool gpu);
void print_tree(Node* node, std::vector<std::string> &headers, std::string prefix = "", bool is_left = false, bool is_root = true);

#endif
