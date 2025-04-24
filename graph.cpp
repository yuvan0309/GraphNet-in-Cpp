#include "graph.h"
#include <iostream>

namespace gnn {

Graph::Graph(torch::Tensor node_features, torch::Tensor node_labels, 
             std::vector<std::pair<int, int>> edge_list) 
    : node_features_(node_features), 
      node_labels_(node_labels),
      num_nodes_(node_features.size(0)),
      num_edges_(edge_list.size()) {
    
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    edge_index_ = torch::zeros({2, static_cast<int64_t>(num_edges_)}, options);
    
    for (size_t i = 0; i < edge_list.size(); ++i) {
        edge_index_[0][i] = edge_list[i].first;
        edge_index_[1][i] = edge_list[i].second;
    }
}

Graph::Graph(torch::Tensor node_features, std::vector<std::pair<int, int>> edge_list) 
    : node_features_(node_features),
      node_labels_(torch::zeros({node_features.size(0)}, torch::kInt64)),
      num_nodes_(node_features.size(0)),
      num_edges_(edge_list.size()) {
    
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    edge_index_ = torch::zeros({2, static_cast<int64_t>(num_edges_)}, options);
    
    for (size_t i = 0; i < edge_list.size(); ++i) {
        edge_index_[0][i] = edge_list[i].first;
        edge_index_[1][i] = edge_list[i].second;
    }
}

torch::Tensor Graph::getNodeFeatures() const {
    return node_features_;
}

torch::Tensor Graph::getNodeLabels() const {
    return node_labels_;
}

torch::Tensor Graph::getEdgeIndex() const {
    return edge_index_;
}

size_t Graph::getNumNodes() const {
    return num_nodes_;
}

size_t Graph::getNumEdges() const {
    return num_edges_;
}

Graph Graph::createSyntheticGraph(int num_classes) {
    
    int num_nodes = 5;
    int feature_dim = num_classes + 1;  
    
    auto node_features = torch::randn({num_nodes, feature_dim});
    
    torch::Tensor node_labels = torch::zeros({num_nodes}, torch::kInt64);
    
    node_labels[1] = 1 % num_classes;
    node_labels[3] = 1 % num_classes;
    
    std::vector<std::pair<int, int>> edge_list = {
        {0, 1}, {1, 0},  
        {1, 2}, {2, 1},  
        {2, 3}, {3, 2},  
        {3, 4}           
    };
    
    return Graph(node_features, node_labels, edge_list);
}

}