#pragma once

#include <torch/torch.h>
#include <vector>
#include <unordered_map>

namespace gnn {

class Graph {
public:
    Graph(torch::Tensor node_features, torch::Tensor node_labels, 
          std::vector<std::pair<int, int>> edge_list);
    
    Graph(torch::Tensor node_features, std::vector<std::pair<int, int>> edge_list);

    torch::Tensor getNodeFeatures() const;
    torch::Tensor getNodeLabels() const;
    torch::Tensor getEdgeIndex() const;
    size_t getNumNodes() const;
    size_t getNumEdges() const;
    static Graph createSyntheticGraph(int num_classes);

private:
    torch::Tensor node_features_;  
    torch::Tensor node_labels_;    
    torch::Tensor edge_index_;     
    size_t num_nodes_;
    size_t num_edges_;
};

}