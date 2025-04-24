#include "graphnet.h"

namespace gnn {

GraphNetLayer::GraphNetLayer(int64_t in_features, int64_t out_features) {
    node_transform_ = register_module("node_transform", 
                                      torch::nn::Linear(in_features, out_features));
    message_transform_ = register_module("message_transform", 
                                         torch::nn::Linear(in_features, out_features));
    update_transform_ = register_module("update_transform", 
                                        torch::nn::Linear(2 * out_features, out_features));
}

torch::Tensor GraphNetLayer::forward(torch::Tensor node_features, torch::Tensor edge_index) {
    auto num_nodes = node_features.size(0);
    auto node_transformed = node_transform_->forward(node_features);
    auto source_nodes = edge_index.select(0, 0);
    auto target_nodes = edge_index.select(0, 1);
    auto source_features = node_features.index_select(0, source_nodes);
    auto messages = message_transform_->forward(source_features);
    auto aggregated_messages = torch::zeros({num_nodes, messages.size(1)}, 
                                            messages.options());
    for (int64_t i = 0; i < source_nodes.size(0); ++i) {
        int64_t target = target_nodes[i].item().toLong();
        for (int64_t j = 0; j < messages.size(1); ++j) {
            aggregated_messages[target][j] += messages[i][j];
        }
    }
    std::vector<torch::Tensor> to_cat = {node_transformed, aggregated_messages};
    auto combined = torch::cat(to_cat, 1);
    auto updated_features = update_transform_->forward(combined);
    return torch::relu(updated_features);
}

GraphNet::GraphNet(int64_t in_features, int64_t hidden_features, 
                   int64_t num_classes, int64_t num_layers) 
    : hidden_features_(hidden_features) {
    layers_.push_back(torch::nn::AnyModule(GraphNetLayer(in_features, hidden_features)));
    for (int64_t i = 1; i < num_layers; ++i) {
        layers_.push_back(torch::nn::AnyModule(GraphNetLayer(hidden_features, hidden_features)));
    }
    classifier_ = register_module("classifier", 
                                  torch::nn::Linear(hidden_features, num_classes));
}

torch::Tensor GraphNet::forward(torch::Tensor node_features, torch::Tensor edge_index) {
    auto x = node_features;
    for (size_t i = 0; i < layers_.size(); ++i) {
        
        torch::Tensor x_copy = x.clone();
        torch::Tensor edge_index_copy = edge_index.clone();
        x = layers_[i].forward<torch::Tensor, torch::Tensor, torch::Tensor>(std::move(x_copy), std::move(edge_index_copy));
    }
    return classifier_->forward(x);
}

torch::Tensor GraphNet::forward(const Graph& graph) {
    return forward(graph.getNodeFeatures(), graph.getEdgeIndex());
}

}