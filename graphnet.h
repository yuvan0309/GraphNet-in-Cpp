#pragma once

#include <torch/torch.h>
#include "graph.h"

namespace gnn {

class GraphNetLayer : public torch::nn::Module {
public:
    GraphNetLayer(int64_t in_features, int64_t out_features);
    torch::Tensor forward(torch::Tensor node_features, torch::Tensor edge_index);

private:
    torch::nn::Linear node_transform_{nullptr};
    torch::nn::Linear message_transform_{nullptr};
    torch::nn::Linear update_transform_{nullptr};
};

class GraphNet : public torch::nn::Module {
public:
    GraphNet(int64_t in_features, int64_t hidden_features, int64_t num_classes, int64_t num_layers = 2);
    torch::Tensor forward(torch::Tensor node_features, torch::Tensor edge_index);
    torch::Tensor forward(const Graph& graph);

private:
    std::vector<torch::nn::AnyModule> layers_;
    torch::nn::Linear classifier_{nullptr};
    int64_t hidden_features_;
};

}