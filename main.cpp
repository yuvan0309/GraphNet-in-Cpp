#include <torch/torch.h>
#include <iostream>
#include "graph.h"
#include "graphnet.h"

int main() {
    std::cout << "Graph Neural Network Example with LibTorch" << std::endl;
    
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    
    int num_classes = 2;  
    auto graph = gnn::Graph::createSyntheticGraph(num_classes);
    
    std::cout << "Created graph with " << graph.getNumNodes() << " nodes and " 
              << graph.getNumEdges() << " edges." << std::endl;
    
    int64_t in_features = graph.getNodeFeatures().size(1);
    int64_t hidden_features = 16;
    int64_t num_layers = 2;
    
    auto model = std::make_shared<gnn::GraphNet>(in_features, hidden_features, num_classes, num_layers);
    model->to(device);
    
    std::cout << "Model structure:" << std::endl << *model << std::endl;
    
    torch::optim::SGD optimizer(model->parameters(), /*lr=*/0.01);
    
    int num_epochs = 100;
    
    auto node_features = graph.getNodeFeatures().to(device);
    auto edge_index = graph.getEdgeIndex().to(device);
    auto labels = graph.getNodeLabels().to(device);
    
    std::cout << "\nStarting training..." << std::endl;
    
    model->train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        auto outputs = model->forward(node_features, edge_index);
        
        auto loss = torch::mse_loss(outputs, torch::one_hot(labels, num_classes).to(torch::kFloat));
        
        loss.backward();
        optimizer.step();
        
        auto predicted = outputs.argmax(1);
        float accuracy_val = static_cast<float>((predicted == labels).sum().item<int64_t>()) / labels.size(0);
        
        if ((epoch + 1) % 1 == 0) {
            float loss_val = loss.item<float>();
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                      << ", Loss: " << loss_val
                      << ", Accuracy: " << accuracy_val << std::endl;
        }
    }
    
    model->eval();
    auto outputs = model->forward(node_features, edge_index);
    auto predicted = outputs.argmax(1);
    float accuracy = static_cast<float>((predicted == labels).sum().item<int64_t>()) / labels.size(0);
    
    std::cout << "\nFinal node predictions:" << std::endl;
    for (int64_t i = 0; i < predicted.size(0); ++i) {
        std::cout << "Node " << i << ": true=" << labels[i].item<int64_t>() 
                  << ", predicted=" << predicted[i].item<int64_t>() << std::endl;
    }
    
    std::cout << "\nFinal accuracy: " << accuracy << std::endl;
    
    torch::save(model, "graphnet_model.pt");
    std::cout << "Model saved to graphnet_model.pt" << std::endl;
    
    return 0;
}