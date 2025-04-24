# GraphNet - Graph Neural Network in C++

A C++ implementation of Graph Neural Networks (GNNs) using LibTorch (PyTorch's C++ frontend).

## Architecture

This project implements a graph neural network with the following components:

### Core Components

1. **Graph Class** (`graph.h`)
   - Manages graph structure (nodes and edges)
   - Handles node features and labels
   - Provides utility methods for graph operations
   - Includes a synthetic graph generator for testing

2. **GraphNet Model** (`graphnet.h`)
   - Implements the GNN architecture using message passing
   - Consists of multiple graph convolutional layers
   - Performs feature transformation and aggregation
   - Supports configurable network depth and width

3. **Main Application** (`main.cpp`)
   - Creates a synthetic graph for demonstration
   - Initializes and configures the GNN model
   - Trains the model using stochastic gradient descent
   - Evaluates model performance and saves the trained model

### Network Architecture

The GNN uses a message-passing architecture with:
- Node feature transformation via linear layers
- Neighborhood aggregation
- Multi-layer design with configurable hidden dimensions
- Classification output layer

## Requirements

- C++14 or higher
- LibTorch (PyTorch C++ API)
- CMake 3.0+
- CUDA (optional, for GPU acceleration)

## Building the Project

1. **Install LibTorch**:
   Download the appropriate version from https://pytorch.org/get-started/locally/

2. **Configure CMake**:
   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
   ```

3. **Build the project**:
   ```bash
   cmake --build .
   ```

## Running the Application

Execute the compiled binary:

```bash
./graphnet
```

The program will:
1. Create a synthetic graph with labeled nodes
2. Initialize a Graph Neural Network model
3. Train the model for 100 epochs
4. Output training progress, predictions, and final accuracy
5. Save the model to "graphnet_model.pt"

## Customization

You can modify key parameters in `main.cpp`:
- `num_classes`: Number of classification categories
- `hidden_features`: Dimensionality of hidden layers
- `num_layers`: Depth of the GNN
- `num_epochs`: Training duration
- Learning rate in the optimizer instantiation

## GPU Acceleration

The code automatically detects and uses CUDA if available. No additional configuration is needed.

## Extending the Project

- Add different graph convolutional layers
- Implement more sophisticated aggregation methods
- Support loading graph data from files
- Add validation/test splits
- Implement early stopping
