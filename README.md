Through the combined model of LSTM and GNN, multi-step disease transmission path prediction can be realized. At each time step, LSTM predicted the current infection trend and lesion area, and GNN dynamically updated the infection status of nodes according to the structure of the contact network and the prediction results of LSTM. After each time step, GNN updates the transmission path using graph convolution operations, simulating how the disease spreads from one individual to neighboring individuals.