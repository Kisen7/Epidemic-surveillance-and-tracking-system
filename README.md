Through the combined model of LSTM and GNN, multi-step disease transmission path prediction can be realized. At each time step, LSTM predicted the current infection trend and lesion area, and GNN dynamically updated the infection status of nodes according to the structure of the contact network and the prediction results of LSTM. After each time step, GNN updates the transmission path using graph convolution operations, simulating how the disease spreads from one individual to neighboring individuals.
![image](https://github.com/user-attachments/assets/135a7277-8f42-4855-98b7-bd523e5025ee) 
![image](https://github.com/user-attachments/assets/86fe19aa-eccd-482a-aaa4-303573dc9fc3)
![image](https://github.com/user-attachments/assets/bd4593fc-5007-4bfd-8f1d-f95482394fd2) 
![image](https://github.com/user-attachments/assets/2cfba58b-347b-430c-b5cc-90960b68e5df)
![image](https://github.com/user-attachments/assets/8140c583-b0b3-4eaa-8183-df05e9b13a4a) 
![image](https://github.com/user-attachments/assets/3cb09a5d-98c1-4dc3-bc57-85f66cfcfd95)
