import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # Define embedding layers for each discrete input
        self.embedding1 = nn.Embedding(32, 8)  # 32 discrete values to 8-dimensional vector
        self.embedding2 = nn.Embedding(11, 4)  # 11 discrete values to 4-dimensional vector
        self.embedding3 = nn.Embedding(2, 2)   # 2 discrete values to 2-dimensional vector
        
        # Calculate the total input size for the first fully connected layer
        total_input_size = 8 + 4 + 2
        
        # Define fully connected layers
        self.fc1 = nn.Linear(total_input_size, 16)
        self.fc2 = nn.Linear(16, 4)
    
    def forward(self, x):
        # Debug: Print the shape and contents of the input tensor
        print(f"Input tensor shape: {x.shape}")
        print(f"Input tensor contents: {x}")
        
        # Ensure input is of the correct type
        x = x.long()
        
        # Extract individual elements from the input tensor
        x1 = x[:, 0].clamp(0, 31)  # Discrete(32)
        x2 = x[:, 1].clamp(0, 10)  # Discrete(11)
        x3 = x[:, 2].clamp(0, 1)   # Discrete(2)
        
        # Debug: Print the indices after clamping
        print(f"Clamped x1: {x1}")
        print(f"Clamped x2: {x2}")
        print(f"Clamped x3: {x3}")
        
        # Pass each input through its corresponding embedding layer
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        x3 = self.embedding3(x3)
        
        # Debug: Print the embeddings
        print(f"Embedding x1: {x1}")
        print(f"Embedding x2: {x2}")
        print(f"Embedding x3: {x3}")
        
        # Concatenate the embeddings
        x = torch.cat((x1, x2, x3), dim=-1)
        
        # Pass the concatenated vector through the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x