import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, embedding_dim):
        super(Net, self).__init__() 
        self.fc1 = nn.Linear(embedding_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
