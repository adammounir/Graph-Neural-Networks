import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# gat-based model for ppi node classification
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # 3 gat layers with skip connections and layer norm
        self.conv1 = GATConv(50, 256, heads=4)
        self.ln1 = nn.LayerNorm(256 * 4)
        self.conv2 = GATConv(256 * 4, 256, heads=4)
        self.ln2 = nn.LayerNorm(256 * 4)
        self.conv3 = GATConv(256 * 4, 121, heads=6, concat=False)

        # linear layers for skip connections
        self.skip1 = nn.Linear(50, 256 * 4)
        self.skip2 = nn.Linear(256 * 4, 256 * 4)

    def forward(self, x, edge_index):
        x = F.elu(self.ln1(self.conv1(x, edge_index) + self.skip1(x)))
        x = F.elu(self.ln2(self.conv2(x, edge_index) + self.skip2(x)))
        x = self.conv3(x, edge_index)
        return x


### This is the part we will run in the inference to grade your model
## Load the model
model = StudentModel()  # !  Important : No argument
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()
print("Model loaded successfully")
