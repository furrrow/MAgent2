import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.conv1 = GCNConv(n_features, 16)
        self.conv2 = GCNConv(16, n_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

n_agents = 3
edge_list = []
for i in range(n_agents):
    for j in range(n_agents):
        if i != j:
            edge_list.append(torch.tensor([i, j]))
edge_index = torch.stack(edge_list)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
y = torch.tensor([0, 1, 2], dtype=torch.long)
n_classes = 3
device = torch.device('cpu')
data = Data(x=x, edge_index=edge_index.t().contiguous())
model = GCN(data.num_node_features, n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)  # out.shape
    loss = F.nll_loss(out, y)
    loss.backward()
    optimizer.step()