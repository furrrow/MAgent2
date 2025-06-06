import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from tqdm import tqdm
"""
gnn_primer.py, based on
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
also checkout
https://pytorch-geometric.readthedocs.io/en/latest/tutorial/gnn_design.html
"""



dataset = Planetoid(root='/tmp/Cora', name='Cora')
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    out = model(data)  # [2708, 7]
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # masked x: [140, 7],  # masked y:[140]
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
# >>> Accuracy: 0.8150