import torch
from config import CFG
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, data, config: CFG):
        super().__init__()
        self.types = data.logicalLabel.shape[1]
        self.conv1 = SAGEConv(data.num_node_features, self.types, normalize=True).to(
            config.device
        )
        self.P = data.P
        self.alpha = config.alpha
        self.label = data.logicalLabel.to(config.device)
        self.label = torch.cat(
            (self.label, torch.ones(self.types, self.types).to(config.device)), dim=0
        ).to(config.device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)  # [num_nodes,self.types]
        x1 = torch.sigmoid(x1)
        for _ in range(50):
            x1 = (
                self.alpha * self.P @ x1 + (1 - self.alpha) * self.label
            )  # [num_nodes,features/2]
        return x1[: data.true_nodes]


class LinearModel(torch.nn.Module):
    def __init__(self, trainFeature, trainLabel):
        super().__init__()
        self.l1 = torch.nn.Linear(trainFeature.shape[1], trainLabel.shape[1])

    def forward(self, trainFeature):
        modProb = torch.sigmoid(self.l1(trainFeature))
        return modProb
