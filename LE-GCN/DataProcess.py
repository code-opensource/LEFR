from scipy.io import loadmat
import scipy.linalg as lg
import torch
from torch_geometric.data import Data
from config import CFG


class computeD(torch.nn.Module):
    def __init__(self, data, config: CFG):
        super().__init__()
        self.W = data.W
        self.D = self.compute_D(self.W).cuda()
        data.D = self.D
        self.P = (self.D) @ (self.W) @ (self.D).cuda()
        data.P = self.P
        self.label = data.logicalLabel.cuda()
        self.label = torch.cat(
            (
                self.label,
                torch.ones(
                    data.logicalLabel.shape[1], data.logicalLabel.shape[1]
                ).cuda(),
            ),
            dim=0,
        ).cuda()
        self.config = config

    def forward(self, data):
        pass

    def compute_D(self, tensor):
        length = tensor.shape[0]
        m = torch.zeros(length, length).to(self.config.device)
        for i in range(length):
            for j in range(length):
                m[i][i] += tensor[i][j]
        for i in range(length):
            m[i][i] = 1 / torch.sqrt(m[i][i])
        return m


class GraphPreprocessor:
    def __init__(self, path, config: CFG):
        self.path = path
        self.types = 0
        self.nodes = 0
        self.features = 0
        self.threshold = config.threshold
        self.edge_index = []
        self.feature = 0
        self.W = torch.Tensor()
        self.data = 0
        self.config = config

    def process(self):
        m = loadmat(self.path)
        features = torch.Tensor(m["features"])
        labelDistribution = torch.Tensor(m["labelDistribution"])
        logicalLabel = torch.Tensor(m["logicalLabel"])
        nodes, features = features.shape
        types = logicalLabel.shape[1]
        feature = torch.cat((features, torch.randn(types, features)), 0)
        edge_index = []
        W = self.compute_W(feature)
        for i in range(nodes):
            for j in range(i + 1, nodes):
                if W[i][j] > self.threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        for i in range(nodes):
            for j in range(types):
                if logicalLabel[i][j]:
                    edge_index.append([i, nodes + j])
                    edge_index.append([nodes + j, i])
        edge_index = torch.Tensor(edge_index).reshape(2, -1).type(torch.long)
        data = Data(x=feature, edge_index=edge_index)
        data.labelDistribution = labelDistribution
        data.features = features
        data.logicalLabel = logicalLabel
        data.true_nodes = logicalLabel.shape[0]
        data.W = W
        return data

    def compute_W(self, tensor):
        samples = tensor.shape[0]
        m = torch.zeros(samples, samples).to(self.config.device)
        for i in range(samples):
            tensor_tmp = tensor - tensor[i].repeat(samples, 1)
            m[i] = torch.exp(-torch.norm(tensor_tmp, 2, 1) / 2)
        return m


class KNNPreprocessor(object):
    def __init__(self, path, data, config: CFG):
        self.data = data
        self.path = path
        self.process()
        self.config = config

    def process(self):
        m = loadmat(self.path)
        features = torch.Tensor(m["features"])
        # features = model(self.data).detach().cpu()
        labelDistribution = torch.Tensor(m["labelDistribution"])
        self.labelDistribution = labelDistribution
        logicalLabel = torch.Tensor(m["logicalLabel"])
        K = torch.tensor(self.config.K)
        d, n = features.shape
        X = features
        data = X.T
        d = torch.tensor(self.config.d)
        m, N = data.shape
        if not K.shape:
            K = K.repeat(1, N).reshape(-1)
        NI = list()
        if m > N:
            a = sum(data * data)
            dist2 = (
                torch.sqrt(a.T.reshape(-1, 1).repeat(1, N))
                + a.reshape(1, -1).repeat(N, 1)
                - 2 * (data.T @ data)
            )
            for i in range(N):
                dist_sort, J = torch.sort(dist2[:][i])
                Ii = J[: K[i]]
                NI.append(Ii)
        else:
            for i in range(N):
                x = data[:, i]
                dist2 = torch.sum((data - x.reshape(-1, 1).repeat(1, N)).pow(2), 0)
                dist_sort, J = torch.sort(dist2)
                Ii = J[: K[i]]
                NI.append(Ii)
        # step 1: local information
        BI = []
        for i in range(N):
            Ii = NI[i]
            ki = K[i]
            Xi = data[:, Ii] - torch.mean(data[:, Ii]).repeat(1, ki)
            W = Xi.T @ Xi
            W = (W + W.T) / 2
            Vi, Si = lg.schur(W.detach())
            s, Ji = torch.sort(-torch.diag(torch.tensor(Si)))
            Vi = Vi[:, Ji[:d]]
            Gi = (
                torch.cat(
                    (
                        torch.tensor(1 / torch.sqrt(ki).clone().detach())
                        .repeat(ki, 1)
                        .clone()
                        .detach(),
                        torch.tensor(Vi).clone().detach(),
                    ),
                    dim=1,
                )
                .clone()
                .detach()
            )
            BI.append(torch.eye(ki) - Gi @ Gi.T)
        B = torch.eye(N)
        for i in range(N):
            Ii = NI[i]
            # for i in range(Ii.shape[0]):
            #       B[Ii[i]][Ii[i]]=B[Ii[i]][Ii[i]]
            B[Ii][:, Ii] = B[Ii][:, Ii] + BI[i]
            B[i][i] = B[i][i] - 1
        self.B = B
        ker = "rbf"
        par = 1 * torch.mean(torch.pdist(features))
        H = self.kernelmatrix(ker, par, features, features)
        UnitMatrix = torch.ones(features.shape[0], 1)
        self.trainFeature = torch.cat((H, UnitMatrix), 1)
        self.trainLabel = logicalLabel
        # item = torch.rand((trainFeature.shape[1],trainLabel.shape[1]))
        # w,fval = self.fminlbfgsGLLE(self.LEbfgsProcess,item)
        # numerical = trainFeature*w

    def kernelmatrix(self, ker, parameter, testX, trainX):
        if ker == "rbf":
            n1sq = torch.sum(testX.T.pow(2), 0)
            n1 = testX.T.shape[1]
            n2sq = torch.sum(trainX.T.pow(2), 0)
            n2 = trainX.T.shape[1]
            D = (
                (torch.ones(n2, 1) * n1sq).T
                + torch.ones(n1, 1) * n2sq
                - 2 * testX @ trainX.T
            )
            K = torch.exp(-D / (2 * parameter**2))
            return K
