import torch


class CFG:
    project = "label-distribution"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = 0
    dataset = str(index) + ".pt"
    model = ("le  gcn + lp",)
    lambda_ = 0.2
    epochs = 1000
    alpha = 0.8
    threshold = 0.8
    K = 20
    d = 10
