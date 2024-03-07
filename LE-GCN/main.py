import torch
import wandb
from utils import measure
import os
import argparse
from config import CFG
from model import GCN, LinearModel
from DataProcess import computeD, GraphPreprocessor, KNNPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", default=8)
parser.add_argument("-m", "--model", default="le gcn + lp")
parser.add_argument("-l", "--lambda_", default=0.2)
parser.add_argument("-e", "--epochs", default=1000)
parser.add_argument("-a", "--alpha", default=0.8)
parser.add_argument("-t", "--threshold", default=0.8)
parser.add_argument("-k", "--K", default=20)
parser.add_argument("-D", "--d", default=10)
args = parser.parse_args()


config = CFG()
config.index = int(args.data)
config.model = args.model
config.lambda_ = float(args.lambda_)
config.epochs = int(args.epochs)
config.alpha = float(args.alpha)
config.threshold = float(args.threshold)
config.K = int(args.K)
config.d = int(args.d)

datasetFileNames = [
    "Movie_binary.mat",
    "Natural_Scene_binary.mat",
    "SBU_2DFE_binary.mat",
    "SJAFFE_binary.mat",
    "Yeast_alpha_binary.mat",
    "Yeast_cdc_binary.mat",
    "Yeast_cold_binary.mat",
    "Yeast_diau_binary.mat",
    "Yeast_dtt_binary.mat",
    "Yeast_elu_binary.mat",
    "Yeast_heat_binary.mat",
    "Yeast_spo5_binary.mat",
    "Yeast_spoem_binary.mat",
    "Yeast_spo_binary.mat",
]
datasetFilePaths = [os.path.join("data", _) for _ in datasetFileNames]

config_kw = {k: v for k, v in dict(vars(config)).items() if "__" not in k}
run = wandb.init(
    project=config.project,
    name=f"{config.index}-{datasetFileNames[config.index]}-{config.model} ",
    config=config_kw,
    save_code=True,
)

data = GraphPreprocessor(datasetFilePaths[config.index], config).process()
model = computeD(data, config).cuda()
pre_feature = GCN(data.cuda()).to(config.device)
data_cuda = data.cuda()


def L_m(modProb, trainLabel):
    return torch.sum(torch.sum((modProb - trainLabel).pow(2)))


data = KNNPreprocessor(datasetFilePaths[config.index], data_cuda, config)
model = LinearModel(data.trainFeature, data.trainLabel).to(config.device)
optimizer = torch.optim.Adamax(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20, eta_min=0, last_epoch=-1
)
data.trainFeature = data.trainFeature.to(config.device)
data.trainLabel = data.trainLabel.to(config.device)
data.labelDistribution = data.labelDistribution.to(config.device)
data.B = data.B.to(config.device)
trainLabel = pre_feature(data_cuda).detach()

def output():
    model.eval()
    target = model(data.trainFeature)
    return target


def train():
    model.train()
    optimizer.zero_grad()
    modProb = model(data.trainFeature) 
    L = L_m(modProb, trainLabel)
    R = torch.trace(modProb.T @ data.B @ modProb)
    target = L + config.lambda_ * R
    target.backward()
    optimizer.step()
    return target.item()


def test():
    model.eval()
    target = model(data.trainFeature)
    target = pre_feature(data_cuda)
    measures = measure(target, data.labelDistribution)
    return measures


for i in range(config.epochs):
    loss = train()
    if i % 50 == 0:
        with torch.no_grad():
            model.eval()
            measures = test()
            log_dict = {}
            log_dict["epoch"] = i
            log_dict["cheb"] = measures[0]
            log_dict["clark"] = measures[1]
            log_dict["canber"] = measures[2]
            log_dict["KL"] = measures[3]
            log_dict["cosine"] = measures[4]
            log_dict["Intersec"] = measures[5]
            log_dict["loss"] = loss
            log_dict["lr"] = scheduler.get_last_lr()[0]
            run.log(log_dict)
    scheduler.step()
