import torch
import os


def measure(predict, target):
    node_nums = predict.shape[0]
    types = predict.shape[1]
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == 0:
                target[i][j] = 0.001
    # cheb
    tensor = torch.abs(predict - target)
    cheb_m = 0
    for i in range(tensor.shape[0]):
        cheb_m += max(tensor[i])
    cheb_m /= tensor.shape[0]
    # clark
    clark_m = 0
    for i in range(node_nums):
        clark_sample = 0
        for j in range(types):
            clark_sample += ((predict[i][j] - target[i][j]).pow(2)) / (
                (predict[i][j] + target[i][j]).pow(2)
            )
        clark_m += torch.sqrt(clark_sample)
    clark_m /= node_nums
    # canber
    canber_m = 0
    for i in range(node_nums):
        canber_sample = 0
        for j in range(types):
            canber_sample += torch.abs(predict[i][j] - target[i][j]) / (
                predict[i][j] + target[i][j]
            )
        canber_m += canber_sample
    canber_m /= node_nums
    # cosine
    cosine_m = 0
    for i in range(node_nums):
        cosine_sample = 0
        tmp1 = 0
        tmp2 = 0
        for j in range(types):
            cosine_sample += predict[i][j] * target[i][j]
        for j in range(types):
            tmp1 += predict[i][j] ** 2
        tmp1 = torch.sqrt(tmp1)
        for j in range(types):
            tmp2 += target[i][j] ** 2
        tmp2 = torch.sqrt(tmp2)
        cosine_sample /= tmp1 * tmp2
        cosine_m += cosine_sample
    cosine_m /= node_nums
    # Intersec
    Intersec_m = 0
    for i in range(node_nums):
        Intersec_sample = 0
        for j in range(types):
            Intersec_sample += torch.min(predict[i][j], target[i][j])
        Intersec_m += Intersec_sample
    Intersec_m /= node_nums
    # KL
    KL_m = 0
    for i in range(node_nums):
        KL_sample = 0
        for j in range(types):
            KL_sample += target[i][j] * torch.log(target[i][j] / predict[i][j] + 1e-4)
        KL_m += KL_sample
    KL_m /= node_nums
    return (
        cheb_m.item(),
        clark_m.item(),
        canber_m.item(),
        KL_m.item(),
        cosine_m.item(),
        Intersec_m.item(),
    )


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
