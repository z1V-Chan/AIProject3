import torch, numpy as np
import torch_pruning as tp

import argparse
import torch
import random
import pickle

from torchvision import datasets, transforms
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params
from models.YourNet import YourNet
import train_yournet

BATCHSIZE = 64
RATE = 0.1
DEVICE = "cpu"
BESTCHECKPOINT = "./checkpoints/YourNet/pruned/10/epoch-11.pth"
CHECKPOINTDIR = "./checkpoints/YourNet/pruned1/"
PKLFILE = "./result1.pkl"

STRUCTURE = [
    # "model.conv1",
    # "model.conv2",
    "model.fc1",
    "model.fc2",
]
SEED = 1007

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(bestCheckpoint):
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    model = torch.load(bestCheckpoint, map_location=DEVICE)
    # print(model)

    cnt = 0

    accs = [get_accuracy(model, test_loader, DEVICE)]
    bestCheckpoints = [bestCheckpoint]

    for m in STRUCTURE:
        while prune(model, eval(m), test_loader, cnt, bestCheckpoints, accs):
            print(model)
            cnt += 1
        model = torch.load(bestCheckpoints[-1], map_location=DEVICE)

    accuracy = get_accuracy(model, test_loader, DEVICE)
    infer_time = get_infer_time(model, test_loader, DEVICE)
    MACs, params = get_macs_and_params(model, DEVICE)

    print("----------------------------------------------------------------")
    print(
        "| %10s | %8s | %14s | %9s | %7s |"
        % ("Model Name", "Accuracy", "Infer Time(ms)", "Params(M)", "MACs(M)")
    )
    print("----------------------------------------------------------------")
    print(
        "| %10s | %8.3f | %14.3f | %9.3f | %7.3f |"
        % (
            "YourNet",
            accuracy,
            infer_time * 1000,
            params / (1000 ** 2),
            MACs / (1000 ** 2),
        )
    )
    print("----------------------------------------------------------------")
    print(bestCheckpoints[-1])

    print(model)
    torch.save(model, "./finalModel1.pth")

    with open(PKLFILE, "wb") as f:
        pickle.dump([bestCheckpoints, accs], f)


def prune(model: YourNet, module, test_loader, cnt, bestCheckpoints: list, accs: list):
    model = model.eval()

    strategy = tp.strategy.L1Strategy()

    DG = tp.DependencyGraph()

    DG.build_dependency(model, example_inputs=iter(test_loader).next()[0])

    pruning_idxs = strategy(module.weight, amount=RATE)
    pruning_plan = DG.get_pruning_plan(module, tp.prune_linear, idxs=pruning_idxs)

    pruning_plan.exec()

    checkpointDir = CHECKPOINTDIR + f"{cnt}/"

    acc, bestCheckpoint = train_yournet.main(checkpointDir, model)

    if acc > 0.981:  # accs[0]
        accs.append(acc)
        bestCheckpoints.append(bestCheckpoint)
        return True
    else:
        return False


if __name__ == "__main__":
    main(BESTCHECKPOINT)
