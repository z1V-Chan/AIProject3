import os, random
import torch, numpy as np
import torch_pruning as tp
import pickle

from torch import nn
from torchvision import datasets, transforms

from models.YourNet import YourNet
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params


BATCHSIZE = 64
RATE = 0.09
HLR = 0.01
LLR = 0.008
EPOCH = 24
DEVICE = "cuda"
BASEACC = 0.984
ITER = 11

SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEFAULTCHECKPOINTDIR = "./checkpoints/YourNet/init/"
CHECKPOINTDIR = "./checkpoints/YourNet/pruning/"
LOGPKLFILE = "./log.pkl"
FINALMODEL = "./finalModel.pth"

# CONVSTRUCTURE = [
#     "model.conv1",
#     "model.conv2",
# ]

LINEARSTRUCTURE = [
    "model.fc1",
    "model.fc2",
]


def train(model, train_loader, test_loader, loss_fn, checkpointDir: str):
    Accs = []
    checkPoints = []
    size = len(train_loader.dataset)
    model.train()
    optimizer = torch.optim.Adamax(model.parameters(), lr=HLR)
    for epoch in range(EPOCH):
        print(f"Epoch {epoch}\n-------------------------------")
        if epoch == EPOCH // 2:
            optimizer = torch.optim.SGD(model.parameters(), lr=LLR, momentum=0.2)

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # Compute prediction error
            pred_y = model(X)
            loss = loss_fn(pred_y, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                loss, current = loss.item(), batch_idx * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = get_accuracy(model, test_loader, DEVICE)
        print("Accuracy: %.3f" % accuracy)
        checkPoint = checkpointDir + f"epoch-{epoch}.pth"
        torch.save(model, checkPoint)
        Accs.append(accuracy)
        checkPoints.append(checkPoint)
    idx = max([i for i in range(len(Accs))], key=lambda x: Accs[x])
    # print(model)
    return Accs[idx], checkPoints[idx]


def main(checkpointDir: str, model=None, lastCheckpoint: str = None):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root="./data",
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

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

    if lastCheckpoint is not None:
        model = torch.load(lastCheckpoint, map_location=DEVICE)

    elif model is None:
        model = YourNet()

    model = model.to(device=DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)

    return train(model, train_loader, test_loader, loss_fn, checkpointDir)


def prune(
    model: YourNet, module, type, test_loader, cnt, bestCheckpoints: list, accs: list
):
    model = model.eval()

    strategy = tp.strategy.L1Strategy()

    DG = tp.DependencyGraph()

    DG.build_dependency(model, example_inputs=iter(test_loader).next()[0])

    pruning_idxs = strategy(module.weight, amount=RATE)
    pruning_plan = DG.get_pruning_plan(module, type, idxs=pruning_idxs)

    pruning_plan.exec()

    checkpointDir = CHECKPOINTDIR + f"{cnt}/"

    acc, bestCheckpoint = main(checkpointDir, model)

    if acc >= BASEACC:  # accs[0]
        accs.append(acc)
        bestCheckpoints.append(bestCheckpoint)
        return True
    else:
        return False


def netPruning(bestCheckpoint):
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

    for _ in range(ITER):
        for m in LINEARSTRUCTURE:
            if prune(
                model,
                eval(m),
                tp.prune_linear,
                test_loader,
                cnt,
                bestCheckpoints,
                accs,
            ):
                print(model)
                cnt += 1
            model = torch.load(bestCheckpoints[-1], map_location=DEVICE)

    # for m in CONVSTRUCTURE:
    #     while prune(
    #         model, eval(m), tp.prune_conv, test_loader, cnt, bestCheckpoints, accs
    #     ):
    #         print(model)
    #         cnt += 1
    #     model = torch.load(bestCheckpoints[-1], map_location=DEVICE)

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
            MACs / (1000 ** 2),
            params / (1000 ** 2),
        )
    )
    print("----------------------------------------------------------------")
    print(bestCheckpoints[-1])

    print(model)
    tmpDict = model.state_dict()
    tmpDict.pop("total_ops")
    tmpDict.pop("total_params")
    torch.save(model.state_dict(), FINALMODEL)

    with open(LOGPKLFILE, "wb") as f:
        pickle.dump([bestCheckpoints, accs], f)


if __name__ == "__main__":
    acc, bestCheckpoint = main(DEFAULTCHECKPOINTDIR)
    print(acc, bestCheckpoint)
    netPruning(bestCheckpoint)
