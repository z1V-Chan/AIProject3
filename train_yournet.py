import os, random
import torch, numpy as np
import lnp


from torch import nn
from torchvision import datasets, transforms

from models.YourNet import YourNet
from eval.metrics import get_accuracy


BATCHSIZE = 64
HLR = 0.012
LLR = 0.008
EPOCH = 15
DEVICE = "cpu"
DEFAULTCHECKPOINTDIR = "./checkpoints/YourNet/init/"
SEED = 1007

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def train(model, train_loader, test_loader, loss_fn, checkpointDir: str):
    Accs = []
    checkPoints = []
    size = len(train_loader.dataset)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=HLR)
    for epoch in range(EPOCH):
        print(f"Epoch {epoch}\n-------------------------------")
        if epoch == EPOCH // 3:
            optimizer = torch.optim.SGD(model.parameters(), lr=LLR, momentum=0.25)

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
        model = YourNet().to(device=DEVICE)

    loss_fn = nn.CrossEntropyLoss()

    if not os.path.exists(checkpointDir):
        os.makedirs(checkpointDir)

    return train(model, train_loader, test_loader, loss_fn, checkpointDir)


if __name__ == "__main__":
    print(main(DEFAULTCHECKPOINTDIR))
