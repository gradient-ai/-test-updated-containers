# Test new container on Gradient with Python: nvcr.io/nvidia/pytorch:21.09-py3
#
# Last updated: Oct 21st 2021

# --- Check versions of packages of interest ---

# Run same commands in notebook and .py
# Not running prefixed by "!" as that calls CLI version

import platform
print(platform.python_version())

import subprocess

# Installing gradient breaks PyTorch because gradient downgrades numpy
#subprocess.run('pip3 install gradient', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

#import gradient
#gradient.version.version

subprocess.run('pip --version', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

import numpy as np
print(np.__version__)

import torch
print(torch.__version__)

# --- Check GPU ---

# PyTorch lines are from [1]

subprocess.run('nvidia-smi', shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)

# --- Check model picks up GPU ---

# Follow PyTorch's Quick Start on FashionMNIST that has GPU [2]
# Include a full set of steps so we can see the model eval, i.e., it ran properly
# This assumes 1 GPU: testing multi-GPU setups is not covered yet

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "cpu" # This line points it to the GPU if available
print("Using {} device".format(device))                 # And we see if it picked it up

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# [1] https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
# [2] Subset of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
