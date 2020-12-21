from __future__ import print_function
from pathlib import Path
import requests
import torch
import pickle
import gzip
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super(Mnist_Logistic, self).__init__()

    def line(self):
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
#PATH.mkdir(parents=True)
URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f)
#pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray")
print(x_train.shape)
x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
#print(x_train, y_train)
#print(x_train.shape)
#print(y_train.min(), y_train.max())

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
bs = 64 # batch size
lr = 0.5
def get_model():
    model = Mnist_Logistic()
    model.line()
    return model, optim.SGD(model.parameters(), lr = lr)
model, opt = get_model()

loss_func = F.cross_entropy
def fit():
    epochs = 2
    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
    print(loss_func(model(xb), yb))

#fit()

print(torch.cuda.is_available())