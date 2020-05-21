import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import pickle


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.lin0 = nn.Linear(12*9*128, 1024)
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(1024, 51)


    def forward(self, xb):
        xb = F.relu(F.max_pool2d(self.conv0(xb), 2))
        xb = F.relu(F.max_pool2d(self.conv1(xb), 2))
        xb = F.relu(F.max_pool2d(self.conv2(xb), 2))
        xb = xb.view(-1, 12*9*128)
        xb = F.relu(self.lin0(xb))
        embedding = xb

        xb = self.drop(xb)
        xb = torch.tanh(self.lin1(xb))
        return xb, embedding

def getModel(lr):
    model = CNN()
    return model, optim.Adam(model.parameters(), lr=lr)

def getDatasets():
    with open("./data/HMDB_train_set.pkl", "rb") as file:
        train_dataset = pickle.load(file)
    with open("./data/HMDB_dev_set.pkl", "rb") as file:
        dev_dataset = pickle.load(file)
    train_set = TensorDataset(torch.Tensor(train_dataset["xs"]), torch.LongTensor(train_dataset["ys"]))
    dev_set = TensorDataset(torch.Tensor(dev_dataset["xs"]), torch.LongTensor(dev_dataset["ys"]))
    return train_set, dev_set

def getDataloaders(batch_size):
    train_set, dev_set = getDatasets()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size)
    return train_loader, dev_loader

def preprocess(x, y):
    return x.view(-1, 3, 96, 72).to(dev), y.to(dev)

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))



def fit(epochs, model, loss_func, opt, train_loader, dev_loader):
    val_loss_curve = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            out, _ = model(xb)
            loss = loss_func(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

            total_loss += loss.detach() * len(xb)
        val_loss_curve.append(total_loss)

        model.eval()
        with torch.no_grad():
            loss = 0
            acc = 0
            data_num = 0
            for xb, yb in dev_loader:
                out, _ = model(xb)
                loss += loss_func(out, yb).detach() * len(xb)
                acc += (torch.argmax(out, dim=1) == yb).sum().item()
                data_num += len(xb)
            #val_loss_curve.append(loss / data_num)
            print("epoch: {}    loss: {}    acc: {}".format(epoch, loss / data_num, float(acc) / data_num))

    return val_loss_curve

def draw_learning_curve(xs, ys):
    plt.figure(figsize=(5,5))
    plt.plot(xs, ys, "*-")
    plt.title("Learning Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("./learning_curve")



if __name__ == "__main__":

    dev = torch.device("cuda")
    batch_size = 32
    lr = 0.0001
    epochs = 20


    train_loader, dev_loader = getDataloaders(batch_size)
    train_loader = WrappedDataLoader(train_loader, preprocess)
    dev_loader = WrappedDataLoader(dev_loader, preprocess)

    model, opt = getModel(lr)
    model.to(dev)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
    loss_func = F.cross_entropy

    val_loss_curve = fit(epochs, model, loss_func, opt, train_loader, dev_loader)
    draw_learning_curve(range(epochs), val_loss_curve)

    torch.save(model.state_dict(), "./data/model.pt")