import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, num_layers=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(32, num_layers, stride=1)
        self.layer2 = self._make_layer(8*32, num_layers, stride=2)
        self.bn = nn.BatchNorm2d(64*32)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1= nn.Linear(64*32, 64)
        self.linear2= nn.Linear(64, 10)

    def _make_layer(self, in_channels, num_layers, stride):
        out_channels = in_channels
        strides = [stride] + [1]*(num_layers-1)
        layers = []
        for stride in strides:
            out_channels *= 2
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

# Neural network with linear activation
def train_model(trainloader, epochs=10, batch_size=500, lr=0.01):
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()

    # Use standard gradient descent optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training of the model. We use 10 epochs.
    losses = []
    accs = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        rolling_loss = 0.0
        rolling_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            predictions = model(inputs)
            loss = loss_fn(predictions, labels)
            loss.backward()
            opt.step()
            acc = (predictions.argmax(1) == labels).float().mean()
            rolling_loss += float(loss)
            rolling_acc += float(acc)

        losses.append(float(rolling_loss) / i)
        accs.append(float(rolling_acc) / i)
        print(f"Epoch: {epoch}, Loss: {float(rolling_loss) / i}, Acc: {float(rolling_acc) / i}")

    # Plot learning curve.
    plt.plot(losses)
    plt.title("Loss vs Epoch")
    plt.savefig('./resnet/results/losses.png')
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.title("Accuracy vs Epoch")
    plt.savefig('./resnet/results/accs.png')
    plt.show()

    torch.save(model, "./resnet/model.h5")

    return model

