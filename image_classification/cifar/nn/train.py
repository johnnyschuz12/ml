import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Neural network with linear activation
def train_model(trainloader, epochs=10, batch_size=500, lr=0.01):
    model = CIFAR10Model()

    # Use standard gradient descent optimizer
    opt = optim.Adam(model.parameters(), lr=lr) #, weight_decay=0.001)
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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
    plt.savefig('./nn/results/losses.png')
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.title("Accuracy vs Epoch")
    plt.savefig('./nn/results/accs.png')
    plt.show()

    torch.save(model, "./nn/model.h5")

    return model