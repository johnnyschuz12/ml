import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Neural network with linear activation
def train_model(trainloader, epochs=10, batch_size=500, lr=0.01):
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    model = nn.DataParallel(model)

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
    plt.savefig('./cnn/results/losses.png')
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.title("Accuracy vs Epoch")
    plt.savefig('./cnn/results/accs.png')
    plt.show()

    torch.save(model, "./cnn/model.h5")

    return model

