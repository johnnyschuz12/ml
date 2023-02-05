import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # input shape is 224 x 224 x 3
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # shape is 55 x 55 x 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 27 x 27 x 64

            nn.Conv2d(64, 192, kernel_size=5, padding=2), # shape is 27 x 27 x 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # shape is 13 x 13 x 192

            nn.Conv2d(192, 384, kernel_size=3, padding=1), # shape is 13 x 13 x 384
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256*6*6, kernel_size=3, padding=1), # shape is 13 x 13 x 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # shape is 6 x 6 x 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    """
    AlexNet model architecture 
    Args:
        pretrained (bool): if True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if torch.cuda.is_available():
        model = model.cuda()
    model = nn.DataParallel(model)
    return model

# Neural network with linear activation
def train_model(trainloader, epochs=10, batch_size=500, lr=0.01, tform=None):
    trainloader = ds_train.pytorch(num_workers=0, batch_size=batch_size, transform = {'images': tform, 'labels': None}, shuffle = True)
    valloader = ds_val.pytorch(num_workers=0, batch_size=batch_size, transform = {'images': tform, 'labels': None}, shuffle = True)

    model = alexnet()

    # Use standard gradient descent optimizer
    opt = optim.Adam(model.parameters(), lr=lr) #, weight_decay=0.001)
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #opt = torch.optim.SGD(model.parameters(), lr=lr)

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    uniques = {}
    losses = []
    accs = []

    # Pre-compute label encoding
    labels_list = []
    for inputs, labels in trainloader:
        labels_list.append(labels)
    labels_encoded = torch.LongTensor([uniques.setdefault(int(l), len(uniques)) for labels in labels_list for l in labels])

    # Training of the model. We use 10 epochs.
    for epoch in range(epochs):  # loop over the dataset multiple times
        rolling_loss = 0.0
        rolling_acc = 0.0
        count = 0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            n = len(inputs)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            predictions = model(inputs)
            
            # Use pre-computed label encoding
            labels = labels_encoded[count:count+n]

            loss = loss_fn(predictions, labels)
            loss.backward()
            opt.step()
            acc = (predictions.argmax(1) == labels).float().mean()
            rolling_loss += float(loss)
            rolling_acc += float(acc)
            count += n
            if count % 10000 == 0:
                print(f"Working part: {count}")

        losses.append(float(rolling_loss) / count)
        accs.append(float(rolling_acc) / count)
        print(f"Epoch: {epoch}, Loss: {float(rolling_loss) / count}, Acc: {float(rolling_acc) / count}")

    # Plot learning curve.
    plt.plot(losses)
    plt.title("Loss vs Epoch")
    plt.savefig('./alexnet/results/losses.png')
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.title("Accuracy vs Epoch")
    plt.savefig('./alexnet/results/accs.png')
    plt.show()

    torch.save(model, "./alexnet/model.h5")

    return model, uniques