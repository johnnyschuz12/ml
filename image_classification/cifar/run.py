import cnn
import nn
from cnn.run import run_cnn
from nn.run import run_nn
from resnet.run import run_resnet
import torch
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib as mpl

transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

# Load datasets for training and testing.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

# Run NN
run_nn(trainloader, testloader)

# Run CNN
run_cnn(trainloader, testloader)

# Run RESNET
run_resnet(trainloader, testloader)