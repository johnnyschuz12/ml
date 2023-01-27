import cnn
import nn
from cnn.run import run_cnn
from nn.run import run_nn

from torchvision import datasets, transforms

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and testing.
mnist_training = datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=t)
mnist_val = datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=t)

# Run CNN
run_nn(mnist_training, mnist_val)

# Run NN
run_cnn(mnist_training, mnist_val)
