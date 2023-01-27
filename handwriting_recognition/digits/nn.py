from train import train_nn_model, train_cnn_model
from test import validate_nn_model, validate_cnn_model
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Transform PIL image into a tensor. The values are in the range [0, 1]
t = transforms.ToTensor()

# Load datasets for training and testing.
mnist_training = datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=t)
mnist_val = datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=t)


# Optimized parameters
epochs = 10
batch_size = 500
lr=0.01

# Train the model and save it
model = train_nn_model(mnist_training, epochs, batch_size, lr)

# Validate the model by making predictions
train_predictions, train_acc, train_loss = validate_nn_model(mnist_training)
val_predictions, val_acc, val_loss = validate_nn_model(mnist_val)


print("Train Accuracy and Loss")
print(train_acc)
print(train_loss)

print("Test Accuracy and Loss")
print(val_acc)
print(val_loss)

# Plot some digits with trained label.
cols = 5
rows = 5

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
for i, ax in enumerate(axes.flatten()):
    image, label = mnist_training[i]          # returns PIL image with its labels
    ax.set_title(f"Label: {label}")
    ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
plt.show()

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
for i, ax in enumerate(axes.flatten()):
    image, label = mnist_training[i][0], train_predictions[i]          # returns PIL image with its labels
    ax.set_title(f"Train?: {label}")
    ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
plt.show()

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
for i, ax in enumerate(axes.flatten()):
    image, label = mnist_val[i][0], val_predictions[i]          # returns PIL image with its labels
    ax.set_title(f"Test?: {label}")
    ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
plt.show()
