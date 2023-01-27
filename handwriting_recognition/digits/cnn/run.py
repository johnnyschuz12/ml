from cnn.train import train_model
from cnn.validate import validate_model
import torch
import matplotlib.pyplot as plt

def run_cnn(mnist_training, mnist_val):
    # Optimized parameters
    epochs = 10
    batch_size = 100
    lr=0.001

    # Train the model and save it
    model = train_model(mnist_training, epochs, batch_size, lr)

    # Validate the model by making predictions
    train_predictions, train_acc, train_loss = validate_model(mnist_training)
    val_predictions, val_acc, val_loss = validate_model(mnist_val)


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
    plt.savefig('results/pil.png')
    plt.show()

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image, label = mnist_training[i][0], train_predictions[i]          # returns PIL image with its labels
        ax.set_title(f"Train?: {label}")
        ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    plt.savefig('cnn/results/train.png')
    plt.show()

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(1.5*cols, 2*rows))
    for i, ax in enumerate(axes.flatten()):
        image, label = mnist_val[i][0], val_predictions[i]          # returns PIL image with its labels
        ax.set_title(f"Test?: {label}")
        ax.imshow(image.squeeze(0), cmap='gray')  # we get a 1x28x28 tensor -> remove first dimension
    plt.savefig('cnn/results/val.png')
    plt.show()
