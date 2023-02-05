from alexnet.train import train_model
from alexnet.validate import validate_model
import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img, file, predictions):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.set_title(f"{' '.join(f'{classes[predictions[j]]:13s}' for j in range(4))}")
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.savefig(file)
    plt.show()

def run_nn(mnist_training, mnist_val):
    # Optimized parameters
    epochs = 10
    batch_size = 500
    lr=0.01

    # Train the model and save it
    model, uniques = train_model(mnist_training, epochs, batch_size, lr)

    #model = torch.load("./nn/model.h5")

    # Validate the model by making predictions
    train_predictions = validate_model(mnist_training, uniques)
    test_predictions = validate_model(mnist_val, uniques)

    with torch.no_grad():
        (image, label) = next(iter(mnist_training))

        real_label = uniques(label)
        imshow(utils.make_grid(image.squeeze(0)), './nn/results/pil.png', label)

    with torch.no_grad():
        (image, label) = next(iter(mnist_training))
        outputs = model(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        imshow(utils.make_grid(image.squeeze(0)), './nn/results/train.png', predicted)

    with torch.no_grad():
        (image, label) = next(iter(mnist_val))
        outputs = model(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        imshow(utils.make_grid(image.squeeze(0)), './nn/results/val.png', predicted)

