import torch
import matplotlib.pyplot as plt

def validate_model(mnist_val):
    model = torch.load("./nn/model.h5")

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    # Load all 10000 images from the validation set.
    n = 10000
    loader = torch.utils.data.DataLoader(mnist_val, batch_size=n)
    images, labels = iter(loader).__next__()

    # The tensor images has the shape [10000, 1, 28, 28]. Reshape the tensor to
    # [10000, 784] as our model expected a flat vector.
    data = images.view(n, -1)

    # Use our model to compute the class scores for all images. The result is a
    # tensor with shape [10000, 10]. Row i stores the scores for image images[i].
    # Column j stores the score for class j.
    predictions = model(data)

    # For each row determine the column index with the maximum score. This is the
    # predicted class.
    predicted_classes = torch.argmax(predictions, dim=1)

    # Accuracy = number of correctly classified images divided by the total number
    # of classified images.
    acc = sum(predicted_classes.numpy() == labels.numpy()) / n

    loss = float(loss_fn(predictions, labels))

    return predicted_classes, acc, loss
