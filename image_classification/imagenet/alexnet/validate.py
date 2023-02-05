import torch
import matplotlib.pyplot as plt

def validate_model(mnist_val, uniques):
    model = torch.load("./nn/model.h5")

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    predictions = []
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in mnist_val:
            images, labels = data

            # convert labels
            real_labels = []
            for l in labels:
                if int(l) not in uniques:
                    uniques[int(l)] = unique_num
                    unique_num += 1
                real_labels.append(uniques[int(l)])
            real_labels = torch.LongTensor(real_labels)

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += real_labels.size(0)
            correct += (predicted == real_labels).sum().item()
            predictions.append(predicted)

    print(f'Test Accuracy: {100 * correct // total} %')

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in mnist_val:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Test Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    return predictions
