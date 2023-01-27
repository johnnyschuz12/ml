import torch
import matplotlib.pyplot as plt

# Neural network with linear activation
def train_nn_model(mnist_training, epochs=10, batch_size=500, lr=0.01):
    model = torch.nn.Sequential(
        torch.nn.Linear(28*28, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    # Use Adam as optimizer.
    opt = torch.optim.Adam(params=model.parameters())

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    # We train the model with batches of examples.
    train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)

    # Training of the model. We use 10 epochs.
    losses = []
    accs = []

    for epoch in range(epochs):
        for imgs, labels in train_loader:
            n = len(imgs)
            # Reshape data from [500, 1, 28, 28] to [500, 784] and use the model to make predictions.
            predictions = model(imgs.view(n, -1))  
            # Compute the loss.
            loss = loss_fn(predictions, labels) 
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            predicted_classes = torch.argmax(predictions, dim=1)
            acc = sum(predicted_classes.numpy() == labels.numpy()) / n
        print(f"Epoch: {epoch}, Loss: {float(loss)}, Acc: {float(acc)}")

    # Plot learning curve.
    plt.plot(losses)
    plt.show()

    torch.save(model, "models/nn_model.h5")

    return model


# Convolutional neural network with pooling
def train_cnn_model(mnist_training, epochs=10, batch_size=100, lr=0.001):
    model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(7*7*64, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)

    losses = []
    accs = []

    # Train the model
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            acc = (output.argmax(1) == labels).float().mean()
            train_acc += acc
            losses.append(float(loss))
            accs.append(acc)

        train_loss /= i + 1
        train_acc /= i + 1
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')


    # Plot learning curve.
    plt.plot(losses)
    plt.savefig('results/cnn/losses.png')
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.savefig('results/cnn/accs.png')
    plt.show()


    torch.save(model, "models/cnn_model.h5")

    return model
