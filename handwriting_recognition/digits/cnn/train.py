import torch
import matplotlib.pyplot as plt

# Convolutional neural network with pooling
def train_model(mnist_training, epochs=10, batch_size=100, lr=0.001):
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
