import torch
import matplotlib.pyplot as plt

# Neural network with linear activation
def train_model(mnist_training, epochs=10, batch_size=500, lr=0.01):
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
            accs.append(float(acc))
        print(f"Epoch: {epoch}, Loss: {float(loss)}, Acc: {float(acc)}")

    # Plot learning curve.
    plt.plot(losses)
    plt.title("Loss vs Epoch")
<<<<<<< HEAD
    plt.savefig('nn/results/losses.png')
=======
    plt.savefig('./nn/results/losses.png')
>>>>>>> 1baf84f (handwriting_recognition/fashion: Add fashion)
    plt.show()

    # Plot learning curve.
    plt.plot(accs)
    plt.title("Accuracy vs Epoch")
<<<<<<< HEAD
    plt.savefig('nn/results/accs.png')
    plt.show()

    torch.save(model, "nn/model.h5")
=======
    plt.savefig('./nn/results/accs.png')
    plt.show()

    torch.save(model, "./nn/model.h5")
>>>>>>> 1baf84f (handwriting_recognition/fashion: Add fashion)

    return model