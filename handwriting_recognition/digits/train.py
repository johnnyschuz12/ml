import torch
import matplotlib.pyplot as plt


def train_model(mnist_training, model, epochs=10, batch_size=500, lr=0.01):
    # Use Adam as optimizer.
    opt = torch.optim.Adam(params=model.parameters())

    # Use CrossEntropyLoss for as loss function.
    loss_fn = torch.nn.CrossEntropyLoss()

    # We train the model with batches of examples.
    train_loader = torch.utils.data.DataLoader(mnist_training, batch_size=batch_size, shuffle=True)

    # Training of the model. We use 10 epochs.
    losses = []

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
        print(f"Epoch: {epoch}, Loss: {float(loss)}")

    # Plot learning curve.
    plt.plot(losses)
    plt.show()


    torch.save(model, "digits/models/model.h5")

    return model








