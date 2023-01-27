# ml
Repository of Useful ML Applications

## Handwritten single digit classification
Classify handwritten digits from the MNIST dataset.
Implementations of CNN and NN and results are shown.

NN Model:

model = torch.nn.Sequential(
  torch.nn.Linear(28x28, 256),
  torch.nn.ReLU(),
  torch.nn.Linear(256, 10)
)

CNN Model:

model = torch.nn.Sequential(
  torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
  torch.nn.ReLU(),
  torch.nn.MaxPool2d(kernel_size=2, stride=2),
  torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
  torch.nn.ReLU(),
  torch.nn.MaxPool2d(kernel_size=2, stride=2),
  torch.nn.Flatten(),
  torch.nn.Linear(7x7x64, 1024),
  torch.nn.ReLU(),
  torch.nn.Linear(1024, 10)
)
