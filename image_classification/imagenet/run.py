import alexnet
from alexnet.run import run_nn
import torch
import torchvision
from torchvision import datasets, transforms, utils
import matplotlib as mpl
import deeplake
import warnings
warnings.filterwarnings("ignore")

ds_train = deeplake.load("hub://activeloop/tiny-imagenet-train")
ds_val = deeplake.load("hub://activeloop/tiny-imagenet-validation")
ds_test = deeplake.load("hub://activeloop/tiny-imagenet-test")


tform = transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.RandomRotation(20), # Image augmentation
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
    transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)), # Some images are grayscale, so we need to add channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# trainloader = ds_train.pytorch(num_workers=0, batch_size=4, transform = {'images': tform, 'labels': None}, shuffle = True)
# valloader = ds_val.pytorch(num_workers=0, batch_size=4, transform = {'images': tform, 'labels': None}, shuffle = True)
# testloader = ds_test.pytorch(num_workers=0, batch_size=4, transform = {'images': tform}, shuffle = True)

# Run NN
run_nn(ds_train, ds_val, tform)
