import torch
import helper
from tqdm import tqdm
from model import CNN, CNN2
from torch import nn, optim
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import torch.utils.data as data
import cv2
from torchvision import datasets, transforms
import random

DATADIR = './alphanumeric'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
CATEGORIES2 = ['false', 'true']
IMSIZE = 28

transform = transforms.Compose([transforms.Resize((IMSIZE, IMSIZE)), transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
train_data = datasets.ImageFolder(root='./alphanumeric/train', transform=transform)
train_data_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.ImageFolder(root='./alphanumeric/train', transform=transform)
test_data_loader = data.DataLoader(test_data, batch_size=64, shuffle=True)

# letters:
# trainset3 = datasets.EMNIST('/.pytorch/EMNIST_data/', 'letters', train=True, download=True, transform=transform)
# image, label = next(iter(train_data_loader))
# print(label)
# print(CATEGORIES2[label[0]])
# img = image[0, :]
# img = img.numpy().transpose((1, 2, 0))
# im2 = img[:, :, 0]
# print(img.shape)
# plt.imshow(im2, 'gray')
# plt.show()

model = CNN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 6

model.train()

for e in range(epochs):
    running_loss = 0
    for images, labels in tqdm(train_data_loader):
        log_ps = model(images)
        loss = criterion(log_ps, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_data_loader)}")

torch.save(model.state_dict(), 'train1.pt')

# model.load_state_dict(torch.load('train1.pt'))

model.eval()

accuracy = 0
for images, labels in test_data_loader:
    log_ps = model(images)

    # get the predictions: argmax etc.
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Accuracy: {:.3f}".format(accuracy / len(test_data_loader)))
dataiter = iter(test_data_loader)
images, labels = dataiter.next()
print(labels)
img = images[1]
img = img.unsqueeze(0)

ps = torch.exp(model(img))
print(ps)
# Plot the image and probabilities
helper.view_classify(img, ps, version='CAT')
