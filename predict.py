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


DATADIR = 'English/pred/imgs/'
CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
IMSIZE = 28

transform = transforms.Compose([transforms.Resize((IMSIZE, IMSIZE)),transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
pred_data = datasets.ImageFolder(root='./English/pred', transform=transform)
pred_data_loader = data.DataLoader(pred_data, batch_size=64, shuffle=False)

categorizer = CNN()
model = CNN2()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

categorizer.load_state_dict(torch.load('train1.pt'))
model.load_state_dict(torch.load('train2.pt'))


save_file = 'plates.txt'
save_file = open(save_file,'a')

dataiter = iter(pred_data_loader)
images, labels = dataiter.next()
plate = ''
print(labels)
for img in images:
    im = img.unsqueeze(0)
    ps = torch.exp(categorizer(im))
    top_p, top_class = ps.topk(1, dim=1)
    # helper.view_classify(im, ps, version='CAT')

    if top_class.item()==0:
        continue
    im = img.unsqueeze(0)
    ps = torch.exp(model(im))
    top_p, top_class = ps.topk(1, dim=1)
    print(CATEGORIES[top_class.item()])
    print(top_p.item())
    if top_p.item()<0.5:
        continue
    if CATEGORIES[top_class.item()]=='O':
        plate = plate + '0'
    elif CATEGORIES[top_class.item()] =='I':
        plate = plate +'1'
    elif CATEGORIES[top_class.item()] == 'U':
        plate = plate +'V'
    else:
        plate = plate + CATEGORIES[top_class.item()]
    # Plot the image and probabilities
    # helper.view_classify(im, ps, version='CHAR')
save_file.write(plate+'\n')
save_file.close()
print(plate)

