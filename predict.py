import torch
import helper
from tqdm import tqdm
from model import CNN, CNN2, CNN3
from torch import nn, optim
from torchsummary import summary
import PIL
import torch.nn.functional as F
import os
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import torch.utils.data as data
import cv2
from torchvision import datasets, transforms

# model = CNN3()
# summary(model, (1,28,28))
# I put this file into a wrapper for my partner to use.
def predict_plate(DIR):
    
    CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    IMSIZE = 28

    transform = transforms.Compose([transforms.Resize((IMSIZE, IMSIZE), interpolation=PIL.Image.NEAREST),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # load the dataset for prediction from DIR
    pred_data = datasets.ImageFolder(root=DIR, transform=transform)
    pred_data_loader = data.DataLoader(pred_data, batch_size=64, shuffle=False)

    # show the head of the dataset to make sure the lata is loaded correctly
    image, label = next(iter(pred_data_loader))
    img = image[0, :]
    img = img.numpy().transpose((1, 2, 0))
    im2 = img[:, :, 0]
    print(img.shape)
    plt.imshow(im2, 'gray')
    plt.show()

    # initialize models and load in the weights.
    categorizer = CNN()
    model = CNN3()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    categorizer.load_state_dict(torch.load('train1.pt'))
    model.load_state_dict(torch.load('train2.pt'))

    
    save_file = 'plates.txt'
    save_file = open(save_file,'w')

    dataiter = iter(pred_data_loader)
    images, labels = dataiter.next()
    plate = ''
    print(labels)
    for img in images:
        # check if an image contains a character.
        im = img.unsqueeze(0)
        ps = torch.exp(categorizer(im))
        top_p, top_class = ps.topk(1, dim=1)
        #         helper.view_classify(im, ps, version='CAT')

        if top_class.item()==0:
            continue
        # if code reaches here, there is a character in image.
        im = img.unsqueeze(0)
        ps = torch.exp(model(im))
        top_p, top_class = ps.topk(1, dim=1)
        print(CATEGORIES[top_class.item()])
        print(top_p.item())
        # some improvements:
        # I'll detail this in report.
        if top_p.item()<0.5:
            continue
        if CATEGORIES[top_class.item()]=='O':
            plate = plate + '0'
        elif CATEGORIES[top_class.item()] =='I':
            plate = plate +'1'
        elif CATEGORIES[top_class.item()] == 'U':
            plate = plate +'V'
        elif CATEGORIES[top_class.item()] == 'G':
            plate = plate +'6'
        else:
            plate = plate + CATEGORIES[top_class.item()]
        # Plot the image and probabilities
        # helper.view_classify(im, ps, version='CHAR')
    save_file.write(plate+'\n')
    save_file.close()
    print(plate)
    return plate

if __name__ == '__main__':
    predict_plate()
