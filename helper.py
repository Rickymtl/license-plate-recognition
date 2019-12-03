import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable

CATEGORIES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# modified version of the pytoch tutorial helper to show class predictions.
def view_classify(img, ps, version="CHAR"):
    ''' Function for viewing an image and it's predicted classes.
    version:CHAR for clasification CNN
            CAT for detection(category) CNN
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    if version == "CHAR":
        ax2.barh(np.arange(36), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(36))

        ax2.set_yticklabels(CATEGORIES, size='small')
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)

        plt.tight_layout()
        plt.show()
    elif version == "CAT":
        ax2.barh(np.arange(2), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(2))

        ax2.set_yticklabels(['False', 'True'], size='small')
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)

        plt.tight_layout()
        plt.show()
