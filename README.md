# license-plate-recognition
the code for our CSC420 project

# English 
This directory contains the cleaned up version of CHARS74Kdataset we used to train our CNN for classification.

# alphanumeric
This directory contains the dataset I put together for the detection CNN

# helper.py
Modified code from pytorch tutorial for display of CNN prediction output

# model.py
The CNN models we used

# train.py 
The training and evaluation of both CNN. 

# train1.pt train2.pt train2v2.pt
The saved weights of the CNN. 
train1 is for detection CNN.
train2 and train2v2 are for classification CNN
train2 is trained on unmodified dataset.
train2v2 is trained on augmented (using randon transformations, perspectives, etc)dataset. 


