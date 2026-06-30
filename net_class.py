import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

##we need all images in same dimension (50 pixels x 50 pixels)
img_size = 50

class Net(nn.Module):
    
    #costructor
    def __init__(self):
        
        super().__init__()

        self.conv1 = nn.Conv2d(1,32,kernel_size=5) #input 1 channel, output 32 channels, kernel size 5x5
        self.conv2 = nn.Conv2d(32,64,kernel_size=5) #input 32 channels, output 64 channels, kernel size 5x5
        self.conv3 = nn.Conv2d(64,128,kernel_size=5) #input 64 channels, output 128 channels, kernel size 5x5

        
        self.fc1 = nn.Linear(128*2*2,512) #input 128 channels * 3 pixels * 3 pixels, output 512
        self.fc2 = nn.Linear(512,2) #input 512, output 2 (benign or malignant) since we are doing binary classification.

        
    def forward(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)) #max pooling with kernel size 2x2
        #print(f"After conv1: {x.shape}")
       
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        #print(f"After conv2: {x.shape}")

        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        #print(f"After conv3: {x.shape}")

        #sys.exit("trying to get shape for linear layer")


        x = x.view(-1,128*2*2) #flatten the output of the convolutional layers to feed into the fully connected layers

        x= F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.softmax(x) #apply softmax to get probabilities for each class

        return(x)



#net = Net()


#test_img = torch.randn(img_size,img_size).view(-1,1,img_size,img_size) #create a random image of size 50x50
#output = net(test_img)