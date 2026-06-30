import numpy as np
import torch
from net_class import Net


#50 x 50 pixels
img_size = 50




net = Net()
net.load_state_dict(torch.load("saved_model.pth")) #load the trained model parameters
net.eval() #set the model to evaluation mode

testing_data = np.load("oscc_testing_data.npy", allow_pickle=True)

# for row in training_data:
#    print(row[0]) #should be (50,50)
#    print(row[1]) #should be [1,0] or [0,1]
#    print()
#    print()
#    input()

# putting all the image arrays into this tensor
test_X = torch.tensor( [item[0] for item in testing_data], dtype=torch.float32).view(-1,1,img_size,img_size) #reshape to be (num_samples, 1, 50, 50) where 1 is the number of channels (grayscale)
test_X = test_X / 255 #normalize the pixel values to be between 0 and 1


#for row in test_X:
#    print(row) #should be (50,50)
#    print()
#    input()

# one-hot vector label tensor
test_y = torch.tensor( [item[1] for item in testing_data], dtype=torch.float32) #should be (num_samples, 2) where 2 is the number of classes (benign or malignant)


correct = 0
total = 0


with torch.no_grad(): #no need to calculate gradients during testing

    for i in range(len(test_X)):

        #real label:
        # [0,1]
        # model prediction:(example only)
        # [0.34, 0.66] = 66% chance of being oscc, 34% chance of being normal



        output = net(test_X[i].view(-1,1,img_size,img_size))[0] #reshape to be (1, 1, 50, 50) since we are passing one image at a time

        if output[0] >= output[1]:
            guess = "normal"
        else:
            guess = "OSCC"

        
        real_label= test_y[i] #real label
        

        if real_label[0] >= output[1]:
            real_class = "normal"
        else:
            real_class = "OSCC"

        
        if guess == real_class:
            correct += 1
        
        total += 1


print(f"Accuracy: {correct/total:.3f}") 





        