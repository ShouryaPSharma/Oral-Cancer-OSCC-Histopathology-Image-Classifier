import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net_class import Net


#50 x 50 pixels
img_size = 50

training_data = np.load("oscc_training_data.npy", allow_pickle=True)


# for row in training_data:
#    print(row[0]) #should be (50,50)
#    print(row[1]) #should be [1,0] or [0,1]
#    print()
#    print()
#    input()

# putting all the image arrays into this tensor
train_X = torch.tensor( [item[0] for item in training_data], dtype=torch.float32).view(-1,1,img_size,img_size) #reshape to be (num_samples, 1, 50, 50) where 1 is the number of channels (grayscale)
train_X = train_X / 255 #normalize the pixel values to be between 0 and 1


#for row in train_X:
#    print(row) #should be (50,50)
#    print()
#    input()

# one-hot vector label tensor
train_y = torch.tensor( [item[1] for item in training_data], dtype=torch.float32) #should be (num_samples, 2) where 2 is the number of classes (normal or oscc)


net = Net()


optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_function = nn.MSELoss() #mean squared error loss since we are doing regression to the one-hot vectors


batch_size = 100 #how many images are passing through the network at once

epochs = 2

for epoch in range(epochs):

    for i in range(0, len(train_X), batch_size):

        print(f"EPOCH {epoch+1}, fraction complete: {i/len(train_X)}")

        batch_X = train_X[i:i+batch_size].view(-1,1,img_size,img_size) #reshape to be (batch_size, 1, 50, 50)
        batch_y = train_y[i:i+batch_size]

        optimizer.zero_grad() #reset the gradients before backpropagation
        

        outputs = net(batch_X)

        

        loss = loss_function(outputs, batch_y)
        # calculate the loss between the model's predictions and the true labels
        #real label:
        # [0,1]
        # model prediction:(example only)
        # [0.34, 0.66]

        loss.backward() #backpropagation to calculate the gradients of the loss with respect to the model's parameters

        optimizer.step() #update the model's parameters using the calculated gradients




torch.save(net.state_dict(), "saved_model.pth") #save the trained model's parameters to a file