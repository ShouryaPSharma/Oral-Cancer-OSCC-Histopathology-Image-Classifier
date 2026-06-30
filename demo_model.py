import cv2
import numpy as np
import torch
from net_class import Net


def apply_model(path):
    
    print("DONT USE FOR ACTUAL MEDICAL ANALYSIS!!")
    print()
    


    #50 x 50 pixels 
    img_size = 50

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (img_size, img_size))

    img_array = np.array(img)

    img_array = img_array / 255 #normalize the pixel values to be between 0 and 1
    img_array = torch.tensor(img_array, dtype=torch.float32)


    net = Net()
    net.load_state_dict(torch.load("saved_model.pth")) #load the trained model parameters
    net.eval() #set the model to evaluation mode


    net_out = net(img_array.view(-1, 1, img_size,img_size))[0] #input shape is (1, 1, 50, 50) for one grayscale image


    if net_out[0] >= net_out[1]:
        print()
        print()
        print("Prediction : NORMAL")
        print(f"Confidence: {round(float(net_out[0]), 3)}")
        print()
        print()

    else:
        print()
        print()
        print("Prediction : OSCC")
        print(f"Confidence: {round(float(net_out[1]), 3)}")
        print()
        print()
