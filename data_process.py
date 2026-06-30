import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

## dataset:
## https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data

# one-hot vectors
# [1,0] = normal
# [0,1] = OSCC (Oral Squamous Cell Carcinoma)

##we need all images in same dimension (50 pixels x 50 pixels)
img_size = 50

#location of image files
normal_training_folder = "Oral_cancer_dataset/train/Normal/"
oscc_training_folder = "Oral_cancer_dataset/train/OSCC/"

normal_testing_folder = "Oral_cancer_dataset/test/Normal/"
oscc_testing_folder = "Oral_cancer_dataset/test/OSCC/"

normal_training_data = []
oscc_training_data = []

normal_testing_data = []
oscc_testing_data = []


for filename in os.listdir(normal_training_folder):
    try:
        #print(filename)

        path = normal_training_folder + filename
        #print(path) 

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #plt.imshow(img)
        #plt.show()
        #break

        img = cv2.resize(img, (img_size, img_size))

        img_array = np.array(img)
        #print(img_array)
        #print(img_array.shape)
        #break
        # greayscale makes image one number per pixel

        normal_training_data.append([img_array, np.array([1,0])])
        
    except:
        pass


for filename in os.listdir(oscc_training_folder):
    try: 
        #print(filename)

        path = oscc_training_folder + filename
        #print(path) 

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #plt.imshow(img)
        #plt.show()
        #break

        img = cv2.resize(img, (img_size, img_size))

        img_array = np.array(img)
        #print(img_array)
        #print(img_array.shape)
        #break
        # greayscale makes image one number per pixel

        oscc_training_data.append([img_array, np.array([0,1])])
        

    except:
        pass

for filename in os.listdir(normal_testing_folder):
    try: 
        #print(filename)

        path = normal_testing_folder + filename
        #print(path) 

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #plt.imshow(img)
        #plt.show()
        #break

        img = cv2.resize(img, (img_size, img_size))

        img_array = np.array(img)
        #print(img_array)
        #print(img_array.shape)
        #break
        # greayscale makes image one number per pixel

        normal_testing_data.append([img_array, np.array([1,0])])
        
    except:
        pass

for filename in os.listdir(oscc_testing_folder):
    try: 
        #print(filename)

        path = oscc_testing_folder + filename
        #print(path) 

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        #plt.imshow(img)
        #plt.show()
        #break

        img = cv2.resize(img, (img_size, img_size))

        img_array = np.array(img)
        #print(img_array)
        #print(img_array.shape)
        #break
        # greayscale makes image one number per pixel

        oscc_testing_data.append([img_array, np.array([0,1])])
        

    except:
        pass


normal_training_data = normal_training_data[0:len(oscc_training_data)]

print()
print()
print(f"normal training count : {len(normal_training_data)}")
print(f"oscc training count : {len(oscc_training_data)}")
print()
print(f"normal testing count : {len(normal_testing_data)}")
print(f"oscc testing count : {len(oscc_testing_data)}")





training_data = normal_training_data + oscc_training_data
np.random.shuffle(training_data)
np.save("oscc_training_data.npy", np.array(training_data, dtype=object))

testing_data = normal_testing_data + oscc_testing_data
np.random.shuffle(testing_data)
np.save("oscc_testing_data.npy", np.array(testing_data, dtype=object))
