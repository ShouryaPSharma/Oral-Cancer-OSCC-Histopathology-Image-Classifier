# Oral Cancer (OSCC) Histopathology Image Classifier

A simple Convolutional Neural Network (CNN) built with PyTorch to classify histopathological images as **Normal** or **OSCC (Oral Squamous Cell Carcinoma)**. This project was built for self-learning in computational biology and deep learning, and is **not intended for clinical or diagnostic use**.

## Dataset

[Histopathological Imaging Database for Oral Cancer Analysis](https://www.kaggle.com/datasets/ashenafifasilkebede/dataset/data) (Kaggle, by Ashenafi Fasil Kebede)

The dataset contains microscopic histopathological images of oral tissue, divided into two classes:
- **Normal** — healthy oral tissue
- **OSCC** — tissue with Oral Squamous Cell Carcinoma

It is pre-split into `train` and `test` folders, each containing `Normal` and `OSCC` subfolders. Expected directory structure for this project:

```
Oral_cancer_dataset/
├── train/
│   ├── Normal/
│   └── OSCC/
└── test/
    ├── Normal/
    └── OSCC/
```

## Project Structure & Workflow

| File | Purpose |
|---|---|
| `data_process.py` | Loads images from the dataset folders, converts them to grayscale, resizes to 50×50, one-hot encodes labels (`[1,0]`=Normal, `[0,1]`=OSCC), balances the training set, shuffles, and saves as `oscc_training_data.npy` / `oscc_testing_data.npy`. |
| `net_class.py` | Defines the CNN architecture (`Net` class): 3 convolutional layers (32→64→128 channels, 5×5 kernels) each followed by ReLU + 2×2 max pooling, then 2 fully connected layers ending in a softmax over 2 classes. |
| `model_train.py` | Loads the processed training tensors, normalizes pixel values (0–1), and trains the CNN using Adam optimizer and MSE loss over mini-batches. Saves trained weights to `saved_model.pth`. |
| `model_test.py` | Loads the saved model and test set, runs inference on each image, and prints overall classification accuracy. |
| `demo_model.py` | Standalone inference script — takes a single image path, runs it through the trained model, and prints a Normal/OSCC prediction with confidence. Sample images are provided in `demo_pics/` for quick testing. |

### Usage order
```bash
python data_process.py    # 1. preprocess raw images into .npy files
python model_train.py     # 2. train the CNN
python model_test.py      # 3. evaluate on test set
python demo_model.py      # 4. run inference on a single demo image
```

## Current Results

Test accuracy achieved: **0.548** (essentially close to random guessing for a binary task). This is a baseline/learning model with significant room for improvement.

## Ideas to Improve Accuracy

A few directions worth exploring, roughly in order of likely impact:

- **Train for more epochs**: only 2 epochs are currently used. Try 15–50+ with early stopping based on validation loss.
- **Use a validation split**: currently there's only train/test, with no validation set to monitor overfitting during training.
- **Data augmentation**: random rotations, flips, zoom, and color/contrast jitter (via `torchvision.transforms`) help the model generalize better, especially with a limited histopathology dataset.
- **Increase image resolution**: 50×50 pixels is very small for histopathology detail. Try 128×128 or 224×224 if compute allows.
- **Use color (RGB) instead of grayscale**: staining color (e.g., H&E) often carries diagnostically relevant information that grayscale discards.
- **Add Batch Normalization and Dropout**: helps stabilize training and reduce overfitting.
- **Tune learning rate**: try a learning rate scheduler (`torch.optim.lr_scheduler`) or experiment with different initial rates.


## Disclaimer

This project is for educational purposes only as part of self-learning in computational biology and deep learning. It is **not validated for medical or diagnostic use**.
