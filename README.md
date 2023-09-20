# AS-XAI


## Getting Started

**Requirements:** Pytorch, Numpy, cv2, Augmentor

Take the cat and dog as an example.

### Preprocess the datasets

1. Download the datasets from the following two links

- Cat and dog can be downloaded from:
  https://drive.google.com/drive/folders/1pUCEdPoe8S-R5YCkZsJjOqdUhiey1Rzi


2. Crop the images and split the cropped images  by `Auto mask/preprocess_data/cropimages.py`

3. Augment the training set by `./preprocess_data/img_aug.py`

## Step 1
### Train the model

1. Provide a correct path for `data_path, train_dir, test_dir, train_push_dir` in `Auto mask/config/settings_cat_dog.py`
2. Run `python main_cat_dog.py`

### Reasoning process

Run `Auto mask/tools/analysis.py` 

## Step 2
### XAI
1. Compile the different semantic concepts obtained in step 1 by `XAI/cut_position.py`
2. Explaining CNN decision-making by `XAI/row_center_pca.py` (You can perform different functions depending on the comments in the code.)

