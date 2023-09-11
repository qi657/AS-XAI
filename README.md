# AS-XAI: self-supervised automatic semantic interpretation for cnn

This repository presents the code for the paper "[AS-XAI: self-supervised automatic semantic interpretation for cnn]" .

<img src="./src/intro1.png" style="zoom:50%;" >

Considering the examples above, how would you identify the bird image as a *Yellow-headed Blackbird* and not a *Cape Glossy Starling*? Maybe you find that the bird's head, legs, and feathers look like those concepts of *Yellow-headed Blackbird* rather than *Cape Glossy Starling*. In other words, you may gather the evidence in your mind and make a final decision. Specifically, humans usually explain their reasoning process by dissecting the image into object parts and pointing out the evidence from these identified parts to the concepts stored in his / her mind. Therefore, for the intelligent machine, it is an emergent issue to determine the object parts and construct the concepts in order to implement interpretable image classification.

>**AS-XAI: self-supervised automatic semantic interpretation for cnn**
>
>Changi Sun, Hao Xu, Yuntian Chen, Dongxiao Zhang
>


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

