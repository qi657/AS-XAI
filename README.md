# AS-XAI
This repository contains code accompanying the paper
> Sun, C., Xu, H., Chen, Y., & Zhang, D. (2024). AS‐XAI: Self‐Supervised Automatic Semantic Interpretation for CNN. Advanced Intelligent Systems, 2400359.

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

## Human Quantitative Evaluation
For detailed design details please see: https://github.com/qi657/AS-XAI-Human-Evaluation

## Reference
If you use this code, our models or data set for your research, please cite [this](https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202400359) publication:
```bibtex
@article{sun2024xai,
  title={AS-XAI: Self-Supervised Automatic Semantic Interpretation for CNN},
  author={Sun, Changqi and Xu, Hao and Chen, Yuntian and Zhang, Dongxiao},
  journal={Advanced Intelligent Systems},
  pages={2400359},
  year={2024},
  publisher={Wiley Online Library}
}
```
