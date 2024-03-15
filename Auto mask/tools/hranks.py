import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import copy
import argparse
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets
from ..util.helpers import makedir


parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--dataset',
    type=str,
    default='cat_vs_dog_scq',
    choices=('cifar10', 'cat_vs_dog_scq','oxford-flowers','flower', 'leave', 'CottonWeedID15'),
    help='dataset')

args = parser.parse_args()

test_image_path = os.path.join("./test_img/dog_1.jpg")
preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5),
                         std=(0.5,0.5,0.5))
    ])
# img_pil = Image.open(test_image_path)
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
# images_test = img_variable.cuda()

data_path = "D:/datasets/cat_vs_dog_scq/"
test_dir = data_path + 'test_1/'

# data_path = 'D:/datasets/oxford_102_flower_dataset/oxford_flower/'
# test_dir = data_path + 'test_1/'

# data_path = 'D:/datasets/flowers_scq/'
# test_dir = data_path + 'test_1/'

# data_path = 'D:/datasets/leaves_scq/'
# test_dir = data_path + 'test/'

# data_path = 'D:/datasets/CottonWeedID15/'
# test_dir = data_path + 'test_1/'

test_dataset = datasets.ImageFolder(
            test_dir,
            preprocess)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=False)  # shuffle=True

def undo_preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y
def undo_preprocess_input_function(x):
    '''
    allocate new tensor like x and undo the normalization used in the
    pretrained model
    '''
    return undo_preprocess(x, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
def save_preprocessed_img(preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])

    plt.imshow(undo_preprocessed_img)
    # plt.show()
    return undo_preprocessed_img


ppnet = torch .load('D:/TesNet-1/saved_models/cat_dog_scq/vgg19/_ortest/2push0.9964.pth')
# ppnet = torch.load('D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19/cluster_test/3push1.0000.pth')  # cat_and_dog_2
# ppnet = torch.load('D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19/opl_test/3push0.9969.pth')  # cat_dog_filter
# ppnet = torch.load('D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19or/test/4push1.0000.pth')
# ppnet = torch.load('./saved_models/oxford_5_flower/vgg19name/test/10push0.7931.pth')  # oxford_5_flower
# ppnet = torch.load('./saved_models/flower_scq/vgg19_61/test/10push0.9803.pth')  # flower
# ppnet = torch.load('./saved_models/leaves_scq/vgg19_61/test/20push0.9429.pth')  # leave
# ppnet = torch.load('./saved_models/CottonWeedID15/vgg19/opltest/10nopush0.8695.pth')  # cottonweed
model = ppnet.cuda()
model.eval()

# conv_output, distances, _ = ppnet.push_forward(images_test)
# sub_model = nn.Sequential(*list(ppnet.features.children())[0][:34])



def generate_irregular_mask(activation_map, threshold1):
    mask = np.zeros_like(activation_map)
    mask[(activation_map >= threshold1)] = 255
    mask = np.uint8(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    irregular_mask = np.zeros_like(activation_map)

    cv2.drawContours(irregular_mask, contours, -1, 255, thickness=cv2.FILLED)

    return irregular_mask

def visualize_feature_map(feature_map):
    plt.imshow(feature_map, cmap='gray')
    plt.axis('off')
    # plt.show()


prototype_cnt = 0
for j, data in enumerate(test_loader):
    # if i<19:
    #     continue
    print('----------第%d次迭代----------' % (j))
    image, label = data

    sub_model = nn.Sequential(nn.Sequential(*list(ppnet.features.children()))[0], ppnet.add_on_layers[:2]).cuda()
    conv_output = sub_model(image.cuda())

    original_img = save_preprocessed_img(image, 0)

    for i in range(64):
        feature_map_1 = conv_output[:,i,:,:][0,:,:].cpu().detach().numpy()
        visualize_feature_map(feature_map_1)
        img = cv2.resize(feature_map_1, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        rescaled_activation_pattern = img - np.amin(img)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)

        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]

        img_now = 0.5*original_img+0.3*heatmap

        percentile1 = 95  
        threshold1 = np.percentile(rescaled_activation_pattern, percentile1)

        irregular_mask = generate_irregular_mask(rescaled_activation_pattern, threshold1)
        masked_image_1 = np.copy(original_img)
        masked_image_1[irregular_mask == 0] = 0
        # masked_image_1[irregular_mask == 0] = masked_image_1[irregular_mask == 0] * 0.7  

        # irregular_mask_binary = np.uint8(irregular_mask > 0)
        #
        # contours, _ = cv2.findContours(irregular_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # masked_image_umat = cv2.UMat(masked_image_1)
        #
        # cv2.drawContours(masked_image_umat, contours, -1, (255, 0, 0), 0)
        #
        # masked_image_1 = cv2.UMat.get(masked_image_umat)

        # plt.imshow(masked_image_1)
        # plt.show()

        makedir(os.path.join(f'./hranks/{args.dataset}_1', 'class-%d' % (j)))
        plt.imsave(os.path.join(f'./hranks/{args.dataset}_1', 'class-%d' % (j),'top-%d.png' % (i)),masked_image_1)

        # masked_image_1 = cv2.cvtColor(masked_image_1, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(os.path.join('./hranks','top-%d.png' % (i)),masked_image_1*255)

        prototype_cnt += 1

