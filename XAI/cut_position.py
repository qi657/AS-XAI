import os

import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


ppnet = torch.load('../saved_models/cat_vs_dog_scq_2/vgg19or/test/0nopush0.9876.pth')
# ppnet.features = nn.Sequential(*list(ppnet.features.children()))
# ppnet.features.add_module('global average', nn.AvgPool2d(7))
use_model = ppnet.cuda()

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet_multi.module.img_size

# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
normalize = transforms.Normalize(mean=mean,std=std)
test_batch_size = 1

test_dir = 'C:/Users/sunchangqi/Desktop/img/'

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)  # shuffle=True


def cut_position(use_model):
    # Automatic mask semantic features
    segment_save=[]
    origin_save=[]
    for i, data in enumerate(test_loader):
        print('----------第%d次迭代----------' % (i))
        image, label = data  # image:[1,3,512,512]
        img = torchvision.utils.make_grid(image)
        img = img.cpu().data.numpy().transpose((1, 2, 0))

        origin_save.append(image.cpu().data.numpy())

        if len(origin_save) == 10:
            # segment_save = np.array(segment_save)
            origin_save = np.array(origin_save)
            # np.save("../result_save/none_save_lion_origin.npy", origin_save)
            np.save("../result_save/petal_save_Sicklepod.npy", origin_save)
            break


# Extract common local features
cut_position(use_model)
