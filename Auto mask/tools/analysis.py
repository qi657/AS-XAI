import math
import torch.nn.functional as F

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from ..util.helpers import makedir, find_high_activation_crop
from ..core.train_and_test import *
from ..core import train_and_test as tnt
from ..util.log import create_logger
from ..util.preprocess import mean, std, undo_preprocess_input_function

from skimage import filters, io

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/cat_dog_scq/vgg19/_ortest/')  ## 2push0.9964.pth
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19/cluster_test/') ## 2push1.0000.pth
parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19/opl_test/') ## 2push1.0000.pth
# parser.add_argument('-modeldir',nargs=1, type=str, default='D:/TesNet-1/saved_models/cat_vs_dog_scq/vgg19/hsvtest/') ## 3push1.0000.pth
# parser.add_argument('-modeldir',nargs=1, type=str, default='D:/TesNet-1/saved_models/cat_vs_dog_scq_2/vgg19or/test/') ## 6push1.0000.pth
parser.add_argument('-model', nargs=1, type=str, default='2push1.0000.pth')  # 2push1.0000.pth,2push0.9969.pth,6push1.0000.pth
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/lion_tiger/vgg19opl_filter_725/test/')
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/lion_tiger/vgg19725/test/')
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/zebra donkey/vgg19815/test/')
# parser.add_argument('-model', nargs=1, type=str, default='6push1.0000.pth')
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/mule/vgg19opl_filter_726/test/')  # D:/TesNet-1/saved_models/cat_dog_scq/vgg19/_ortest/, opl_clustertest
# parser.add_argument('-model', nargs=1, type=str, default='5push1.0000.pth')  # 2push0.9964.pth
# parser.add_argument('-modeldir', nargs=1, type=str, default='D:/TesNet-1/saved_models/tapir/vgg19opl_filter_731/test/')  # D:/TesNet-1/saved_models/cat_dog_scq/vgg19/_ortest/, opl_clustertest
# parser.add_argument('-model', nargs=1, type=str, default='4push0.9389.pth')  # 2push0.9964.pth
parser.add_argument('-imgdir', nargs=1, type=str, default="./test_img/Counterfactual") # ./test_img/tiger_lion/, D:/datasets/attack/test_1/dog
parser.add_argument('-img', nargs=1, type=str, default='cat_1.jpg')  # Lazuli_Bunting_0001_14916.jpg, n02085782_23.jpg
parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]


# specify the test image to be analyzed


image_dir = args.imgdir
image_name = args.img
image_label = 1 # 14, 28
test_image_dir = image_dir
test_image_name = image_name
test_image_label = image_label

test_image_path = os.path.join(test_image_dir, test_image_name)


# load the model
check_test_accu = True

model_dir_list = [args.modeldir]
model_name_list = [args.model]

for model_dir,model_name in zip(model_dir_list,model_name_list):
    load_model_dir = model_dir
    load_model_name = model_name

    #if load_model_dir[-1] == '/':
    #    model_base_architecture = load_model_dir.split('/')[-3]
    #    experiment_run = load_model_dir.split('/')[-2]
    #else:
    #    model_base_architecture = load_model_dir.split('/')[-2]
    #    experiment_run = load_model_dir.split('/')[-1]


    model_base_architecture = load_model_dir.split('/')[-3]
    experiment_run = load_model_dir.split('/')[-2]

    save_analysis_path = os.path.join(test_image_dir, model_base_architecture,experiment_run, load_model_name)
    print(save_analysis_path)
    makedir(save_analysis_path)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name)
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    start_epoch_number = int(epoch_number_str)

    log('load model from ' + load_model_path)
    log('model base architecture: ' + model_base_architecture)
    log('experiment run: ' + experiment_run)

    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)


    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    class_specific = True

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    # load the test data and check test accuracy
    from ..config.settings_cat_dog import test_dir
    if check_test_accu:
        test_batch_size = 1

        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True,
            num_workers=0, pin_memory=False)
        log('test set size: {0}'.format(len(test_loader.dataset)))

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=print)
        print('=====================')
        print('accu:', accu[0])

        exit(0)

    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(load_model_dir, 'img')

    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
    prototype_img_identity = prototype_info[:, -1]

    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')

    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
        undo_preprocessed_img = undo_preprocess_input_function(img_copy)
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0]
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])

        plt.imsave(fname, undo_preprocessed_img)
        return undo_preprocessed_img

    def save_prototype(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_new_act(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-new_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_new_mask(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-new_mask'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_new_round(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-self_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)


    def save_prototype_self_activation(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                        'prototype-img-original_with_self_act'+str(index)+'.png'))
        # plt.imshow(p_img)
        # plt.show()
        #plt.axis('off')
        plt.imsave(fname, p_img)


    def save_prototype_original_img_with_bbox(fname, epoch, index,
                                              bbox_height_start, bbox_height_end,
                                              bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
        cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                      color, thickness=-1)  # thickness = -1 为mask
        p_img_rgb = p_img_bgr[...,::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255
        plt.imshow(p_img_rgb)
        plt.axis('off')
        plt.imsave(fname, p_img_rgb)

    def save_prototype_original_img(fname, epoch, index,
                                              bbox_height_start, bbox_height_end,
                                              bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))

        p_img_rgb = p_img_bgr[...,::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255

        plt.imsave(fname, p_img_rgb)

    def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                         bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                      color, thickness=-1)
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        #plt.imshow(img_rgb_float)
        #plt.axis('off')
        plt.imsave(fname, img_rgb_float)


    # 定义颜色通道转换
    def color_space_transform_rgb(img):
        img_np = img.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
        img_np = img_np.transpose((1, 2, 0))  # 将颜色通道放到最后一个维度
        img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB)  # 将颜色空间从 HSV 转换为 RGB
        img_np = img_np.transpose((2, 0, 1))  # 将颜色通道放回第一个维度
        img = torch.from_numpy(img_np)  # 将 NumPy 数组转换为 PyTorch 张量
        return img

    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    def sobel(img):
        height = img.shape[0]
        width = img.shape[1]
        tmp_img = img.copy()
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                tmpx = np.sum(np.sum(gx * img[i - 1:i + 2, j - 1:j + 2]))
                tmpy = np.sum(np.sum(gy * img[i - 1:i + 2, j - 1:j + 2]))
                tmp_img[i, j] = np.sqrt(tmpx ** 2 + tmpy ** 2)
        return tmp_img


    def generate_irregular_mask(activation_map, threshold):
        # gray_value = (192, 192, 192)  # RGB值为十六进制灰色，这里设置为中灰色
        # 创建二值掩码
        mask = np.zeros_like(activation_map)
        mask[activation_map >= threshold] = 255
        mask = np.uint8(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        irregular_mask = np.zeros_like(activation_map)

        cv2.drawContours(irregular_mask, contours, -1, 255, thickness=cv2.FILLED)

        return irregular_mask


    # load the test image and forward it through the network
    preprocess = transforms.Compose([
       transforms.Resize((img_size,img_size)),
       transforms.ToTensor(),
       normalize
    ])

    img_pil = Image.open(test_image_path)

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    images_test = img_variable.cuda()
    labels_test = torch.tensor([test_image_label])


    project_distances,cosine_distances = ppnet.prototype_distances(images_test)
    prototype_activations = ppnet.global_max_pooling(project_distances)


    logits, min_distances, _ = ppnet_multi(images_test)
    conv_output, distances, _ = ppnet.push_forward(images_test)
    prototype_activation_patterns = -distances


    tables = []
    for i in range(logits.size(0)):
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        log(str(i) + ' ' + str(tables[-1]))

    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))


    original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                         images_test, idx)

    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])

    if False:
        for i in range(0,50):
            log('top {0} activated prototype for this image:'.format(i))
            # 与激活位置相似的原型区域
            save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                        'top-%d_activated_prototype.png' % i),
                           start_epoch_number, sorted_indices_act[-i].item())

            # save_prototype_new_act(os.path.join(save_analysis_path, 'most_activated_prototypes',
            #                             'top-%d_activated_prototype_new_act.png' % i),
            #                start_epoch_number, sorted_indices_act[-i].item())

            # save_prototype_new_mask(os.path.join(save_analysis_path, 'most_activated_prototypes',
            #                             'top-%d_activated_prototype_new_mask.png' % i),
            #                start_epoch_number, sorted_indices_act[-i].item())
            # save_prototype_new_round(os.path.join(save_analysis_path, 'most_activated_prototypes',
            #              'top-%d_activated_prototype_new_round.png' % i),
            #                 start_epoch_number, sorted_indices_act[-i].item())


            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                     'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                  epoch=start_epoch_number,
                                                  index=sorted_indices_act[-i].item(),
                                                  bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                                  bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                                  bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                                  bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                                  color=(0, 255, 255))
            save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                           start_epoch_number, sorted_indices_act[-i].item())
            log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
            log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
            if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
            log('activation value (similarity score): {0}'.format(array_act[-i]))
            log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))

            activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()

            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC) # 上采样到与输入图像相同大小



            # 假设你已经有了激活图 activation_map 和原始图像 original_image
            activation_map = upsampled_activation_pattern  # 激活图
            original_image = original_img  # 原始图像

            percentile = 96  # 阈值
            threshold = np.percentile(activation_map, percentile)

            # 生成不规则掩码
            irregular_mask = generate_irregular_mask(activation_map, threshold)
            irregular_mask_1 = generate_irregular_mask(activation_map, threshold)
            irregular_mask_2= generate_irregular_mask(activation_map, threshold)
            irregular_mask_3 = generate_irregular_mask(activation_map, threshold)
            # plt.imshow(irregular_mask)
            # plt.show()

            # 应用不规则掩码到原始图像
            masked_image_1 = np.copy(original_image)
            masked_image_1[irregular_mask_1 == 0] = 0  # 黑白的高激活区域（用来做mask）

            alpha = 0.3  # 透明度参数，调整透明度的程度

            masked_image_2 = np.copy(original_image)
            mask_indices = np.where(irregular_mask_2 != 0)  # mask的索引

            # 将mask以外的像素设置为更透明
            masked_image_2[mask_indices] = masked_image_2[mask_indices] * (1 - alpha)

            # 将mask以内的像素设置为更清晰
            masked_image_2[irregular_mask_2 != 0] = masked_image_2[irregular_mask_2 != 0] * (1 - alpha) + \
                                                    original_image[irregular_mask_2 != 0] * alpha

            masked_image_2 = np.copy(original_image)
            masked_image_2[irregular_mask_2 == 0] = masked_image_2[irregular_mask_2 == 0] * 0.7  # 调整亮度值   可视化（贴图用）

            masked_image_3 = np.copy(original_image)
            masked_image_3[irregular_mask_3 == 255] = [255,255,0]

            # 将不规则掩码转换为二值图像
            irregular_mask_binary = np.uint8(irregular_mask_2 > 0)

            # 绘制不规则掩码边缘
            contours, _ = cv2.findContours(irregular_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 将 masked_image 转换为 UMat 类型
            masked_image_umat = cv2.UMat(masked_image_2)

            # 绘制不规则掩码边缘
            cv2.drawContours(masked_image_umat, contours, -1, (255, 0, 0), 0)

            # 将 masked_image_umat 转换回 ndarray 类型
            masked_image_2 = cv2.UMat.get(masked_image_umat)


            # 可视化结果
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axes[0].imshow(activation_map, cmap='hot')
            axes[0].set_title('Activation Map')
            axes[0].axis('off')

            axes[1].imshow(masked_image_1)
            axes[1].set_title('Masked Image1')
            axes[1].axis('off')

            axes[2].imshow(masked_image_2)
            axes[2].set_title('Masked Image2')
            axes[2].axis('off')

            # plt.show()


            masked_image_2 = cv2.cvtColor(masked_image_2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                     'most_highly_activated_patch_by_top-%d_prototype_vis.png' % i), masked_image_2 * 255)
            cv2.imwrite(
                os.path.join(save_analysis_path, 'most_activated_prototypes',
                                     'most_highly_activated_patch_by_top-%d_prototype_mask.png' % i),masked_image_3[:, :, [2, 1, 0]] * 255)



            # show the most highly activated patch of the image by this prototype  显示了该原型激活程度最高的图像补丁
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)


            # 激活之后的patch
            # high_act_patch = original_img[prototype_info[sorted_indices_act[-i].item()][1]:prototype_info[sorted_indices_act[-i].item()][2].
            #                             prototype_info[sorted_indices_act[-i].item()][3]:prototype_info[sorted_indices_act[-i].item()][4],:]
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                          high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % i), masked_image_1)

            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map  显示与原型激活图重叠的图像
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]

            # plt.imshow(heatmap)
            # plt.show()

            overlayed_img = 0.5 * original_img + 0.3 * heatmap

            # plt.imshow(overlayed_img)
            # plt.show()

            log('prototype activation map of the chosen image:')
            plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'prototype_activation_map_by_top-%d_prototype.png' % i),
                       overlayed_img)
            log('--------------------------------------------------------------')

            prototype_img = io.imread(os.path.join(load_img_dir, 'epoch-' + str(start_epoch_number), 'prototype-img' + str(sorted_indices_act[-i].item()) + '.png'), as_gray=True)
            print(prototype_img.shape)


    # exit(0)

    ##### PROTOTYPES FROM TOP-k CLASSES
    global_score = 0
    k = 2
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k)
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

        log('top %d predicted class: %d' % (i+1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_cnt = 1

        # for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        # if i == 0:
        #     ## 1_eye
        #     index = (19, 120, 59, 90, 75, 135, 168, 74, 141, 40, 138, 4, 35, 149, 67, 173, 107, 36, 15, 129, 185, 194, 150)
        #     ## 2_eye
        #     index = (4, 99, 47, 57, 49, 36, 76, 103, 19, 32, 18, 54, 28, 63, 77)  # or_test
        #     ## 3_eye
        #     index = (24, 106, 76, 84, 99, 61, 55, 117, 14, 49, 118, 111, 66, 104, 47, 30, 59, 119) # or_test
        #     ## 4_eye
        #     # index = (94,46,16,64,40,76,29,65,54,58,55,66,0,53,27,83,19,96,82,75,4,14,87,69)
        #     ## 5_eye
        #     # index = (53,75,63,62,57,25,73,113,16,22,59,35,69,46,93,71,27,1,112,6,54,33,18,83,99,34,50,61,51,107,91,87)
        #     ## cats_eye
        #     index = (
        #     49, 72, 47, 498, 115, 68, 325, 27, 144, 323, 407, 244, 288, 25, 431, 390, 251, 113, 123, 419, 247, 337, 335,
        #     425, 488, 156, 160, 217, 201, 272, 34, 118, 454, 262, 82, 231, 399, 119, 198, 315, 462, 154, 461, 79, 56,
        #     357, 499, 332, 147, 142, 237, 405, 389, 292)
        #     ## 1_ear
        #     index = (176, 50, 122, 190, 154, 5, 148, 146, 100, 23) # oplcluster

        #     ## 1_leg
        #     index = (195, 184, 196, 31, 139, 167, 145, 179, 20, 197, 158, 177, 109, 76, 101, 73, 162, 103, 45, 187)  # oplcluster

        # elif i == 1:
            # index = (18, 28, 35, 54, 36, 58, 4, 57, 104, 103, 47) # or_test_eye
            # index = (82, 165, 23, 151, 74, 108, 32, 9, 5, 122, 110, 84, 49, 174, 105, 136, 69, 76, 35, 127) # opl_cluster_eye
            # index = (99, 189, 66, 81, 95, 67, 65, 147, 126, 10, 138) # opl_cluster_ear
            # index = (122, 135, 78, 11, 181, 51) # opl_cluster_leg
        # elif i == 2:
            # index = (30, 71, 55, 61, 106, 117, 118, 99, 104, 38, 47, 119)  # or_test_eye
            # index = (18, 142, 149, 36, 101, 47, 24, 80, 71, 44, 151, 59, 189, 174, 11, 158, 92, 164, 188, 167, 159, 40, 5, 141, 134, 138) # opl_cluster_eye
            # index = ( 84, 126, 176, 187, 29, 33, 181, 1)  # opl_cluster_ear
            # index = (170, 7, 104, 38, 65, 125, 183, 195, 182, 86, 34, 147, 103, 169) # opl_cluster_leg
        # elif i == 3:
            # index = (23, 20, 103, 107, 116, 54, 100, 117, 24, 79, 95, 91)  # or_test
            # index = (129, 170, 82, 13, 127, 48, 98, 189, 124, 31) # opl_cluster_eye
            # index = (126, 119, 191, 184, 38, 85, 62, 158, 172) # opl_cluster_ear
            # index = (134, 71, 110, 117, 97, 190, 174, 125, 108, 136, 32, 9, 135, 180) # opl_cluster_leg
        # elif i == 4:
            # index = (21, 86, 7, 70, 112, 75, 25, 47, 119, 12, 87, 26, 88, 118, 53, 10, 36, 50, 102) # or_test
            # index = (151, 41, 66, 146, 172, 166, 67, 48, 138, 50, 136, 130, 99, 71, 2, 106) # opl_cluster_eye
            # index = (14, 13, 177, 176, 143, 144, 44, 62, 189, 152, 188, 116) # opl_cluster_ear
            # index = (1, 145, 184, 121, 175, 88, 9, 173, 64, 53) # opl_cluster_leg
        # else:
        _, sorted_indices_cls_act_1 = torch.sort(class_prototype_activations, descending=True)
        index = sorted_indices_cls_act_1.detach().cpu().numpy()

        # for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        for j in index:
            prototype_index = class_prototype_indices[j]

            save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'top-%d_activated_prototype.png' % prototype_cnt),
                           start_epoch_number, prototype_index)

            # save_prototype_new_act(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
            #                                     'top-%d_activated_prototype_new_act.png' % prototype_cnt),
            #                        start_epoch_number, prototype_index)
            #
            # save_prototype_new_mask(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
            #                                      'top-%d_activated_prototype_new_mask.png' % prototype_cnt),
            #                         start_epoch_number, prototype_index)
            # save_prototype_new_round(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
            #                                       'top-%d_activated_prototype_new_round.png' % prototype_cnt),
            #                          start_epoch_number, prototype_index)

            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                     'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                  epoch=start_epoch_number,
                                                  index=prototype_index,
                                                  bbox_height_start=prototype_info[prototype_index][1],
                                                  bbox_height_end=prototype_info[prototype_index][2],
                                                  bbox_width_start=prototype_info[prototype_index][3],
                                                  bbox_width_end=prototype_info[prototype_index][4],
                                                  color=(0, 255, 255))
            save_prototype_original_img(
                fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                   'top-%d_activated_in_original_pimg.png' % prototype_cnt),
                epoch=start_epoch_number,
                index=prototype_index,
                bbox_height_start=prototype_info[prototype_index][1],
                bbox_height_end=prototype_info[prototype_index][2],
                bbox_width_start=prototype_info[prototype_index][3],
                bbox_width_end=prototype_info[prototype_index][4],
                color=(0, 255, 255)
            )

            save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                        'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                           start_epoch_number, prototype_index)
            log('prototype index: {0}'.format(prototype_index))
            log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))

            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()

            original_img_1 = cv2.resize(original_img, (7, 7), interpolation=cv2.INTER_AREA)

            rescaled_activation_pattern = activation_pattern - np.amin(activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(activation_pattern)  # 均值处理
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap_1 = heatmap[..., ::-1]
            overlayed_img_1 = 0.5 * original_img_1 + 0.3 * heatmap_1

            # plt.imshow(heatmap_1)
            # plt.savefig('./test_img/heatmap-%d.jpg'%prototype_cnt, pad_inches=0)
            # plt.show()



            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC)


            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)

            if False:
                # new code
                proto_height_dis = prototype_info[prototype_index][2]-prototype_info[prototype_index][1]
                proto_width_dis = prototype_info[prototype_index][4]-prototype_info[prototype_index][3]
                img_height_dis = high_act_patch_indices[1]-high_act_patch_indices[0]
                img_width_dis = high_act_patch_indices[3]-high_act_patch_indices[2]
                height_dlta = proto_height_dis - img_height_dis
                width_dlta = proto_width_dis - img_width_dis

                high_patch_indices_begin = high_act_patch_indices[0]
                high_patch_indices = high_act_patch_indices[1]

                if proto_height_dis < img_height_dis:
                    high_patch_indices = high_act_patch_indices[0] + proto_height_dis
                else:
                    high_patch_indices = high_act_patch_indices[1] +height_dlta
                    # high_patch_indices = high_act_patch_indices[1] + height_dlta//2
                    # high_patch_indices_begin = high_act_patch_indices[0] - height_dlta//2+1

                height_new_dis = high_patch_indices - high_patch_indices_begin

                if height_new_dis < proto_height_dis:
                    dlta = proto_height_dis - height_new_dis
                    high_patch_indices_begin = high_act_patch_indices[0] - dlta
                else:
                    high_patch_indices_begin = high_act_patch_indices[0]

                if proto_width_dis < img_width_dis:
                    width_patch_indices = high_act_patch_indices[2] + proto_width_dis
                else:
                    width_patch_indices = high_act_patch_indices[3] + width_dlta

                width_patch_indices_begin = high_act_patch_indices[2]
                width_new_dis = width_patch_indices - width_patch_indices_begin

                if width_new_dis < proto_width_dis:
                    dlta = proto_width_dis - width_new_dis
                    width_patch_indices_begin = high_act_patch_indices[2] - dlta
                else:
                    width_patch_indices_begin = high_act_patch_indices[2]

            # # 假设你已经有了激活图 activation_map 和原始图像 original_image
            # activation_map = upsampled_activation_pattern  # 激活图
            # original_image = original_img  # 原始图像
            #
            # percentile = 96  # 阈值
            # threshold = np.percentile(activation_map, percentile)
            #
            # # 生成不规则掩码
            # irregular_mask = generate_irregular_mask(activation_map, threshold)
            # irregular_mask_1 = generate_irregular_mask(activation_map, threshold)
            # irregular_mask_2= generate_irregular_mask(activation_map, threshold)
            # irregular_mask_3 = generate_irregular_mask(activation_map, threshold)
            # # irregular_mask_4 = generate_irregular_mask(activation_map, threshold)
            #
            #
            # # 应用不规则掩码到原始图像
            # masked_image_1 = np.copy(original_image)
            # # masked_image_1[irregular_mask_1 == 0] = 0  # 黑白的高激活区域（用来做mask）
            # masked_image_1[irregular_mask_1 == 0] = 0.8 # mask以外的区域变成灰色
            #
            #
            # brightness_factor = 1.2
            # darkness_factor = 0.2
            # masked_image_2 = np.copy(original_image)
            # masked_image_2[irregular_mask_2 == 0] = masked_image_2[irregular_mask_2 == 0] * darkness_factor  # mask以外的部分亮度低
            # mask_indices = np.where(irregular_mask_2 != 0)
            # masked_image_2[mask_indices] = masked_image_2[mask_indices] * brightness_factor  # mask以内的部分亮度高
            #
            # # new code
            # irregular_mask_binary = np.uint8(irregular_mask_2 > 0)
            # contours, _ = cv2.findContours(irregular_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # bounding_box = cv2.boundingRect(contours[0])
            # x, y, w, h = bounding_box
            # # 计算边界框的中心点
            # center_x, center_y = x + w // 2, y + h // 2
            # # 将mask以外的部分置为透明
            # masked_image_2[irregular_mask_1 == 0] = 0.9
            # # 等比例放大mask内部的图像
            # scale_factor = 2  # 假设放大倍数为2
            # resized_image = cv2.resize(masked_image_2[y:y + h, x:x + w], None, fx=scale_factor, fy=scale_factor)
            #
            # h_resized, w_resized, _ = resized_image.shape
            # # x1, x2 = center_x - w_resized // 2, center_x + w_resized // 2
            # # y1, y2 = center_y - h_resized // 2, center_y + h_resized // 2
            # #
            # # x1, x2 = max(center_x - w_resized // 2, 0), max(center_y - h_resized // 2, 0)
            # # y1, y2 = min(center_x + w_resized // 2, masked_image_2.shape[1]), min(center_y + h_resized // 2, masked_image_2.shape[0])
            #
            # x1_resized, y1_resized = max(center_x - w_resized // 2, 0), max(center_y - h_resized // 2, 0)
            # x2_resized, y2_resized = min(center_x + w_resized // 2, masked_image_2.shape[1]), min(center_y + h_resized // 2, masked_image_2.shape[0])
            #
            # h_resized_inside = y2_resized - y1_resized
            # w_resized_inside = x2_resized - x1_resized
            # resized_image = np.clip(resized_image, 0, 1)
            # # if h_resized_inside == h_resized and w_resized_inside == w_resized:
            # masked_image_2[y1_resized:y2_resized, x1_resized:x2_resized] = resized_image
            # masked_image_2 = np.clip(masked_image_2, 0, 1)
            #
            #
            # # masked_image_2[y1:y2, x1:x2] = resized_image
            # # target_region = masked_image_2[y1:y1 + h_resized, x1:x1 + w_resized]
            # # h_resized_inside = y2 - x2
            # # w_resized_inside = y1 - x1
            # # target_region = masked_image_2[x2:y2, x1:y1]
            # # target_region = np.clip(target_region, 0, 1)
            #
            # # if target_region.shape[:2] != resized_image.shape[:2]:
            # #     resized_image = cv2.resize(resized_image, (w_resized, h_resized))
            # # if (y1+h_resized <= 224 and x1+w_resized <=224) or (y1+h_resized <= 224 or x1+w_resized <=224):
            # #     masked_image_2[y1:y1 + h_resized, x1:x1 + w_resized] = resized_image
            # # elif y1+h_resized > 224 and x1+w_resized <=224:
            # #     h_new = 224
            # #     masked_image_2[y1:h_new, x1:x1 + w_resized] = resized_image
            # # elif y1+h_resized <= 224 and x1+w_resized > 224:
            # #     w_new = 224
            # #     masked_image_2[y1:h_resized, x1:x1 + w_new] = resized_image
            # # else:
            # #     h_new = 224
            # #     w_new = 224
            # #     masked_image_2[y1:h_new, x1:x1 + w_new] = resized_image
            #
            # brightness_factor = 1.2
            # darkness_factor = 0.2
            # masked_image_4 = np.copy(original_image)
            # masked_image_4[irregular_mask_4 == 0] = masked_image_4[irregular_mask_4 == 0] * darkness_factor  # mask以外的部分亮度低
            # mask_indices = np.where(irregular_mask_4 != 0)
            # masked_image_4[mask_indices] = masked_image_4[mask_indices] * brightness_factor  # mask以内的部分亮度高
            # # 等比例放大不规则边缘的轮廓点
            # scaled_contours = [cv2.approxPolyDP(cnt, scale_factor, True) for cnt in contours]
            #
            # # 在当前放大后的 masked_image_2 上绘制不规则边缘
            # masked_image_4_with_contours = np.copy(masked_image_4)
            # masked_image_4_with_contours_umat = cv2.UMat(masked_image_4_with_contours)
            # cv2.drawContours(masked_image_4_with_contours_umat, scaled_contours, -1, (255, 0, 0), 2)
            # masked_image_4 = cv2.UMat.get(masked_image_4_with_contours_umat)
            #
            #
            # masked_image_3 = np.copy(original_image)
            # masked_image_3[irregular_mask_3 == 255] = [255, 255, 0]
            #
            #
            # # 可视化结果
            # fig, axes = plt.subplots(1, 4, figsize=(10, 5))
            # axes[0].imshow(activation_map, cmap='hot')
            # axes[0].set_title('Activation Map')
            # axes[0].axis('off')
            #
            # axes[1].imshow(masked_image_1)
            # axes[1].set_title('Masked Image1')
            # axes[1].axis('off')
            #
            # axes[2].imshow(masked_image_2)
            # axes[2].set_title('Masked Image2')
            # axes[2].axis('off')
            #
            # axes[3].imshow(masked_image_4)
            # axes[3].set_title('Masked Image4')
            # axes[3].axis('off')
            # plt.show()
            # exit(0)

            # 假设你已经有了激活图 activation_map 和原始图像 original_image
            activation_map = upsampled_activation_pattern  # 激活图
            original_image = original_img  # 原始图像

            percentile = 95  # 阈值
            threshold = np.percentile(activation_map, percentile)

            # 生成不规则掩码
            irregular_mask = generate_irregular_mask(activation_map, threshold)
            irregular_mask_1 = generate_irregular_mask(activation_map, threshold)
            irregular_mask_2 = generate_irregular_mask(activation_map, threshold)
            irregular_mask_3 = generate_irregular_mask(activation_map, threshold)

            # 应用不规则掩码到原始图像
            masked_image_1 = np.copy(original_image)
            masked_image_1[irregular_mask_1 == 0] = 0.8  # 黑白的高激活区域（用来做mask）

            brightness_factor = 1
            darkness_factor = 0.5
            masked_image_2 = np.copy(original_image)
            masked_image_2[irregular_mask_2 == 0] = masked_image_2[
                                                        irregular_mask_2 == 0] * darkness_factor  # mask以外的部分亮度低
            mask_indices = np.where(irregular_mask_2 != 0)
            masked_image_2[mask_indices] = masked_image_2[mask_indices] * brightness_factor  # mask以内的部分亮度高

            masked_image_3 = np.copy(original_image)
            masked_image_3[irregular_mask_3 == 255] = [255, 255, 0]

            # 将不规则掩码转换为二值图像
            irregular_mask_binary = np.uint8(irregular_mask_2 > 0)

            # 绘制不规则掩码边缘
            contours, _ = cv2.findContours(irregular_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 将 masked_image 转换为 UMat 类型
            masked_image_umat = cv2.UMat(masked_image_2)

            # 绘制不规则掩码边缘
            cv2.drawContours(masked_image_umat, contours, -1, (255, 0, 0), 2)

            # 将 masked_image_umat 转换回 ndarray 类型
            masked_image_2 = cv2.UMat.get(masked_image_umat)

            # 可视化结果
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axes[0].imshow(activation_map, cmap='hot')
            axes[0].set_title('Activation Map')
            axes[0].axis('off')

            axes[1].imshow(masked_image_1)
            axes[1].set_title('Masked Image1')
            axes[1].axis('off')

            axes[2].imshow(masked_image_2)
            axes[2].set_title('Masked Image2')
            axes[2].axis('off')
            # plt.show()
            # exit(0)


            # masked_image_4 = cv2.cvtColor(masked_image_4, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
            #                         'most_highly_activated_patch_by_top-%d_prototype_vis.png' % prototype_cnt), masked_image_4 * 255)

            masked_image_2 = cv2.cvtColor(masked_image_2, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                     'most_highly_activated_patch_by_top-%d_prototype_vis.png' % prototype_cnt), masked_image_2 * 255)

            cv2.imwrite(
                os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                    'most_highly_activated_patch_by_top-%d_prototype_mask.png' % prototype_cnt), masked_image_3[:, :, [2, 1, 0]] * 255)

            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                             high_act_patch_indices[2]:high_act_patch_indices[3], :]

            log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                       masked_image_1)

            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
                                    'most_highly_activated_patch_by_top-%d_or_prototype.png' % prototype_cnt),
                       high_act_patch)


            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]


            # plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i + 1),
            #                         'prototype_heatmap_by_top-%d_prototype.png' % prototype_cnt),
            #            heatmap)

            overlayed_img = 0.5 * original_img + 0.3 * heatmap

            log('prototype activation map of the chosen image:')
            plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                       overlayed_img)
            log('--------------------------------------------------------------')
            prototype_cnt += 1

            global_score += (prototype_activations[idx][prototype_index] * ppnet.last_layer.weight[c][prototype_index])
            print(global_score)
        global_score = global_score/len(index)
        log('***************************************************************')
        log('global score: {0}'.format(global_score))


    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()

