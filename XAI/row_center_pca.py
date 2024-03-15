import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import time
import numpy as np
import math
import argparse
import scipy
from sklearn.decomposition import PCA
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util.preprocess import mean, std, undo_preprocess_input_function
import scipy.stats as st
from scipy import stats



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_seed', type=int, default=324, help='random seeds for torch')  # 525
    parser.add_argument('--path', type=str, default='D:/datasets/afhq/', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--fig_size', type=int, default=224, help='fig_size')
    parser.add_argument('--TV_beta', type=int, default=2, help='beta for TV')

    parser.add_argument('--model', type=str, default='vgg', help='used model')

    parser.add_argument('--inverse_technique',type=str,default='GD',help='inverse methods including GD,unet')
    parser.add_argument('--layer_type', type=int, default=2, help='0,1,2')
    parser.add_argument('--visual_layer', type=int, default=0, help='visual_layer')
    parser.add_argument('--PCA_main', type=int, default=1, help='selected PCA main')
    parser.add_argument('--PCA_animal', type=str, default='dog', help='dog or cat')
    parser.add_argument('--dataset_situation', type=str, default='val', help='the used dataset train, val or train_dog')
    parser.add_argument('--PCA_data_num', type=int, default=500, help='selected PCA main')
    parser.add_argument('--L2_norm', type=float, default=0, help='selected PCA main')
    parser.add_argument('--position_space', type=str, default='eye', help='selected Semantics of masked: eye, ear, nose, leg')
    parser.add_argument('--position_animal', type=str, default='dog', help='dog or cat')

    # ----------params for GD--------------
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--TV_coef', type=float, default=2, help='beta for TV')
    parser.add_argument('--max_epoch', type=int, default=4000, help='max_epoch')

    parser.add_argument('--visual_layer_start', type=int, default=0, help='start_layer')
    parser.add_argument('--visual_layer_end', type=int, default=1, help='end_layer')
    parser.add_argument('--plot_row', type=int, default=1, help='plot_row')
    parser.add_argument('--plot_col', type=int, default=2, help='plot_col')

    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.std = [0.5, 0.5, 0.5]
    config.mean = [0.5, 0.5, 0.5]
    return config

config= parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


data_path = "D:/datasets/cat_vs_dog_scq/"
test_dir = data_path + 'test_2/'
# data_path = "D:/TesNet-1/test_img/"
# test_dir = data_path + 'counterfactual/1'
# data_path = "D:/datasets/flowers_scq/"
# data_path='D:/datasets/oxford_102_flower_dataset/xford_flower/'
# test_dir = data_path + 'test_1/'
# data_path = "C:/Users/sunchangqi/Desktop/"
# test_dir = data_path + 'img'

normalize = transforms.Normalize(mean=config.mean, std=config.std)

test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                normalize,
            ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=False)  # shuffle=True


font1 = {'family': 'Arial',
         'weight': 'normal',
         # "style": 'italic',
         'size': 7,
         }
font2 = {'family': 'Arial',
         'weight': 'bold',
         # "style": 'italic',
         'size': 20,
         }
legend_font={'family': 'Arial',
         'weight': 'normal',
         # "style": 'italic',
         'size': 6,
         }


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        try:
            self.hook = model[layer_num].register_forward_hook(self.hook_fn)
        except TypeError:
            self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()

def generate_feature_matrix(use_model,conv_out,seed,lime=False):
    use_model.eval()
    use_model.to(config.device)
    print(use_model)
    feature_save = []
    image_best_save = torch.from_numpy(np.load(f'../lime_save/lime_dog_500_324.npy').astype(np.float32))
    image_origin_save = torch.from_numpy(np.load(f'../lime_save/lime_dog_500_origin_324.npy').astype(np.float32))
    if lime == True:
        for i in range(image_best_save.shape[0]):
            print(f'--------第{i}次处理-------')
            if lime==True:
                image= image_best_save[i].reshape([1,3,224,224])
            else:
                image = image_origin_save[i].reshape([1, 3, 224, 224])
            image= Variable(image.to(config.device))
            _, _, _ = use_model(image)
            x_feature = conv_out.features
            feature_save.append(x_feature.data.numpy())
            if i == config.PCA_data_num - 1:
                break
    else:
        for i in range(image_origin_save.shape[0]):
            print(f'--------第{i}次处理-------')
            image = image_origin_save[i].reshape([1, 3, 224, 224])
            image= Variable(image.to(config.device))
            _, _, _ = use_model(image)
            # _ = use_model(image)
            x_feature = conv_out.features
            feature_save.append(x_feature.data.numpy())
            if i == config.PCA_data_num - 1:
                break

    # try:
    #     os.makedirs(f'2kinds_feature_save_{config.PCA_animal}')
    # except OSError:
    #     pass
    np.save("../2kinds_feature_save_%s/lime_feature_matrix_%d_layer_%d_%d.npy" % (config.PCA_animal, config.PCA_data_num,config.layer_type, config.visual_layer),
            feature_save, allow_pickle=True)
    # np.save("../2kinds_feature_save_%s/feature_matrix_%d_layer_%d_%d.npy" % (config.PCA_animal, config.PCA_data_num, config.layer_type, config.visual_layer),
    #         feature_save, allow_pickle=True)


def load_PCA(n_components=2):
    '''
    row-centered PCA
    :param n_components: the number of saved components
    :return: obtained PC after relu
    '''

    # no lime
    # feature_matrix = np.load(
    #     "../2kinds_feature_save_%s/feature_matrix_%d_layer_%d_%d.npy" % (config.PCA_animal, config.PCA_data_num,config.layer_type, config.visual_layer))

    # lime
    feature_matrix = np.load(
        "../2kinds_feature_save_%s/lime_feature_matrix_%d_layer_%d_%d.npy" % (
        config.PCA_animal, config.PCA_data_num, config.layer_type, config.visual_layer))

    feature_matrix_main = np.zeros(
        [feature_matrix.shape[1], feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    ratio_record = []

    feature_matrix_total = np.zeros(
        [n_components, feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])  # n_components

    for i in range(feature_matrix.shape[3]):
        for j in range(feature_matrix.shape[4]):
            feature_matrix_PCA = np.squeeze(feature_matrix[:, 0, :, i, j]).T
            pca = PCA(n_components=2)
            pca.fit(feature_matrix_PCA)
            ratio_record.append(pca.explained_variance_ratio_[0])
            feature_space = pca.fit_transform(feature_matrix_PCA)
            feature_matrix_main[0, :, i, j] = feature_space[:, config.PCA_main]
            feature_matrix_total[:, :, i, j] = feature_space[:, 0:n_components].T  # 0:n_components
            print(i, j)

    return torch.nn.functional.relu(torch.from_numpy(feature_matrix_main.astype(np.float32)))

def transform_raw_picture(pic):
    img = torchvision.utils.make_grid(pic)  
    img = img.numpy().transpose((1, 2, 0))  
    img = img * config.std + config.mean 
    return img

def total_variance(img, beta=2):
    TV_row = torch.clone(img)
    TV_col = torch.clone(img)
    _, C, H, W = img.shape
    TV_row[:, :, :, 0:W - 1] = img[:, :, :, 1:W] - img[:, :, :, 0:W - 1]
    TV_col[:, :, 0:H - 1, :] = img[:, :, 1:H, :] - img[:, :, 0:H - 1, :]
    TV = ((TV_row ** 2 + TV_col ** 2) ** (beta / 2)).sum()

    return TV

def picture_inverse(use_model, feature_true, conv_out, max_epoch, TV=True):
    if config.inverse_technique == 'GD':
        learning_rate = config.learning_rate
        # -------产生空白图----------
        pic_prior = torch.zeros([1, 3, config.fig_size, config.fig_size])

        pic_prior = Variable(pic_prior.to(config.device), requires_grad=True)
        feature_true = Variable(feature_true.to(config.device))

        # conv_out = LayerActivations(use_model.features[2], config.visual_layer)  # vgg
        # conv_out = LayerActivations(use_model.features[-1], config.visual_layer)  # resnet

        _,_, feature_prediction = use_model(pic_prior)
        # feature_prediction = use_model(pic_prior)
        act = conv_out.features
        feature_prediction = act.to(config.device)

        if TV == True:
            MSE = torch.sum((feature_prediction - feature_true) ** 2) + config.TV_coef * total_variance(pic_prior, beta=config.TV_beta)
        else:
            MSE = torch.sum((feature_prediction - feature_true) ** 2)

        H_grad = torch.autograd.grad(outputs=MSE.sum(), inputs=pic_prior, create_graph=True)[0]
        mu = - learning_rate * H_grad
        pic_new = pic_prior - learning_rate * H_grad

        img = torchvision.utils.make_grid(pic_new)  #
        img = img.cpu().data.numpy().transpose((1, 2, 0)) 
        img = img * config.std + config.mean  # (228, 906, 3)
        # plt.imshow(img)
        # plt.show()

        # inv_pic_plot = Visualization.transform_raw_picture(config,pic_new.cpu().data)
        # plt.imshow(inv_pic_plot)
        # plt.show()

        for epoch in range(max_epoch):
            a = time.time()
            _, _, feature_prediction = use_model(pic_new)
            # feature_prediction = use_model(pic_new)
            act = conv_out.features
            feature_prediction = act.to(config.device)
            if TV == True:
                MSE = torch.sum((feature_prediction - feature_true) ** 2) + config.TV_coef * total_variance(pic_new,beta=config.TV_beta) + config.L2_norm * torch.mean(pic_new ** 2)
            else:
                MSE = torch.sum((feature_prediction - feature_true) ** 2)
            H_grad = torch.autograd.grad(outputs=MSE.sum(), inputs=pic_new, create_graph=True)[0]
            pic_new = (pic_new - learning_rate * H_grad).cpu().data.numpy()

            pic_new = Variable(torch.from_numpy(pic_new.astype(np.float32)).to(config.device), requires_grad=True)
            b = time.time()
            # print(b-a,epoch, MSE, Visualization.total_variance(config,pic_new, beta=2))
            if (epoch + 1) % 100 == 0:
                print(f'mean:{torch.mean(pic_new ** 2)}')
                print(
                    f'epoch:{epoch + 1},MSE:{torch.sum((feature_prediction - feature_true) ** 2) / (feature_true.shape[0] * feature_true.shape[1] * feature_true.shape[2] * feature_true.shape[3])},'
                    f'TV:{config.TV_coef * total_variance(pic_new, beta=config.TV_beta) / (feature_true.shape[0] * feature_true.shape[1] * feature_true.shape[2] * feature_true.shape[3])}')
                # inv_pic_plot = Visualization.transform_raw_picture(config,pic_new.cpu().data)
                # plt.imshow(inv_pic_plot)
                # plt.show()

            if (epoch + 1) % 1000 == 0:
                learning_rate = learning_rate * 0.5

    return pic_new.cpu().data, feature_prediction.cpu().data.numpy()

def plot_lime_PCA(use_model, conv_out, N_select):
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 16,
             }

    config.PCA_data_num=500
    # generate_feature_matrix(324,lime=False)  # 525
    PCA_feature=load_PCA()
    plt.figure(1, figsize=(4, 4), dpi=300)
    plt.bar(range(N_select), PCA_feature.cpu().data.numpy().reshape(N_select), color='blue')
    plt.xlabel('Neuron Index', font1)
    plt.ylabel('Activation after PCA', font1)
    plt.savefig(f'../lime_save_2kinds/neuron_{config.PCA_animal}_{config.PCA_data_num}_lime_{config.PCA_main}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../lime_save_2kinds/neuron_{config.PCA_animal}_{config.PCA_data_num}_lime_{config.PCA_main}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()


    config.TV_coef=2
    plt.style.use('default')
    pic_new,feature_prediction=picture_inverse(use_model,PCA_feature,conv_out,max_epoch=4000)
    img = transform_raw_picture(pic_new)
    plt.figure(1,dpi=300)
    plt.imshow(np.clip(img,0,1))
    plt.axis('off')
    plt.savefig(f'../lime_save_2kinds/inverse_{config.PCA_animal}_{config.PCA_data_num}_no_lime_{config.PCA_main}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../lime_save_2kinds/inverse_{config.PCA_animal}_{config.PCA_data_num}_no_lime_{config.PCA_main}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

class PLOT():
    plt.style.use("ggplot")

    def __init__(self,conv_out,reshape_num):
        self.conv_out=conv_out
        self.reshape_num=reshape_num
        self.font1 = {'family': 'Arial',
                 'weight': 'normal',
                 # "style": 'italic',
                 'size': 16,
                 }

    def plot_one_sample(self):
        dataset = test_loader
        for i, data in enumerate(dataset):
            print('----------第%d次迭代----------' % (i))
            image, label = data

            image = Variable(image.to(config.device), requires_grad=True)
            _, _, x_feature = use_model(image)
            act = self.conv_out.features
            x_feature = act.to(config.device).reshape(self.reshape_num)
            feature_space = x_feature
            if i<15:
                plt.figure(1,figsize=(4,4),dpi=300)
                plt.bar(range(self.reshape_num),feature_space.cpu().data.numpy(),color='blue')
                plt.xlabel('Neuron Index',self.font1)
                plt.ylabel('Activation Values',self.font1)
                plt.savefig(f'../lime_save_2kinds/512_example_bar_{i}.tiff',
                            bbox_inches='tight', dpi=300)
                plt.savefig(f'../lime_save_2kinds/512_example_bar_{i}.pdf',
                            bbox_inches='tight', dpi=300)

                #plt.show()
                pic = image.cpu().data
                img = transform_raw_picture(pic)
                plt.figure(2,figsize=(4,4),dpi=300)
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(f'../lime_save_2kinds/512_example_{i}.tiff',
                            bbox_inches='tight', dpi=300)
                plt.savefig(f'../lime_save_2kinds/512_example_{i}.pdf',
                            bbox_inches='tight', dpi=300)
                plt.show()


    def plot_position(self):
        position = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_519.npy').astype(np.float32))
        position_origin = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin_519.npy').astype(np.float32))

        # position = torch.from_numpy(
        #     np.load(f'../result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
        # position_origin = torch.from_numpy(
        #     np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))



        try:
            os.makedirs(f'../lime_save/position_{config.position_space}_{config.position_animal}')
        except OSError:
            pass

        for i in range(position.shape[0]):

            image_plot = transform_raw_picture(position[i].cpu().data)
            image_plot_origin = transform_raw_picture(position_origin[i].cpu().data)
            plt.figure(1,figsize=(4,4),dpi=300)
            plt.imshow(image_plot)
            plt.axis('off')
            plt.savefig(f'../lime_save_2kinds/position_{config.position_space}_{config.position_animal}/no_{i}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'../lime_save_2kinds/position_{config.position_space}_{config.position_animal}/no_{i}.pdf',
                        bbox_inches='tight', dpi=300)

            plt.figure(2,figsize=(4,4),dpi=300)
            plt.imshow(image_plot_origin)
            plt.axis('off')
            plt.savefig(f'../lime_save_2kinds/position_{config.position_space}_{config.position_animal}/origin_{i}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'../lime_save_2kinds/position_{config.position_space}_{config.position_animal}/origin_{i}.pdf',
                        bbox_inches='tight', dpi=300)

            if i==10:
                break

def get_position(conv_out,N_select_PCA,N_select = 512,show_picture=False):
    # position = torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    #
    # position_origin=torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))

    position = torch.from_numpy(
        np.load(f'D:/TesNet-1/result_save/artificial mask/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))

    position_origin = torch.from_numpy(
        np.load(f'D:/TesNet-1/result_save/artificial mask/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))

    # position = torch.from_numpy(
    #     np.load(f'D:/TesNet-1/result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_2kinds.npy').astype(np.float32))
    # position_origin = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_origin_2kinds.npy').astype(np.float32))

    # position_auto = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/{config.position_space}_save_{config.position_animal}_531.npy').astype(np.float32))
    # position_origin_auto = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/{config.position_space}_save_{config.position_animal}_origin_531.npy').astype(np.float32))

    if show_picture==True:
        for i in range(position.shape[0]):
            plt.figure(1)
            image_plot = transform_raw_picture(position[i].cpu().data)
            plt.imshow(image_plot)


    feature_save=[]
    feature_save_origin=[]
    for i in range(position.shape[0]):
        image = position[i]
        image_origin=position_origin[i]


        label_true, x_feature = use_model(image.cuda())
        act = conv_out.features
        x_feature = act.to(device).reshape(512 )

        feature_save.append(act.data.numpy())

        label_true, x_feature = use_model(image_origin.cuda())
        act = conv_out.features
        feature_save_origin.append(act.data.numpy())
        x_feature_origin = act.to(device).reshape(512)

        if show_picture == True:
            plt.figure(2)
            plt.subplot(10,10,i+1)
            plt.plot((x_feature).cpu().data.numpy())
            plt.plot((-x_feature_origin).cpu().data.numpy())


    feature_matrix=np.array(feature_save)
    feature_matrix_origin = np.array(feature_save_origin)
    feature_PCA=position_PCA(feature_matrix,n_components=5)
    feature_PCA_origin=position_PCA(feature_matrix_origin,n_components=5)

    if show_picture==True:
        plt.figure(3,figsize=(2,2),dpi=300)
        plt.bar(range(512), feature_PCA_origin[0, :, 0, 0].cpu().data.numpy(), color='red', label=f'With {config.position_space}s')
        plt.bar(range(512), -feature_PCA[0,:,0,0].cpu().data.numpy(), color='blue',label=f'With {config.position_space}s masked')
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Scores of $1^{st}$ PC', fontproperties='Arial', fontsize=7)
        plt.ylim(-20, 30)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks([ -20,-10,  0, 10, 20,30], [20, 10, 0, 10, 20,30], fontproperties='Arial', size=7)
        plt.legend(loc='lower right',  
    frameon=False, 
    prop=legend_font)
        plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.tiff',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.pdf',
                    bbox_inches='tight', dpi=300)


        plt.figure(4, figsize=(2, 2), dpi=300)
        sort_index_cat = np.argsort(-np.abs(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))[0:1]  
        plt.bar(range(512),-(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))
        for i in range(len(sort_index_cat)):
            if sort_index_cat[i]<0:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                     fontproperties='Arial', fontsize=7)
            else:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] - 0.25,
                         '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                         fontproperties='Arial', fontsize=7)
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Difference', fontproperties='Arial', fontsize=7)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks( fontproperties='Arial',size=7)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.tiff',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.pdf',
                    bbox_inches='tight', dpi=300)

    sort_index_origin = np.argsort(-(feature_PCA_origin[0,:,0,0]))[0:N_select]
    PCA_cat=(feature_PCA_origin[0, :, 0, 0]-feature_PCA[0,:,0,0])[sort_index_origin].cpu().data.numpy()
    sort_index_position=np.argsort(-(np.abs(PCA_cat)))[0:N_select_PCA]
    #print(np.sign(PCA_cat[sort_index_position]))
    space_value=PCA_cat[sort_index_position]
    #print(sort_index_origin[sort_index_position]*np.sign(PCA_cat[sort_index_position]))
    space_index=sort_index_origin[sort_index_position] * np.sign(PCA_cat[sort_index_position])
    if show_picture==True:
        plt.show()
    return sort_index_origin,sort_index_position,space_index,space_value

def get_position_2(conv_out,N_select_PCA,N_select = 512,show_picture=False):
    # position = torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    #
    # position_origin=torch.from_numpy(np.load(f'result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))

    # position = torch.from_numpy(
    #     np.load(f'D:/TesNet-1/result_save/artificial mask/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    #
    # position_origin = torch.from_numpy(
    #     np.load(f'D:/TesNet-1/result_save/artificial mask/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))

    position = torch.from_numpy(
        np.load(f'D:/TesNet-1/result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_2kinds.npy').astype(np.float32))
    position_origin = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_origin_2kinds.npy').astype(np.float32))

    # position = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/{config.position_space}_save_{config.position_animal}_521.npy').astype(np.float32))
    # position_origin = torch.from_numpy(np.load(f'D:/TesNet-1/result_save/{config.position_space}_save_{config.position_animal}_origin_521.npy').astype(np.float32))

    if show_picture==True:
        for i in range(position.shape[0]):
            plt.figure(1)
            image_plot = transform_raw_picture(position[i].cpu().data)
            plt.imshow(image_plot)


    feature_save=[]
    feature_save_origin=[]
    for i in range(position.shape[0]):
        image = position[i]
        image_origin=position_origin[i]


        label_true, x_feature = use_model(image.cuda())
        act = conv_out.features
        x_feature = act.to(device).reshape(512 )

        feature_save.append(act.data.numpy())

        label_true, x_feature = use_model(image_origin.cuda())
        act = conv_out.features
        feature_save_origin.append(act.data.numpy())
        x_feature_origin = act.to(device).reshape(512)

        if show_picture == True:
            plt.figure(2)
            plt.subplot(10,10,i+1)
            plt.plot((x_feature).cpu().data.numpy())
            plt.plot((-x_feature_origin).cpu().data.numpy())


    feature_matrix=np.array(feature_save)
    feature_matrix_origin = np.array(feature_save_origin)
    feature_PCA=position_PCA(feature_matrix,n_components=5)
    feature_PCA_origin=position_PCA(feature_matrix_origin,n_components=5)

    if show_picture==True:
        plt.figure(3,figsize=(2,2),dpi=300)
        plt.bar(range(512), feature_PCA_origin[0, :, 0, 0].cpu().data.numpy(), color='red', label=f'With {config.position_space}s')
        plt.bar(range(512), -feature_PCA[0,:,0,0].cpu().data.numpy(), color='blue',label=f'With {config.position_space}s masked')
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Scores of $1^{st}$ PC', fontproperties='Arial', fontsize=7)
        plt.ylim(-20, 30)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks([ -20,-10,  0, 10, 20,30], [20, 10, 0, 10, 20,30], fontproperties='Arial', size=7)
        plt.legend(loc='lower right',  
    frameon=False, 
    prop=legend_font)
        plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.tiff',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'PPT_fig/position/{config.position_space}_{config.position_animal}.pdf',
                    bbox_inches='tight', dpi=300)


        plt.figure(4, figsize=(2, 2), dpi=300)
        sort_index_cat = np.argsort(-np.abs(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))[0:1]  
        plt.bar(range(512),-(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy()))
        for i in range(len(sort_index_cat)):
            if sort_index_cat[i]<0:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                     fontproperties='Arial', fontsize=7)
            else:
                plt.text(sort_index_cat[i], -(feature_PCA[0,:,0,0].cpu().data.numpy()-feature_PCA_origin[0, :, 0, 0].cpu().data.numpy())[sort_index_cat[i]] - 0.25,
                         '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
                         fontproperties='Arial', fontsize=7)
        plt.xlabel('Neuron Index', fontproperties='Arial', fontsize=7)
        plt.ylabel('Difference', fontproperties='Arial', fontsize=7)
        plt.xticks(fontproperties='Arial', size=7)
        plt.yticks( fontproperties='Arial',size=7)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.tiff',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}.pdf',
                    bbox_inches='tight', dpi=300)

    sort_index_origin = np.argsort(-(feature_PCA_origin[0,:,0,0]))[0:N_select]
    PCA_cat=(feature_PCA_origin[0, :, 0, 0]-feature_PCA[0,:,0,0])[sort_index_origin].cpu().data.numpy()
    sort_index_position=np.argsort(-(np.abs(PCA_cat)))[0:N_select_PCA]
    #print(np.sign(PCA_cat[sort_index_position]))
    space_value=PCA_cat[sort_index_position]
    #print(sort_index_origin[sort_index_position]*np.sign(PCA_cat[sort_index_position]))
    space_index=sort_index_origin[sort_index_position] * np.sign(PCA_cat[sort_index_position])
    if show_picture==True:
        plt.show()
    return sort_index_origin,sort_index_position,space_index,space_value


def val_distribution(conv_out, reshape_num):
    cat_space = []
    cat_space_pic=[]
    for i, data in enumerate(test_loader):
        print('----------第%d次迭代----------' % (i))
        if i<271:
            continue
        image, label = data
        image = Variable(image.to(config.device), requires_grad=True)
        _, _, x_feature = use_model(image)
        act = conv_out.features
        x_feature = act.to(config.device).reshape(reshape_num).cpu().data.numpy()
        cat_space.append(x_feature)
        cat_space_pic.append(image.cpu().data.numpy())
        if i==269:
            break

    cat_space = np.array(cat_space)
    cat_space_pic=np.array(cat_space_pic)
    print(cat_space.shape)
    print(cat_space_pic.shape)

    np.save(f'../result_save_2kinds/{config.position_animal}_space_2000.npy', cat_space)
    np.save(f'../result_save_2kinds/{config.position_animal}_space_2000_pic.npy', cat_space_pic)

    return cat_space,cat_space_pic

def get_cdf_ratio_1(x, mean):
    min_cdf = st.norm.cdf((np.min(mean) - np.mean(mean)) / np.std(mean))
    max_cdf = st.norm.cdf((np.max(mean) - np.mean(mean)) / np.std(mean))
    cdf = st.norm.cdf((x - np.mean(mean)) / np.std(mean))
    cdf_ratio = (cdf - min_cdf) / (max_cdf - min_cdf)
    cdf_ratio = cdf_ratio

    return cdf_ratio

def get_cdf_ratio_2(x, mean):
    min=st.norm.cdf((np.min(mean) - np.mean(mean)) / np.std(mean))
    max=st.norm.cdf((np.max(mean) - np.mean(mean)) / np.std(mean))
    return (st.norm.cdf((x - np.mean(mean)) / np.std(mean))-min)/(max-min)

def get_cdf_ratio(x, mean, scale_factor):
    min_cdf = st.norm.cdf((np.min(mean) - np.mean(mean)) / np.std(mean))
    max_cdf = st.norm.cdf((np.max(mean) - np.mean(mean)) / np.std(mean))
    cdf = st.norm.cdf((x - np.mean(mean)) / np.std(mean))
    cdf_ratio = (cdf - min_cdf) / (max_cdf - min_cdf)
    cdf_ratio = cdf_ratio * scale_factor * 2

    return cdf_ratio

def pca_scores(use_model):
    for i, data in enumerate(test_loader):
        # if i<19:
        #     continue
        print('----------第%d次迭代----------' % (i))
        image, label = data
        #image=true_sample[i]
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        #fname = r'D:\pycharm project\VAE_PDE\big_cat_1.jpg'
        file_name='val'
        # image = Image.open(fname)
        # image = transform(image).reshape([1, 3, 224, 224])


        image = Variable(image.cuda(), requires_grad=True)
        label_true, _, x_feature = use_model(image)

        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.cuda().reshape(512)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')

        cdf_record=[]
        scale_factor = {
            ('flower3', 'petal'): 0.6781,
            ('flower3', 'center'): 0.3219,
            ('flower4', 'center'): 0.338,
            ('flower4', 'petal'): 0.6619,
        }
        for i in ['center', 'petal']:
            for j in ['flower4']:
                config.position_space = i
                config.position_animal = j
                _, space_index, space_value = get_position_2(conv_out, 5, show_picture=False)
                space_index = np.array(space_index, dtype=int)
                space = np.load(f'../result_save_2kinds/{config.position_animal}_space_2000.npy')

                eye_space = space[:, space_index]
                sum = 0
                for k in range(eye_space.shape[1]):
                    sum += eye_space[:, k] * space_value[k]
                mean = sum / eye_space.shape[1]

                eye_space = feature_space[space_index].cpu().data.numpy()
                sum = 0
                for k in range(eye_space.shape[0]):
                    sum += eye_space[k] * space_value[k]
                mean_pic = sum / eye_space.shape[0]
                cdf_record.append(get_cdf_ratio(mean_pic,mean, scale_factor[(config.position_animal, config.position_space)]))
                print(f'{j},{i},{get_cdf_ratio(mean_pic,mean, scale_factor[(config.position_animal, config.position_space)])}')
        # dog_cdf=np.array(cdf_record)[[0,2]]
        # cat_cdf=np.array(cdf_record)[[1,3]]

    '''
    for i, data in enumerate(test_loader):
        print('----------第%d次迭代----------' % (i))
        image, label = data
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        image = Variable(image.cuda(), requires_grad=True)
        label_true, _, x_feature = use_model(image)

        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.cuda().reshape(512)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    cdf_record = []
    scale_factor = {
        ('flower', 'center'): 0.4671,
        ('flower', 'petal'): 0.5329,
        ('flower2', 'center'): 0.0684,
        ('flower2', 'petal'): 0.3337,
    }
    for i in ['center', 'petal']:
        for j in ['flower', 'flower2']:
            config.position_space = i
            config.position_animal = j
            _, space_index, space_value = get_position(conv_out, 5, show_picture=False)
            space_index = np.array(space_index, dtype=int)
            space = np.load(f'../result_save_2kinds/{config.position_animal}_space_2000.npy')

            eye_space = space[:, space_index]
            sum = 0
            for k in range(eye_space.shape[1]):
                sum += eye_space[:, k] * space_value[k]
            mean = sum / eye_space.shape[1]

            eye_space = feature_space[space_index].cpu().data.numpy()
            sum = 0
            for k in range(eye_space.shape[0]):
                sum += eye_space[k] * space_value[k]
            mean_pic = sum / eye_space.shape[0]
            # cdf_record.append(get_cdf_ratio(mean_pic, mean))
            print(f'{j},{i},{get_cdf_ratio(mean_pic, mean, scale_factor, node=True)}')
    '''

def get_space(conv_out, reshape_num,add_position='origin'):
    cat_space = []
    cat_space_pic=[]
    if add_position=='origin':
        # position = torch.from_numpy(np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_2kinds.npy').astype(np.float32))
        # position_origin = torch.from_numpy(np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin_2kinds.npy').astype(np.float32))
        position = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_519.npy').astype(np.float32))
        position_origin = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin_519.npy').astype(np.float32))

        for i in range(position_origin.shape[0]):
            print('----------第%d次迭代----------' % (i))
            image_origin = position_origin[i]
            image = image_origin
            image = Variable(image.to(config.device), requires_grad=True)
            _, _, x_feature = use_model(image)
            act = conv_out.features
            x_feature = act.to(config.device).reshape(reshape_num)
            cat_space.append(x_feature.cpu().data.numpy())
            cat_space_pic.append(image.cpu().data.numpy())

    elif add_position=='position':
        # position = torch.from_numpy(np.load(f'../result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_2kinds.npy').astype(np.float32))
        # position_origin = torch.from_numpy(np.load(f'../result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_origin_2kinds.npy').astype(np.float32))
        # position = torch.from_numpy(
        #     np.load(f'../result_save/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
        # position_origin = torch.from_numpy(
        #     np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin.npy').astype(
        #         np.float32))
        position = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_519.npy').astype(np.float32))
        position_origin = torch.from_numpy(
            np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin_519.npy').astype(np.float32))

        for i in range(58):
            print('----------第%d次迭代----------' % (i))
            image_origin = position[i]
            image = image_origin
            image = Variable(image.to(config.device), requires_grad=True)
            _, _, x_feature = use_model(image)
            act = conv_out.features
            x_feature = act.to(config.device).reshape(reshape_num)
            cat_space.append(x_feature.cpu().data.numpy())
            cat_space_pic.append(image.cpu().data.numpy())

    cat_space=np.array(cat_space)
    cat_space_pic=np.array(cat_space_pic)
    return cat_space,cat_space_pic

def plot_distribution(space_index,space_index_auto,space_value,space_value_auto,picture='big'):

    space_index=np.array(space_index, dtype=int)
    space_index_auto = np.array(space_index_auto, dtype=int)
    space = np.load(f'result_save/{config.position_animal}_space_2000.npy')
    space_pic = np.load(f'result_save/{config.position_animal}_space_2000_pic.npy')
    eye_space = space[:, space_index]
    eye_space_auto = space[:, space_index_auto]
    sum=0
    for j in range(eye_space.shape[1]):
        sum+=eye_space[:,j]*space_value[j]
    mean=sum/eye_space.shape[1]

    sum_1=0
    for k in range(eye_space_auto.shape[1]):
        sum_1+=eye_space_auto[:,k]*space_value_auto[k]
    mean_1=sum_1/eye_space_auto.shape[1]

    print(np.min(mean), np.max(mean))
    print(np.min(mean_1), np.max(mean_1))

    min_value = min(np.min(mean), np.min(mean_1))
    max_value = max(np.max(mean), np.max(mean_1))

    normalized_mean = (mean - np.mean(mean)) / np.std(mean)
    normalized_mean_1 = (mean_1 - np.mean(mean_1)) / np.std(mean_1)

    plt.figure(1, figsize=(3, 3), dpi=300)
    plt.style.use("seaborn-darkgrid")  # seaborn-darkgrid, ggplot
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    a, b = stats.probplot(normalized_mean, dist="norm", fit=True)
    c, d = stats.probplot(normalized_mean_1, dist="norm", fit=True)
    plt.scatter(a[0], a[1], c='blue', s=20) 
    plt.scatter(c[0], c[1], c='red', s=20) 
    plt.plot(c[0], d[0] * c[0] + d[1], color='black',linewidth=3)  
    plt.ylabel('Ordered values', font2)
    plt.xlabel('Theoretical quantities', font2)
    plt.title(f'{config.position_animal} {config.position_space}', font2, pad=7)  # {config.position_animal}
    plt.xticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], fontsize=10)
    y_min = min(min(normalized_mean), min(normalized_mean_1)) - 0.5
    y_max = max(max(normalized_mean), max(normalized_mean_1)) + 0.5

    # plt.yticks(range(int(y_min), int(y_max)), fontsize=10)
    plt.yticks(fontsize=10)

    n, m = scipy.stats.shapiro(normalized_mean)
    n_1, m_1 = scipy.stats.shapiro(normalized_mean_1)
    r_squared = n
    r_squared_1 = n_1

    print(1111)
    print(r_squared, r_squared_1)


    line1 = plt.scatter(a[0], a[1], c='blue', s=20) 
    line2 = plt.scatter(c[0], c[1], c='red', s=20)  

    plt.legend((line1, line2),(f'S-XAI: R\u00b2 = {r_squared:.3f}',f'AS-XAI: R\u00b2 = {r_squared_1:.3f}'),loc='lower right', prop={'family': 'Arial', 'size': 10},shadow=True,frameon=True,facecolor='white',labelspacing=0.5, handlelength=0.5)

    plt.tight_layout()

    plt.savefig(f'PPT_fig/position/qq_{config.position_animal}_{config.position_space}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/position/qq_{config.position_animal}_{config.position_space}.pdf',
                bbox_inches='tight', dpi=300)

    plt.show()

    plt.figure(2,figsize=(2,2),dpi=300)
    mu = np.mean(mean)  
    sig = np.std(mean)  

    print('TPSA正态性检验：\n', scipy.stats.shapiro(mean))
    print(mu,sig)
    x = np.linspace(np.min(mean),np.max(mean),100) 
    y = np.exp(-(x - mu) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig) 
    #plt.xlim(0,16)
    plt.plot(x, y, "red", linewidth=2)
    plt.hist(mean,bins=40,color='blue',density=True)
    plt.ylabel('Probability density',font1)
    plt.xlabel('Values of $A_s$',font1)
    plt.xticks(fontproperties='Arial', size=7)
    plt.yticks(fontproperties='Arial', size=7)

    plt.savefig(f'PPT_fig/position/dis_{config.position_animal}_{config.position_space}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'PPT_fig/position/dis_{config.position_animal}_{config.position_space}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()



    plt.style.use('default')
    if picture=='big':
        for i in range(mean.shape[0]):
            if get_cdf_ratio(mean[i],mean) >0.9:
                print(get_cdf_ratio(mean[i],mean))
                image_origin_plot = transform_raw_picture(torch.from_numpy(space_pic[i]))
                plt.figure(2,figsize=(4,4),dpi=300)
                plt.imshow(image_origin_plot)
                plt.axis('off')
                # plt.savefig(f'PPT_fig/position/big_{config.position_animal}_{config.position_space}_{i}.tiff',
                #             bbox_inches='tight', dpi=300)
                # plt.savefig(f'PPT_fig/position/big_{config.position_animal}_{config.position_space}_{i}.pdf',
                #             bbox_inches='tight', dpi=300)
                plt.show()
    if picture=='small':
        for i in range(mean.shape[0]):
            if get_cdf_ratio(mean[i],mean) <0.1:
                print(get_cdf_ratio(mean[i],mean))
                image_origin_plot = transform_raw_picture(torch.from_numpy(space_pic[i]))
                plt.figure(2, figsize=(4, 4), dpi=300)
                plt.imshow(image_origin_plot)
                plt.axis('off')
                # plt.savefig(f'PPT_fig/position/small_{config.position_animal}_{config.position_space}_{i}.tiff',
                #             bbox_inches='tight', dpi=300)
                # plt.savefig(f'PPT_fig/position/small_{config.position_animal}_{config.position_space}_{i}.pdf',
                #             bbox_inches='tight', dpi=300)
                plt.show()


    if picture=='location':
        cat_space,cat_space_pic=get_space(conv_out,512,add_position='position')
        cat_space_origin, cat_space_pic_origin = get_space(conv_out, 512, add_position='origin')
        eye_space = cat_space[:, space_index]
        sum = 0
        for j in range(eye_space.shape[1]):
            sum += eye_space[:, j] * space_value[j]
        mean_location = sum / eye_space.shape[1]

        eye_space = cat_space_origin[:, space_index]
        sum = 0
        for j in range(eye_space.shape[1]):
            sum += eye_space[:, j] * space_value[j]
        mean_origin = sum / eye_space.shape[1]
        for i in range(mean_location.shape[0]):
            print(get_cdf_ratio(mean_location[i],mean))
            print(get_cdf_ratio(mean_origin[i],mean))
            plt.figure(6,dpi=300)
            plt.subplot(1,2,1)
            plt.imshow(transform_raw_picture(torch.from_numpy(cat_space_pic[i])))
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(transform_raw_picture(torch.from_numpy(cat_space_pic_origin[i])))
            plt.axis('off')
            # plt.savefig(f'PPT_fig/position/compare_{config.position_animal}_{config.position_space}_{i}.tiff',
            #             bbox_inches='tight', dpi=300)
            # plt.savefig(f'PPT_fig/position/compare_{config.position_animal}_{config.position_space}_{i}.pdf',
            #             bbox_inches='tight', dpi=300)
            plt.show()
            print('-------------------------')


def position_PCA(feature_matrix,n_components):
    n_components_record = []
    ratio_record = []
    ratio_record_origin = []

    feature_matrix_main = np.zeros(
        [feature_matrix.shape[1], feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    feature_matrix_total = np.zeros(
        [n_components, feature_matrix.shape[2], feature_matrix.shape[3], feature_matrix.shape[4]])

    for i in range(feature_matrix.shape[3]):
        for j in range(feature_matrix.shape[4]):
            feature_matrix_PCA = np.squeeze(feature_matrix[:, 0, :, i, j]).T
            pca = PCA(n_components=n_components)
            pca.fit(feature_matrix_PCA)
            ratio_record.append(pca.explained_variance_ratio_[0])
            feature_space = pca.fit_transform(feature_matrix_PCA)
            feature_matrix_main[0, :, i, j] = feature_space[:, 0]
            feature_matrix_total[:, :, i, j] = feature_space[:, 0:n_components].T

    feature_matrix_main_1 = torch.nn.functional.relu(torch.from_numpy(feature_matrix_main.astype(np.float32)))
    return feature_matrix_main_1


def compute_orthogonality(feature_matrix):
    cov_matrix = np.cov(feature_matrix.T)

    num_features = cov_matrix.shape[0]
    orthogonality = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            orthogonality[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])

    return orthogonality

def get_inverse_position(conv_out,N_select = 512, show_picture=False):
    # position = torch.from_numpy(np.load(f'../result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_2kinds.npy').astype(np.float32))
    # position_origin=torch.from_numpy(np.load(f'../result_save/cat_kinds/{config.position_space}_save_{config.position_animal}_origin_2kinds.npy').astype(np.float32))

    position = torch.from_numpy(np.load(f'../result_save/artificial mask/{config.position_space}_save_{config.position_animal}.npy').astype(np.float32))
    position_origin = torch.from_numpy(np.load(f'../result_save/artificial mask/{config.position_space}_save_{config.position_animal}_origin.npy').astype(np.float32))

    # position = torch.from_numpy(np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_520.npy').astype(np.float32))
    # position_origin=torch.from_numpy(np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_origin_520.npy').astype(np.float32))

    # position = torch.from_numpy(np.load(f'../result_save/{config.position_space}_save_{config.position_animal}_unorth.npy').astype(np.float32))


    if show_picture==True:
        for i in range(position.shape[0]):
            plt.figure(1)
            image_plot = transform_raw_picture(position[i].cpu().data)
            plt.imshow(image_plot)


    feature_save=[]
    feature_save_origin=[]
    for i in range(position.shape[0]):
        image = position[i]
        image_origin=position_origin[i]


        _,_, x_feature = use_model(image.cuda())
        act = conv_out.features
        x_feature = act.to(config.device).reshape(N_select)

        feature_save.append(act.data.numpy())

        _, _, x_feature = use_model(image_origin.cuda())
        act = conv_out.features
        feature_save_origin.append(act.data.numpy())
        x_feature_origin = act.to(config.device).reshape(N_select)

        if show_picture == True:
            plt.figure(2)
            plt.subplot(10,10,i+1)
            plt.plot((x_feature).cpu().data.numpy())
            plt.plot((-x_feature_origin).cpu().data.numpy())


    feature_matrix=np.array(feature_save)
    feature_matrix_origin = np.array(feature_save_origin)
    feature_PCA=position_PCA(feature_matrix,n_components=10)
    feature_PCA_origin=position_PCA(feature_matrix_origin,n_components=10)

    #cat space visualization
    eye_feature=np.zeros([1,N_select,1,1])
    error=feature_PCA_origin-feature_PCA
    error=feature_PCA
    if config.position_space == 'ear':
        if config.position_animal == 'cat':
            sort_index_cat=[37, 351, 171, 463, 338, 137, 461, 297, 443, 286, 274, 395, 319,
       251, 199, 191, 391, 163, 336, 234, 429,  90, 109, 151, 263, 121,
       229,  14, 412,  11, 182, 224,   9, 238, 112, 431, 100, 305, 503,
       111,  88, 283, 465,  61, 320, 358, 321, 458,  48, 120,  34,  16,
       228, 441, 346, 204, 411,  13,  12, 186,  91, 421, 405, 127, 489,
       486,  63, 379, 240,  77, 136, 287, 198, 210, 502, 233, 142, 145,
        49, 124, 214, 498, 377, 302,  25, 375, 349, 378, 400, 495, 222,
       347, 292, 504, 392,   8,  75, 188,  40, 219, 174, 259, 442, 399,
       456, 285, 107, 179, 249, 350, 139, 484, 459, 348, 328,  47, 129,
        44, 119,  56, 294, 303, 354, 440,  33, 398, 130, 501, 451, 376,
       407, 269, 447,  26, 126, 493, 380, 218, 232, 299,   2,  19, 434,
       444, 230,   4, 435, 505, 310, 401, 413, 206,  20, 284, 339, 438,
        87, 213, 252, 221, 388, 453,  78,  18, 156, 288,  23, 323, 462,
       140, 363, 200,   5, 277, 293, 153, 267,  21, 497, 211, 231, 329,
       448, 340, 422, 275,  53, 314, 315, 316, 364, 317, 365, 322, 318,
       366, 324, 353, 326, 355, 356, 357, 345, 344, 343, 342, 341, 337,
       325, 335, 334, 333, 332, 360, 361, 362, 352, 330, 327, 359, 331,
         0, 368, 474, 473, 472, 471, 470, 469, 468, 467, 475, 466, 460,
       457, 455, 454, 452, 450, 449, 446, 464, 476, 477, 478, 509, 508,
       507, 506, 500, 499, 496, 494, 492, 491, 490, 488, 487, 485, 483,
       482, 481, 480, 479, 445, 439, 437, 436, 397, 396, 394, 393, 390,
       389, 387, 386, 385, 384, 383, 382, 381, 374, 373, 372, 371, 370,
       369, 402, 367, 403, 406, 433, 432, 430, 428, 427, 426, 425, 424,
       423, 420, 419, 418, 417, 416, 415, 414, 410, 409, 408, 404, 313,
       255, 311, 110, 108, 106, 105, 104, 103, 102, 101,  99,  98,  97,
        96,  95,  94,  93,  92,  89,  86,  85,  84,  83, 113,  82, 114,
       116, 152, 150, 149, 148, 147, 146, 144, 143, 141, 138, 135, 134,
       133, 132, 131, 128, 125, 123, 122, 118, 117, 115, 154,  81,  79,
        42,  41,  39,  38,  36,  35,  32,  31,  30,  29,  28,  27,  24,
        22,  17,  15,  10,   7,   6,   3,   1,  43,  80,  45,  50,  76,
        74,  73,  72,  71,  70,  69,  68,  67,  66,  65,  64,  62,  60,
        59,  58,  57,  55,  54,  52,  51,  46, 312, 155, 158, 265, 264,
       262, 261, 260, 258, 257, 256, 510, 254, 253, 250, 248, 247, 246,
       245, 244, 243, 242, 241, 239, 266, 237, 268, 271, 309, 308, 307,
       306, 304, 301, 300, 298, 296, 295, 291, 290, 289, 282, 281, 280,
       279, 278, 276, 273, 272, 270, 157, 236, 227, 184, 183, 181, 180,
       178, 177, 176, 175, 173, 172, 170, 169, 168, 167, 166, 165, 164,
       162, 161, 160, 159, 185, 235, 187, 190, 226, 225, 223, 220, 217,
       216, 215, 212, 209, 208, 207, 205, 203, 202, 201, 197, 196, 195,
       194, 193, 192, 189, 511] # cat_ear
        else:
            sort_index_cat=[37, 145,   9, 296, 468, 217, 306, 172, 462, 211, 137, 338,   8,
       124, 395, 140,  84, 286, 233, 417, 464, 344,   5,   1, 274, 429,
       301, 342, 105, 126, 318, 251, 121, 102, 109, 454, 327, 186, 204,
       113, 466, 160, 489, 258,  85, 354, 112, 465, 114, 228, 302, 142,
       347, 221, 411,  36, 111, 419, 177,  12, 487, 349,  88, 205, 426,
        96,  56, 481, 502, 171, 119, 229, 495, 463, 509, 500, 287,  23,
        61, 407, 129, 291, 440, 421,  41, 275,  97, 498, 255,  38,  11,
       348, 408, 490, 182, 183, 441, 351, 199, 208, 144, 345, 409, 453,
       312, 352, 436, 325,  79, 404, 444, 398, 299, 226, 122,  71, 305,
       350, 391, 267, 422, 431,  75, 308, 474, 218,  18, 238, 174, 328,
       456, 103, 188, 321, 191, 270, 315, 187, 392, 156, 451, 192,  39,
       316, 219,   2,  19,  30, 415, 263,  52,  31, 382, 285, 261, 279,
       134, 380, 427, 249, 104, 277,  50, 399, 210, 141, 236, 107, 367,
       446,  53, 289,   3, 447, 493, 313, 132, 240, 283, 227, 503, 254,
       179, 214,  34,  93, 405, 412, 136, 310, 459,  35, 201, 402,  99,
       168, 423,  91, 508, 241,  49, 388, 393,  78, 363, 259, 139, 248,
       387, 457, 149, 130, 163,  87, 317, 269, 507, 346, 492, 506, 361,
       337, 360, 501, 339, 343, 364, 365, 504, 341, 362, 499, 336, 359,
       355, 505, 330, 329, 494, 496, 356, 331, 357, 332, 333, 334, 335,
       358, 497, 353, 340, 455, 368, 435, 434, 433, 432, 475, 476, 473,
       477, 478, 428, 479, 425, 424, 480, 430, 437, 472, 438, 452, 458,
       450, 449, 448, 460, 445, 461, 443, 442, 467, 469, 470, 439, 471,
       420, 418, 326, 416, 384, 383, 381, 491, 379, 378, 377, 376, 375,
       374, 373, 372, 371, 370, 369, 385, 366, 386, 488, 414, 413, 482,
       410, 483, 484, 406, 403, 485, 401, 400, 397, 396, 486, 394, 389,
       390,   0, 323, 115, 110, 108, 106, 101, 100,  98,  95,  94,  92,
        90,  89,  86,  83,  82,  81,  80,  77,  76, 116,  74, 117, 120,
       157, 155, 154, 153, 152, 151, 150, 148, 147, 146, 143, 138, 135,
       133, 131, 128, 127, 125, 123, 118,  73,  72,  70,  32,  29,  28,
        27,  26,  25,  24,  22,  21,  20,  17,  16,  15,  14,  13,  10,
         7,   6,   4,  33,  40,  42,  43,  69,  68,  67,  66,  65,  64,
        63,  62,  60, 158,  59,  57,  55,  54,  51,  48,  47,  46,  45,
        44,  58, 324, 159, 162, 276, 273, 272, 271, 268, 266, 265, 264,
       262, 260, 257, 256, 510, 253, 252, 250, 247, 246, 245, 278, 244,
       280, 282, 322, 320, 319, 314, 311, 309, 307, 304, 303, 300, 298,
       297, 295, 294, 293, 292, 290, 288, 284, 281, 243, 242, 239, 195,
       194, 193, 190, 189, 185, 184, 181, 180, 178, 176, 175, 173, 170,
       169, 167, 166, 165, 164, 196, 197, 198, 200, 237, 235, 234, 232,
       231, 230, 225, 224, 223, 161, 222, 216, 215, 213, 212, 209, 207,
       206, 203, 202, 220, 511] # dog_ear

    elif config.position_space == 'leg':
        if config.position_animal == 'cat':
            sort_index_cat=[91, 251, 191, 199, 124, 422, 249, 391, 447, 200, 9, 182, 286,
        49, 329, 230,   5,  40, 188, 283, 109, 229,  88, 463, 321, 407,
       174, 453,  56,  78, 193, 392, 379, 219, 405, 347, 126, 263, 204,
       139, 421, 498, 440, 413, 111, 112, 336, 228, 459,  83, 198,  12,
       259, 297, 151, 224, 302, 349, 145,  37, 465,   2, 210, 299, 458,
        11, 375, 142, 268, 277, 107, 163, 431, 292, 136, 222, 350, 461,
       442, 434, 240, 267, 289, 399, 328, 415, 401,  75, 100, 438, 269,
       264, 443,  31, 120, 318, 441, 127, 288, 446,   8, 387, 129, 503,
       493, 238, 339, 121, 287, 456,  23,  47,  60,  13, 305, 377, 303,
       502, 206, 156, 435,  19,  38,  61, 117, 234, 320, 282, 380, 395,
       338,  25, 137,  74, 393, 341, 378, 397,  14,  63, 345, 427, 239,
       371,  64,  71, 218, 243, 489, 448, 451,  20,  16, 501, 358, 388,
       412, 257, 285, 398, 505, 319, 462, 186, 293, 274, 340,  77,  33,
       363, 179, 153, 429, 150, 175, 351, 232, 444,   3, 116, 171, 354,
       485, 130, 348, 481,  18, 138, 495, 252,  90, 476,  41, 170, 195,
        70, 214, 114, 346,   4, 445, 213, 504, 141, 324, 342, 343, 359,
       333, 331, 368, 367, 322, 323, 369, 344, 352, 353, 325, 360, 337,
       334, 335, 361, 327, 355, 366, 356, 365, 326, 364, 362, 357, 330,
       332,   0, 372, 477, 475, 474, 473, 472, 471, 470, 478, 469, 467,
       466, 464, 460, 457, 455, 454, 468, 479, 480, 482, 509, 508, 507,
       506, 500, 499, 497, 496, 494, 492, 491, 490, 488, 487, 486, 484,
       483, 452, 370, 450, 439, 403, 402, 400, 396, 394, 390, 389, 404,
       386, 384, 383, 382, 381, 376, 374, 373, 385, 406, 408, 409, 437,
       436, 433, 432, 430, 428, 426, 425, 424, 423, 420, 419, 418, 416,
       414, 411, 410, 449, 417, 255, 316, 113, 110, 108, 106, 105, 104,
       103, 102, 101, 115,  99,  97,  96,  95,  94,  93,  92,  89,  87,
        86,  98,  85, 118, 122, 158, 157, 155, 154, 152, 149, 148, 147,
       146, 119, 144, 140, 135, 134, 133, 132, 131, 128, 125, 123, 143,
        84,  82,  81,  42,  39,  36,  35,  34,  32,  30,  29,  28,  43,
        27,  24,  22,  21,  17,  15,  10,   7,   6,   1,  26,  44,  45,
        46,  80,  79,  76,  73,  72,  69,  68,  67,  66,  65,  62,  59,
        58,  57,  55,  54,  53,  52,  51,  50,  48, 159, 160, 161, 162,
       275, 273, 272, 271, 270, 266, 265, 262, 261, 276, 260, 256, 510,
       254, 253, 250, 248, 247, 246, 245, 258, 278, 279, 280, 315, 314,
       313, 312, 311, 310, 309, 308, 307, 306, 304, 301, 300, 298, 296,
       295, 294, 291, 290, 284, 281, 244, 317, 242, 237, 192, 190, 189,
       187, 185, 184, 183, 181, 180, 194, 178, 176, 173, 172, 169, 168,
       167, 166, 165, 164, 177, 196, 197, 201, 236, 235, 233, 231, 227,
       226, 225, 223, 221, 220, 217, 216, 215, 212, 211, 209, 208, 207,
       205, 203, 202, 241, 511] # cat_leg
        else:
            sort_index_cat=[104, 352, 124, 217, 409, 317, 408, 347,  57,  84, 251, 318,  38,
         3, 151, 453,   8, 218, 509,   1, 466, 286, 396, 241, 171,  79,
        37, 197, 502, 198,  11, 395, 211,  14,  40,  23, 243, 156, 417,
       440, 181, 462, 465, 393, 136, 504,  17, 145, 468, 204, 186, 234,
       464,  96, 302, 102, 301, 490, 177,  91, 375, 244, 421,   9,   5,
       334, 188, 451, 367, 327, 446, 249, 258,  81, 349,  56, 351, 429,
       267, 487, 339, 121, 470, 263, 172, 214, 392, 436, 498, 403, 114,
       228, 110, 382, 208, 113, 338,  78, 191, 412, 259, 132, 119, 129,
       441, 381, 344,  50,   2, 141,   7, 230, 444, 329, 160, 304, 183,
       505, 342, 140, 137, 312, 463, 274, 146, 163, 508,  85, 182, 169,
       161, 250, 358,  41, 348, 240, 184, 489, 287, 210, 306, 398, 221,
       431, 476, 319, 305, 407, 290, 174, 419, 418, 427,  63, 109, 454,
       293,  61, 134, 328, 185, 495, 456,  66, 238, 426, 321, 299, 283,
       411,  18, 354,  90, 397, 296, 126,  55, 391, 474,  34,  35, 423,
       415, 308, 486, 350, 320, 200, 142, 459, 400, 105,  75, 187, 107,
       233, 360, 270, 168,  21, 447,  30, 245,  87, 219,  77, 365, 364,
       368, 366, 345, 362, 322, 323, 324, 325, 326, 330, 331, 332, 333,
       335, 363, 336, 340, 341, 343, 346, 353, 355, 356, 357, 359, 361,
       337, 369,   0, 371, 477, 475, 473, 472, 471, 469, 467, 478, 461,
       458, 457, 455, 452, 450, 449, 448, 460, 479, 480, 481, 507, 506,
       503, 501, 500, 499, 497, 496, 494, 493, 492, 491, 488, 485, 484,
       483, 482, 445, 443, 442, 439, 394, 390, 389, 388, 387, 386, 385,
       384, 383, 380, 379, 378, 377, 376, 374, 373, 372, 399, 370, 401,
       404, 438, 437, 435, 434, 433, 432, 430, 428, 425, 424, 422, 420,
       416, 414, 410, 406, 405, 402, 413, 255, 315, 112, 111, 108, 106,
       103, 101, 100,  99,  98, 115,  97,  94,  93,  92,  89,  88,  86,
        83,  82,  80,  95,  76, 116, 118, 153, 152, 150, 149, 148, 147,
       144, 143, 139, 117, 138, 133, 131, 130, 128, 127, 125, 123, 122,
       120, 135, 154,  74,  72,  36,  33,  32,  31,  29,  28,  27,  26,
        25,  39,  24,  20,  19,  16,  15,  13,  12,  10,   6,   4,  22,
        73,  42,  44,  71,  70,  69,  68,  67,  65,  64,  62,  60,  43,
        59,  54,  53,  52,  51,  49,  48,  47,  46,  45,  58, 155, 157,
       158, 276, 275, 273, 272, 271, 269, 268, 266, 265, 277, 264, 261,
       260, 257, 256, 510, 254, 253, 252, 248, 262, 247, 278, 280, 314,
       313, 311, 310, 309, 307, 303, 300, 298, 279, 297, 294, 292, 291,
       289, 288, 285, 284, 282, 281, 295, 246, 242, 239, 196, 195, 194,
       193, 192, 190, 189, 180, 179, 199, 178, 175, 173, 170, 167, 166,
       165, 164, 162, 159, 176, 201, 202, 203, 237, 236, 235, 232, 231,
       229, 227, 226, 225, 224, 223, 222, 220, 216, 215, 213, 212, 209,
       207, 206, 205, 316, 511] # dog_leg
    elif config.position_space == 'eye':
        if config.position_animal == 'cat':
            sort_index_cat= [1006,  821,  959,  586,  872,  597,  707,  994,  777,  851, 1009,
        939,  842,  772,  834, 1007,  560,  932,  427,  958,  820,  747,
        871,  725,  848,  559,  437,  960,  968,  694,  849,  532,  806,
        934,  992,  771,  805,  719,  979,  908,  726,  801,  809,  750,
        583,  817,  696,  818, 1012,  926,  657,  988,  766,  729,  547,
        877,  835,  956,  944,  863,  845,  892,  846,  918,  875,  759,
        737,  937,  757,  682,  981,  552,  803,  813,  524,  920,  711,
        882,  761,  847,  232,  562, 1002,  603,  656,  653,  800,  804,
        765,  584,  734,  923,  900,  578,  914,  955,  714,  952,  778,
        732,  791,  743,  940,  838,  942,  731,  897,  898,  935,  866,
        618,  693,  855,  984,  837,  143,  674,  867,  989,  781,  828,
         89,  651,  642,  980,  792,  790,  833,  826, 1019,  553,  844,
        786,  512,  736,  901, 1014,  754,  678,  409,  655,  676,  628,
        785,  715, 1011,  836,  856,  554,  542,  912,  963,  970,  513,
        675,  784,  529,  796,  974,  824,  997,  539,  961,  702, 1003,
        615,  660,  894,  893,  858,  741,  599,  880,  531,  710,  911,
        823,  680,  797,  969,  691,  982,  929,  646,  605,  950,  567,
        688,  819, 1010,  516,  534,  861,  841,  822,  787,  879,  999,
        756,  620,  807,  793,  966,  762,  751, 1021,  881,  745,  699,
        884,  864,  889,  938,  878,  870, 1017,  262,  706,  930,  722,
       1016,  582,  308,  910,  636,  465,  523,  852,  962,  991,  873,
        740,  839, 1018,  669,  735,  739,  527,  519,  659, 1005,  767,
        802,  721,  610,  464,  795, 1013,  883,  556,  906,  665,  623,
       1022,  965,  811,  780,  626,  588,  977,  608,  995,  886,  953,
        649,  684,  645,  705,  936,  593,  782,  783, 1015,  985,  663,
        810,  585,  859,  921,  831,  973,  843,  106,  574,  709,  924,
        640,  633,  648,  671,  799,  604,  990,  233,   17,  594,  561,
        909,  860,  829,  643,  354,  951,  449,  718,  515,  943,  212,
        925,  775,  209,  131,  716,  295,  967,  931,  546,  738,  876,
        832,  602,  548,  749,  946,  701,  896,  570,  888,  687,  891,
        862,  774,  752,  798,  609, 1004,  569,  895,  976,  779,  549,
        887,  463,  545, 1000,  904,  641,  723,  654,  840,  916,  868,
         34,  573,  903,  664,  683,  692,  971,  579,  661,  564,  170,
        708,  753,  975,  928,  557,  624,  748,  865,  812,  558,  139,
        996,  685,  814,  303,  228,  154,  788,  652,  907,  572,  698,
        954,  919,  720,  885,  690,  638, 1001,  306,  617,  945,  712,
        127,  598,  670,  933,  769,  447,  874,  742,  948,  613, 1008,
        854,  544,  816,  679,  317,  116,  135,  890,  830,  555,  147,
        727,  789,  435,  498,  566, 1020,  993,  672,  662,  681,  565,
        998,  947,  972,  915,  540,  587,  987,  563,  589,  590,  978,
        983,  913,  917,  986,  541,  949,  927,  571,  575,  905,  576,
        577,  922,  543,  538,  957,  580,  964,  551,  581,  550,  568,
        537,  941,  616,  592,  794,  658,  776,  666,  667,  668,  536,
        773,  673,  770,  768,  677,  764,  763,  760,  758,  686,  713,
        724,  728,  730,  704,  703,  808,  733,  697,  695,  744,  746,
        689,  755,  700,  650,  815,  647,  869,  619,  717,  614,  612,
        611,  621,  607,  899,  601,  600] # cat_eye
            sort_index_cat=[263, 461, 391, 336,  37, 395, 463, 392,  14, 120,  12, 112,  49,
       121, 400, 286,  90, 297, 378, 321, 100, 451, 305, 198,  25, 338,
       163,   9, 151,  11, 199, 229, 250, 145, 251, 186,   5, 401, 328,
       444, 399, 222, 139, 191,  47, 171,  64, 393,   8, 498,  16,  91,
       504, 136, 244, 440, 351, 182, 319, 274, 348, 503,   2, 149, 443,
       283, 465, 137,  33, 234,  61, 495, 111, 489, 456, 505, 174, 302,
       230, 124, 193, 375, 127, 442, 422,  56, 288, 441, 142, 287, 509,
       211,  75, 458, 206,  13,  71, 345, 204, 407, 405, 240, 219, 415,
       346, 347, 255, 223, 320,  18, 269, 354, 501, 210, 200, 179,  48,
       388, 466,  88, 226, 249,  66, 277, 214, 493, 429, 224, 421,  30,
       358, 379, 292, 218, 317, 252, 195, 280, 431, 490, 267, 228, 428,
       259, 194,  53, 303, 377, 329, 453, 150, 398, 486, 156, 268, 293,
       412, 408, 299, 188, 435,  19, 135,  77, 413, 478, 296, 363,  26,
       479, 232, 475, 380, 341, 238,  36, 510, 350, 390, 126, 107, 339,
       323,  72,  20, 459, 434, 371, 447,  82, 265,  23,  78,  63, 467,
       502, 233,  83,  28, 109,  31, 404, 500,  21, 340,  39, 289, 318,
       446,   4, 203, 481, 480, 477, 476, 474, 473, 472, 471, 470, 342,
       343, 344, 482, 468, 464, 462, 460, 457, 349, 455, 454, 452, 450,
       469, 337, 483, 484, 508, 507, 322, 506, 324, 325, 326, 327, 499,
       497, 496, 330, 494, 492, 331, 332, 491, 333, 488, 487, 334, 335,
       485, 449, 448, 382, 352, 420, 419, 418, 417, 416, 411, 372, 410,
       373, 409, 406, 423, 403, 402, 397, 396, 376, 394, 389, 387, 386,
       385, 384, 381, 374, 383, 370, 368, 353, 445, 439, 355, 356, 357,
       438, 437, 359, 436, 433, 369, 432, 360, 361, 362, 427, 426, 364,
       365, 366, 425, 367, 424, 430, 414,   0, 315, 116, 115, 114, 113,
       110, 108, 106, 105, 104, 117, 103, 101,  99,  98,  97,  96,  95,
        94,  93,  92, 102,  89, 118, 122, 154, 153, 152, 148, 147, 146,
       144, 143, 141, 119, 140, 134, 133, 132, 131, 130, 129, 128, 125,
       123, 138, 155,  87,  85,  44,  43,  42,  41,  40,  38,  35,  34,
        32,  45,  29,  24,  22,  17,  15,  10,   7,   6,   3,   1,  27,
        86,  46,  51,  84,  81,  80,  79,  76,  74,  73,  70,  69,  50,
        68,  65,  62,  60,  59,  58,  57,  55,  54,  52,  67, 157, 158,
       159, 275, 273, 272, 271, 270, 266, 264, 262, 261, 276, 260, 257,
       256, 254, 253, 248, 247, 246, 245, 243, 258, 242, 278, 281, 314,
       313, 312, 311, 310, 309, 308, 307, 306, 279, 304, 300, 298, 295,
       294, 291, 290, 285, 284, 282, 301, 241, 239, 237, 184, 183, 181,
       180, 178, 177, 176, 175, 173, 185, 172, 169, 168, 167, 166, 165,
       164, 162, 161, 160, 170, 187, 189, 190, 236, 235, 231, 227, 225,
       221, 220, 217, 216, 215, 213, 212, 209, 208, 207, 205, 202, 201,
       197, 196, 192, 316, 511]
        else:
            sort_index_cat=[145, 296, 217, 306, 417, 37, 9, 84, 462, 468, 489, 466, 1,
                                177, 429, 327, 258, 172, 344, 464, 126, 407, 8, 85, 301, 286,
                                348, 241, 142, 338, 233, 347, 487, 5, 251, 113, 454, 105, 199,
                                210, 112, 204, 107, 395, 124, 137, 171, 79, 211, 274, 392, 12,
                                481, 174, 465, 342, 186, 36, 156, 38, 351, 114, 221, 345, 52,
                                97, 218, 11, 160, 275, 205, 56, 498, 505, 102, 41, 317, 229,
                                354, 412, 91, 500, 111, 200, 391, 352, 441, 96, 367, 129, 259,
                                219, 191, 277, 61, 502, 451, 302, 140, 287, 182, 249, 422, 270,
                                305, 325, 299, 188, 238, 393, 230, 409, 127, 104, 3, 240, 228,
                                459, 203, 318, 139, 267, 486, 408, 419, 168, 503, 490, 436, 474,
                                121, 446, 315, 130, 387, 427, 255, 103, 227, 456, 434, 263, 208,
                                109, 470, 308, 201, 88, 321, 183, 136, 34, 329, 380, 310, 196,
                                134, 18, 144, 388, 316, 411, 508, 404, 132, 214, 402, 93, 71,
                                431, 262, 53, 179, 440, 328, 31, 457, 444, 119, 181, 421, 453,
                                405, 2, 279, 493, 509, 415, 268, 495, 141, 261, 243, 87, 50,
                                23, 187, 349, 390, 166, 269, 423, 39, 477, 398, 289, 283, 99,
                                303, 75, 116, 463, 163, 382, 371, 399, 438, 64, 245, 447, 312,
                                323, 445, 175, 151, 40, 122, 19, 484, 357, 360, 355, 359, 358,
                                480, 482, 356, 485, 483, 353, 424, 340, 350, 330, 331, 332, 333,
                                334, 507, 506, 504, 335, 336, 488, 337, 339, 499, 497, 341, 496,
                                343, 494, 492, 346, 491, 501, 479, 472, 362, 455, 452, 450, 394,
                                449, 448, 396, 397, 443, 442, 400, 401, 403, 406, 439, 437, 435,
                                410, 433, 432, 413, 414, 418, 430, 428, 420, 426, 458, 389, 460,
                                461, 363, 364, 478, 365, 366, 476, 368, 369, 370, 475, 473, 372,
                                373, 361, 374, 376, 425, 471, 377, 378, 379, 381, 469, 383, 467,
                                384, 385, 386, 375, 416, 0, 324, 101, 100, 98, 95, 94, 92,
                                90, 89, 106, 86, 82, 81, 80, 78, 77, 76, 74, 73, 83,
                                108, 110, 115, 154, 153, 152, 150, 149, 148, 147, 146, 143, 138,
                                135, 133, 131, 128, 125, 123, 120, 118, 117, 72, 70, 69, 68,
                                30, 29, 28, 27, 26, 25, 24, 22, 21, 20, 17, 16, 15,
                                14, 13, 10, 7, 6, 4, 32, 155, 33, 42, 67, 66, 65,
                                63, 62, 60, 59, 58, 57, 55, 54, 51, 49, 48, 47, 46,
                                45, 44, 43, 35, 326, 157, 159, 280, 278, 276, 273, 272, 271,
                                266, 265, 281, 264, 257, 256, 510, 254, 253, 252, 250, 248, 260,
                                282, 284, 285, 322, 320, 319, 314, 313, 311, 309, 307, 304, 300,
                                298, 297, 295, 294, 293, 292, 291, 290, 288, 247, 246, 244, 242,
                                195, 194, 193, 192, 190, 189, 185, 184, 180, 178, 176, 173, 170,
                                169, 167, 165, 164, 162, 161, 197, 158, 198, 206, 239, 237, 236,
                                235, 234, 232, 231, 226, 225, 224, 223, 222, 220, 216, 215, 213,
                                212, 209, 207, 202, 511] # dog_eye
    elif config.position_space == 'nose':
        if config.position_animal == 'cat':
            sort_index_cat=[461, 163,  14, 263, 392,  25,  90, 171,  16, 249, 378, 199,  49,
        12, 286, 503, 251, 391, 336, 329, 401, 191, 120,   9, 112, 358,
       288, 375, 305, 145, 429, 230, 139, 137, 294, 395,   2,   5, 234,
       347, 182, 297, 412, 283,  13, 136, 484, 338, 222, 250, 388, 204,
        40, 206, 399, 293, 451, 179, 142, 228,  11,  33, 100,  47, 328,
       107, 151, 219, 390,  48, 320,  53,   8, 278, 238, 486,  63, 421,
       440, 200, 224, 400, 321, 458, 441, 339, 229, 465, 287, 226, 274,
       442, 121, 267, 444, 240,  37, 269,  91, 415, 504,  56, 214,  61,
       302,  88, 379, 502, 443, 193, 277, 127, 111, 463, 405, 393,   4,
       285, 210, 198, 495, 398, 318,  78,  96, 292,  23, 453, 186, 456,
       299, 431, 275, 244, 509, 131, 490,  18, 134, 467, 307, 351, 498,
        77, 354, 281, 124, 259,  19,  64, 363, 497, 248, 422, 459, 350,
       505, 434, 356,  27, 282, 119, 435, 501,  75, 447, 272, 376, 109,
       218, 252, 407,  72, 114, 340, 346, 367, 489, 150, 156, 313, 233,
       129, 140, 331,  31, 319, 348, 457, 474, 450, 332, 473, 333, 334,
       472, 335, 471, 470, 337, 460, 452, 468, 466, 475, 454, 341, 462,
       342, 455, 343, 344, 345, 469, 464, 396, 477, 315, 316, 508, 317,
       507, 506, 500, 499, 496, 322, 494, 323, 324, 493, 492, 491, 325,
       326, 327, 488, 487, 485, 483, 482, 481, 480, 479, 478, 330, 449,
       476, 349, 448, 371, 372, 373, 417, 374, 416, 414, 413, 377, 314,
       411, 410, 409, 418, 408, 381, 406, 382, 404, 383, 384, 385, 386,
       403, 387, 402, 389, 397, 380, 394, 370, 368, 446, 445, 352, 353,
       439, 438, 437, 355, 436, 357, 433, 432, 359, 369, 430, 360, 427,
       361, 426, 362, 425, 424, 364, 423, 365, 366, 420, 419, 428,   0,
       255, 311, 108, 106, 105, 104, 103, 102, 101,  99,  98,  97,  95,
        94,  93,  92,  89,  87,  86,  85,  84,  83,  82, 110,  81, 113,
       116, 153, 152, 149, 148, 147, 146, 144, 143, 141, 138, 135, 133,
       132, 130, 128, 126, 125, 123, 122, 118, 117, 115, 154,  80,  76,
        39,  38,  36,  35,  34,  32,  30,  29,  28,  26,  24,  22,  21,
        20,  17,  15,  10,   7,   6,   3,   1,  41,  79,  42,  44,  74,
        73,  71,  70,  69,  68,  67,  66,  65,  62,  60,  59,  58,  57,
        55,  54,  52,  51,  50,  46,  45,  43, 312, 155, 158, 262, 261,
       260, 258, 257, 256, 510, 254, 253, 247, 246, 245, 243, 242, 241,
       239, 237, 236, 235, 232, 231, 264, 227, 265, 268, 310, 309, 308,
       306, 304, 303, 301, 300, 298, 296, 295, 291, 290, 289, 284, 280,
       279, 276, 273, 271, 270, 266, 157, 225, 221, 183, 181, 180, 178,
       177, 176, 175, 174, 173, 172, 170, 169, 168, 167, 166, 165, 164,
       162, 161, 160, 159, 184, 223, 185, 188, 220, 217, 216, 215, 213,
       212, 211, 209, 208, 207, 205, 203, 202, 201, 197, 196, 195, 194,
       192, 190, 189, 187, 511]
        else:
            sort_index_cat =[104, 352, 124, 217, 409, 317, 408, 347,  57,  84, 251, 318,  38,
         3, 151, 453,   8, 218, 509,   1, 466, 286, 396, 241, 171,  79,
        37, 197, 502, 198,  11, 395, 211,  14,  40,  23, 243, 156, 417,
       440, 181, 462, 465, 393, 136, 504,  17, 145, 468, 204, 186, 234,
       464,  96, 302, 102, 301, 490, 177,  91, 375, 244, 421,   9,   5,
       334, 188, 451, 367, 327, 446, 249, 258,  81, 349,  56, 351, 429,
       267, 487, 339, 121, 470, 263, 172, 214, 392, 436, 498, 403, 114,
       228, 110, 382, 208, 113, 338,  78, 191, 412, 259, 132, 119, 129,
       441, 381, 344,  50,   2, 141,   7, 230, 444, 329, 160, 304, 183,
       505, 342, 140, 137, 312, 463, 274, 146, 163, 508,  85, 182, 169,
       161, 250, 358,  41, 348, 240, 184, 489, 287, 210, 306, 398, 221,
       431, 476, 319, 305, 407, 290, 174, 419, 418, 427,  63, 109, 454,
       293,  61, 134, 328, 185, 495, 456,  66, 238, 426, 321, 299, 283,
       411,  18, 354,  90, 397, 296, 126,  55, 391, 474,  34,  35, 423,
       415, 308, 486, 350, 320, 200, 142, 459, 400, 105,  75, 187, 107,
       233, 360, 270, 168,  21, 447,  30, 245,  87, 219,  77, 365, 364,
       368, 366, 345, 362, 322, 323, 324, 325, 326, 330, 331, 332, 333,
       335, 363, 336, 340, 341, 343, 346, 353, 355, 356, 357, 359, 361,
       337, 369,   0, 371, 477, 475, 473, 472, 471, 469, 467, 478, 461,
       458, 457, 455, 452, 450, 449, 448, 460, 479, 480, 481, 507, 506,
       503, 501, 500, 499, 497, 496, 494, 493, 492, 491, 488, 485, 484,
       483, 482, 445, 443, 442, 439, 394, 390, 389, 388, 387, 386, 385,
       384, 383, 380, 379, 378, 377, 376, 374, 373, 372, 399, 370, 401,
       404, 438, 437, 435, 434, 433, 432, 430, 428, 425, 424, 422, 420,
       416, 414, 410, 406, 405, 402, 413, 255, 315, 112, 111, 108, 106,
       103, 101, 100,  99,  98, 115,  97,  94,  93,  92,  89,  88,  86,
        83,  82,  80,  95,  76, 116, 118, 153, 152, 150, 149, 148, 147,
       144, 143, 139, 117, 138, 133, 131, 130, 128, 127, 125, 123, 122,
       120, 135, 154,  74,  72,  36,  33,  32,  31,  29,  28,  27,  26,
        25,  39,  24,  20,  19,  16,  15,  13,  12,  10,   6,   4,  22,
        73,  42,  44,  71,  70,  69,  68,  67,  65,  64,  62,  60,  43,
        59,  54,  53,  52,  51,  49,  48,  47,  46,  45,  58, 155, 157,
       158, 276, 275, 273, 272, 271, 269, 268, 266, 265, 277, 264, 261,
       260, 257, 256, 510, 254, 253, 252, 248, 262, 247, 278, 280, 314,
       313, 311, 310, 309, 307, 303, 300, 298, 279, 297, 294, 292, 291,
       289, 288, 285, 284, 282, 281, 295, 246, 242, 239, 196, 195, 194,
       193, 192, 190, 189, 180, 179, 199, 178, 175, 173, 170, 167, 166,
       165, 164, 162, 159, 176, 201, 202, 203, 237, 236, 235, 232, 231,
       229, 227, 226, 225, 224, 223, 222, 220, 216, 215, 213, 212, 209,
       207, 206, 205, 316, 511]

    elif config.position_space == 'center':
        sort_index_cat=[156, 312, 405,  18, 476, 431, 358, 272,  94, 288, 266, 323,  43,
       182, 249, 151,  46, 484, 505,  13, 287,  19, 366,  14, 445, 292,
       303, 214,  12, 443,  53, 425, 123,  87, 423, 259, 299, 386, 322,
       144, 486,  72,  78,  64,   5, 285, 444,  26, 398, 245, 412,  35,
       497,  73, 483, 213,   3, 376, 234, 297, 500, 230, 453, 373, 464,
       275, 282, 324, 309, 109,   0, 509, 380, 194, 401,  93, 173, 430,
       434, 461, 252, 456,  56, 469, 219, 341, 267, 504, 240, 473, 200,
        22, 498,  50, 442, 455, 281,  92, 355, 231,  63, 216, 307, 187,
       354, 381,  80, 449, 479, 263,  88,  99, 319, 482,  75, 244, 190,
       336, 360, 385, 379,  21, 192,  32, 447, 438, 357, 178, 101, 458,
       328, 327, 459, 326, 452, 460, 344, 501, 502, 462, 345, 325, 457,
       330, 331, 332, 333, 321, 335, 343, 337, 338, 342, 450, 451, 339,
       340, 454, 329, 334, 428, 320, 302, 301, 300, 474, 508, 298, 475,
       477, 296, 295, 294, 293, 478, 485, 480, 291, 290, 481, 289, 487,
       507, 472, 304, 463, 465, 318, 317, 316, 503, 315, 314, 466, 346,
       313, 311, 310, 468, 308, 506, 470, 306, 305, 471, 467, 347, 353,
       349, 407, 406, 494, 404, 403, 402, 433, 408, 400, 399, 435, 397,
       396, 395, 394, 496, 495, 409, 410, 411, 427, 426, 492, 424, 493,
       422, 421, 420, 491, 419, 418, 417, 416, 415, 414, 413, 432, 393,
       348, 392, 390, 489, 365, 364, 363, 362, 361, 359, 490, 446, 356,
       488, 448, 429, 352, 351, 350, 499, 367, 368, 369, 436, 389, 388,
       387, 384, 383, 437, 439, 378, 377, 375, 374, 440, 441, 372, 371,
       370, 391, 382, 255, 284, 111, 110, 108, 107, 106, 105, 104, 103,
       102, 100,  98,  97,  96,  95,  91,  90,  89,  86,  85,  84,  83,
        82,  81, 112, 113, 114, 115, 140, 139, 138, 137, 136, 135, 134,
       133, 132, 131, 130,  79, 129, 127, 126, 125, 124, 122, 121, 120,
       119, 118, 117, 116, 128,  77,  76,  74,  36,  34,  33,  31,  30,
        29,  28,  27,  25,  24,  23,  37,  20,  16,  15,  11,  10,   9,
         8,   7,   6,   4,   2,   1,  17, 141,  38,  40,  71,  70,  69,
        68,  67,  66,  65,  62,  61,  60,  59,  39,  58,  55,  54,  52,
        51,  49,  48,  47,  45,  44,  42,  41,  57, 142, 143, 145, 243,
       242, 241, 239, 238, 237, 236, 235, 233, 232, 229, 246, 228, 226,
       225, 224, 223, 222, 221, 220, 218, 217, 215, 212, 227, 211, 247,
       250, 283, 280, 279, 278, 277, 276, 274, 273, 271, 270, 269, 248,
       268, 264, 262, 261, 260, 258, 257, 256, 510, 254, 253, 251, 265,
       286, 210, 208, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162,
       161, 172, 160, 158, 157, 155, 154, 153, 152, 150, 149, 148, 147,
       146, 159, 209, 174, 176, 207, 206, 205, 204, 203, 202, 201, 199,
       198, 197, 196, 175, 195, 191, 189, 188, 186, 185, 184, 183, 181,
       180, 179, 177, 193, 511]

    elif config.position_space == 'petal':
        sort_index_cat=[156, 405, 431, 445, 312,  64, 259,  19,  18,  13,  73, 476, 386,
       299, 249, 123,   0,  94, 505, 358, 272, 498, 109, 285, 213,  43,
       136, 328, 442, 360, 481, 438, 456, 355,  88, 151, 464,  35,  26,
       398, 192,  72, 303, 469, 191,   5, 322,  93, 449,  99, 376, 240,
        46, 373, 211, 467, 245, 292, 326, 267, 214, 354, 453, 144,  53,
       287, 340, 380, 228, 444, 486, 309, 244,  10,  22, 255, 269, 423,
        87, 448, 434, 234,  14, 382, 379,  85, 368, 457, 357, 392, 484,
       332, 412, 363, 325, 218, 254, 485, 411, 182,  50, 420, 421, 443,
       315, 293, 277, 465, 148, 154, 186, 187, 210, 281, 320,  92, 446,
       341, 163,  62, 173, 436, 339, 324, 288, 350, 366, 178, 451,  75,
       478,  63,  61, 473,  80, 251, 310, 497, 455,  90, 195,  12, 345,
        78, 490, 206,  40, 219,  39, 441, 461, 252, 230, 111, 447, 330,
       437, 440, 508, 370, 361, 369, 503, 367, 364, 365, 362, 371, 316,
       509, 504, 335,  29, 374, 375, 317, 314, 502, 377, 378, 501, 313,
        27, 381, 383, 372,  28, 359, 319, 346, 384, 327, 344,  36, 343,
        37, 342,  34, 331, 333, 334, 338, 337, 329, 347, 348, 349,  30,
         1, 336,  31, 506, 321, 356,  38, 507,  32,  33, 353, 323, 352,
       351, 318, 385,   3, 387, 459, 458,  11, 454, 452, 450, 491, 492,
        15,  16, 493,  17, 439,  20, 435, 494, 433, 460, 462, 463,   9,
       488, 489, 483, 482,   4, 480, 479, 477, 432,   6, 474, 472, 471,
       470,   7, 468,   8, 466, 475,  21, 430, 429,  23, 404, 403, 402,
       401, 400, 399, 487, 406, 397, 395, 394, 393, 391, 390, 500, 389,
       388, 396,  25, 407, 409, 428, 427, 426, 425, 424, 495, 422, 496,
       408, 419, 418, 417, 416, 415, 414, 413, 499, 410,   2,  24,  76,
       308, 158, 157, 155, 153, 152, 150, 149, 147, 146, 145,  65, 143,
       142, 141, 140, 139, 138, 137,  66, 135, 134, 159, 133, 160, 162,
       189, 188, 185, 184, 183, 181, 180, 179, 177, 176, 175, 174, 172,
       171, 170, 169, 168, 167, 166, 165, 164, 161, 190, 132, 130, 103,
       102, 101, 100,  69,  98,  97,  96,  95,  70,  71,  91,  89,  74,
        86,  84,  83,  82,  81,  79,  77, 104, 131, 105, 107, 129, 128,
       127, 126, 125, 124,  67, 122, 121, 120, 119, 118, 117, 116, 115,
       114, 113, 112, 110,  68, 108, 106,  60, 193, 194, 279, 278, 276,
       275, 274, 273,  48, 271, 270, 268,  49, 266, 265, 264, 263, 262,
       261, 260, 258, 257, 256, 280, 510, 282, 284, 307, 306, 305, 304,
        41, 302, 301, 300,  42, 298, 297, 296, 295, 294,  44, 291, 290,
       289,  45, 286,  47, 283, 253, 250,  51, 220, 217, 216, 215,  57,
        58, 212,  59, 209, 208, 207, 205, 204, 203, 202, 201, 200, 199,
       198, 197, 196, 221, 222, 223, 224, 248, 247, 246,  52,  54, 243,
       242, 241,  55, 239, 311, 238, 236, 235, 233, 232, 231, 229,  56,
       227, 226, 225, 237, 511]

    elif config.position_space == 'blade':
        sort_index_cat = [156,  35,  56,  88, 392, 456,  61, 411,  19, 421,  46,  18, 386,
       228, 286, 186, 267, 340, 348, 434, 457, 307, 412,  50, 360, 218,
       442, 323, 431, 405, 508, 312, 211, 388, 109, 481, 398, 350, 492,
       325, 277, 446, 226, 240, 285,  44, 136, 500, 382, 130, 453,  26,
       357,  53, 269, 448,  72,  43,   2, 190, 179, 465, 495,  14, 320,
       272,  90, 191, 214,  10,  21,  73, 243, 210, 279, 168, 177, 309,
        75, 339,   0, 292, 328, 449, 268, 224, 371, 310,  23, 438, 213,
        28, 287, 379, 101, 206,  42, 464, 441, 447, 305, 208, 151, 204,
       494, 230, 507, 114,  95,  78, 117, 396, 311, 259, 173, 369, 252,
       478, 324, 346,  93, 345, 289, 200, 427, 354, 451, 493, 351, 436,
       238, 115, 413, 367, 232, 170, 123, 475, 498, 251, 383, 165, 376,
       326, 148, 178,  83, 420, 393, 174, 469, 284, 111, 303, 205, 406,
       510, 422, 419,  13, 459, 380, 254, 505, 428, 322, 150, 295, 399,
        99, 304, 331, 302, 355, 332, 359, 358, 329, 356, 306, 333, 352,
       335, 336, 337, 338, 321, 319, 318, 317, 341, 342, 343, 344, 316,
       347, 315, 314, 330, 334, 313, 349, 308, 353, 327, 400, 362, 473,
       472, 471, 470, 468, 467, 466, 463, 462, 461, 460, 458, 455, 454,
       452, 450, 445, 444, 443, 474, 476, 477, 479, 509, 506, 504, 503,
       502, 501, 499, 497, 496, 440, 491, 489, 488, 487, 486, 485, 484,
       483, 482, 480, 490, 439, 437, 435, 394, 391, 390, 389, 387, 385,
       384, 381, 378, 395, 377, 374, 373, 372, 370, 368, 366, 365, 364,
       363, 375, 361, 397, 401, 433, 432, 430, 429, 426, 425, 424, 423,
       418, 301, 417, 415, 414, 410, 409, 408, 407, 404, 403, 402, 416,
       300, 255, 298, 105, 104, 103, 102, 100,  98,  97,  96,  94,  92,
       106,  91,  87,  86,  85,  84,  82,  81,  80,  79,  77,  76,  89,
       107, 108, 110, 140, 139, 138, 137, 135, 134, 133, 132, 131, 129,
       128, 127, 126, 125, 124, 122, 121, 120, 119, 118, 116, 113, 112,
        74, 141,  71,  69,  32,  31,  30,  29,  27,  25,  24,  22,  20,
        17,  33,  16,  12,  11,   9,   8,   7,   6,   5,   4,   3,   1,
        15,  34,  36,  37,  68,  67,  66,  65,  64,  63,  62,  60,  59,
        58,  57,  55,  54,  52,  51,  49,  48,  47,  45,  41,  40,  39,
        38,  70, 299, 142, 144, 256, 253, 250, 249, 248, 247, 246, 245,
       244, 242, 257, 241, 237, 236, 235, 234, 233, 231, 229, 227, 225,
       223, 239, 258, 260, 261, 297, 296, 294, 293, 291, 290, 288, 283,
       282, 281, 280, 278, 276, 275, 274, 273, 271, 270, 266, 265, 264,
       263, 262, 222, 143, 221, 219, 175, 172, 171, 169, 167, 166, 164,
       163, 162, 161, 176, 160, 158, 157, 155, 154, 153, 152, 149, 147,
       146, 145, 159, 180, 181, 182, 217, 216, 215, 212, 209, 207, 203,
       202, 201, 199, 198, 197, 196, 195, 194, 193, 192, 189, 188, 187,
       185, 184, 183, 220, 511]

    elif config.position_space == 'root':
        sort_index_cat = [156,  88, 278, 442,  21, 307, 357,  46, 350, 326, 456, 398, 340,
       432, 136, 448, 224, 277, 411, 421, 388, 435, 320,  18, 214, 191,
        14,  19,  50, 369, 286,  43,  26,  76,  13, 325, 116, 492, 441,
       360, 427, 377, 346, 281,  78, 431, 392, 178, 312,  56, 345,  72,
       490, 154, 430, 367, 331, 269,  75, 179,  36,  27, 199, 500, 379,
       481,  53, 226,   6, 125, 353, 420,  80,  35, 170,  25,   0, 453,
       382, 167, 413, 319, 186, 157, 210, 311, 508,   7, 446, 100, 381,
       184, 412,  62, 223,  32, 221, 267, 141, 292, 455, 272, 472, 287,
       173,  61, 348, 252, 118, 507, 229, 115, 189, 275, 310, 329, 439,
       390,  37, 190, 205, 475, 505, 488, 489, 339, 330, 334, 338, 337,
       333, 336, 335, 332, 491, 493, 434, 341, 356, 355, 354, 480, 352,
       351, 482, 487, 483, 347, 484, 485, 486, 328, 343, 342, 349, 344,
       324, 494, 302, 301, 502, 300, 503, 299, 298, 504, 303, 297, 295,
       506, 294, 293, 509, 291, 290, 289, 296, 327, 304, 306, 495, 479,
       323, 322, 321, 496, 497, 318, 305, 317, 315, 314, 313, 498, 499,
       309, 308, 501, 316, 358, 362, 359, 450, 451, 410, 409, 408, 407,
       452, 406, 449, 405, 403, 454, 402, 457, 400, 399, 458, 397, 404,
       396, 414, 416, 433, 437, 438, 429, 428, 440, 426, 425, 415, 424,
       423, 422, 444, 445, 419, 418, 417, 447, 443, 478, 395, 393, 373,
       469, 470, 372, 371, 370, 471, 473, 374, 368, 366, 476, 365, 364,
       363, 436, 361, 477, 474, 394, 375, 378, 459, 391, 460, 461, 462,
       389, 463, 387, 376, 386, 384, 383, 464, 465, 466, 380, 467, 468,
       385, 401, 255, 285, 108, 107, 106, 105, 104, 103, 102, 101,  99,
        98,  97, 109,  96,  94,  93,  92,  91,  90,  89,  87,  86,  85,
        84,  83,  95,  82, 110, 112, 142, 140, 139, 138, 137, 135, 134,
       133, 132, 131, 130, 111, 129, 127, 126, 124, 123, 122, 121, 120,
       119, 117, 114, 113, 128, 143,  81,  77,  38,  34,  33,  31,  30,
        29,  28,  24,  23,  22,  20,  39,  17,  15,  12,  11,  10,   9,
         8,   5,   4,   3,   2,   1,  16,  79,  40,  42,  74,  73,  71,
        70,  69,  68,  67,  66,  65,  64,  63,  41,  60,  58,  57,  55,
        54,  52,  51,  49,  48,  47,  45,  44,  59, 288, 144, 146, 248,
       247, 246, 245, 244, 243, 242, 241, 240, 239, 238, 249, 237, 235,
       234, 233, 232, 231, 230, 228, 227, 225, 222, 220, 236, 219, 250,
       253, 284, 283, 282, 280, 279, 276, 274, 273, 271, 270, 268, 251,
       266, 264, 263, 262, 261, 260, 259, 258, 257, 256, 510, 254, 265,
       145, 218, 216, 176, 175, 174, 172, 171, 169, 168, 166, 165, 164,
       163, 177, 162, 160, 159, 158, 155, 153, 152, 151, 150, 149, 148,
       147, 161, 217, 180, 182, 215, 213, 212, 211, 209, 208, 207, 206,
       204, 203, 202, 181, 201, 198, 197, 196, 195, 194, 193, 192, 188,
       187, 185, 183, 200, 511]

    for i in range(512):
        eye_feature[0,sort_index_cat[i],0,0]=error[0,sort_index_cat[i],0,0]
    eye_feature=eye_feature*30/np.max(eye_feature)
    plt.style.use('ggplot')
    plt.figure(4, figsize=(4, 4), dpi=300)
    plt.bar(range(N_select), eye_feature[0,:,0,0])

    # for i in range(len(sort_index_cat)):
    #     plt.text(sort_index_cat[i],
    #              (feature_PCA_origin[0, :, 0, 0].cpu().data.numpy() - feature_PCA[0, :, 0, 0].cpu().data.numpy())[
    #                  sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
    #              fontproperties='Arial', fontsize=8)

    plt.xlabel('Neuron Index', font1)
    plt.ylabel('Scores of 1st PC', font1)
    # plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}_only.tiff',
    #             bbox_inches='tight', dpi=300)
    # plt.savefig(f'PPT_fig/position/difference_{config.position_space}_{config.position_animal}_only.pdf',
    #             bbox_inches='tight', dpi=300)

    plt.show()
    inverse_pic, inverse_feature = picture_inverse(use_model, torch.from_numpy(eye_feature.astype((np.float32))).to(config.device), conv_out, max_epoch=4000)
    inverse_pic_plot = transform_raw_picture(inverse_pic)



    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_only.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_only.pdf',
                bbox_inches='tight', dpi=300)  
    plt.show()

    inverse_pic, inverse_feature = picture_inverse(use_model, feature_PCA.to(config.device),conv_out,max_epoch=4000)
    inverse_pic_plot = transform_raw_picture(inverse_pic)

    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_no.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_no.pdf',
                bbox_inches='tight', dpi=300)
    # plt.show()

    inverse_pic, inverse_feature = picture_inverse(use_model, feature_PCA_origin.to(config.device), conv_out, max_epoch=4000) 
    inverse_pic_plot = transform_raw_picture(inverse_pic)

    plt.style.use('default')
    plt.figure(5, dpi=300)
    plt.axis('off')
    plt.imshow(np.clip(inverse_pic_plot, 0, 1))
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_yes.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../lime_save_2kinds/inverse_pic_{config.position_space}_save_{config.position_animal}_yes.pdf',
                bbox_inches='tight', dpi=300)
    # plt.show()


def compare_different_num_inverse(use_model,conv_out,N_select):
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 6,
             }
    data_num=[2,5,20,50,100,300,500]
    max_y=[3,5,8,14,18,32,40]

    seed=1101 # 525, 324, 1101
    PCA_save=np.zeros([7,512])
    for i in range(len(data_num)):
        config.PCA_data_num = data_num[i]
        generate_feature_matrix(use_model,conv_out,seed)
        PCA_feature = load_PCA()
        PCA_feature=PCA_feature#*30/torch.max(PCA_feature)
        PCA_save[i]=PCA_feature[0, :, 0, 0].cpu().data.numpy()
        fig=plt.figure(1, figsize=(1.2, 1.2), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.bar(range(N_select), PCA_feature[0, :, 0, 0].cpu().data.numpy().reshape(512), color='blue')
        sort_index_cat = np.argsort(-np.abs(PCA_feature[0, :, 0, 0].cpu().data.numpy()))[0:10]
        print((PCA_feature[0, :, 0, 0].cpu().data.numpy())[sort_index_cat])
        print(PCA_feature[0, :, 0, 0].cpu().data.numpy().max())

        # for i in range(len(sort_index_cat)):
        #     plt.text(sort_index_cat[i],
        #              (PCA_feature[0, :, 0, 0].cpu().data.numpy())[
        #                  sort_index_cat[i]] + 0.25, '%s' % round(np.round(sort_index_cat[i], 1), 3), ha='center',
        #              fontproperties='Arial', fontsize=8)
        plt.ylim(0,max_y[i])

        # plt.xlabel('Neuron Index', font1)
        # plt.ylabel('Scores of 1st PC', font1)
        import matplotlib.ticker as mtick

        plt.xticks(np.arange(0,512,100),np.arange(0,512,100),fontproperties='Arial', size=6)
        plt.yticks(np.linspace(0,max_y[i],5),np.linspace(0,max_y[i],5),fontproperties='Arial', size=6)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        # plt.savefig(f'PPT_fig/paper/neuron_{config.PCA_animal}_{config.PCA_data_num}_{seed}.tiff',
        #             bbox_inches='tight', dpi=300)
        # plt.savefig(f'PPT_fig/paper/neuron_{config.PCA_animal}_{config.PCA_data_num}_{seed}.pdf',
        #             bbox_inches='tight', dpi=300)
        # plt.show()
    np.save(f'../lime_save_2kinds/paper/PCA_save_{config.PCA_animal}_{config.PCA_data_num}_{seed}.npy',PCA_save)

def plot_N():
    data_num = [2, 5, 20, 50, 100, 300, 500]
    PCA_525=np.load('../lime_save_2kinds/paper/PCA_save_cat_500_525.npy')

    PCA_324=np.load('../lime_save_2kinds/paper/PCA_save_cat_500_324.npy')

    PCA_1101=np.load('../lime_save_2kinds/paper/PCA_save_cat_500_1101.npy')
    error=[]
    for i in range(7):
        average=(PCA_525[i]+PCA_324[i]+PCA_1101[i])/3
        relative_error=((np.abs(PCA_525[i]-average)+np.abs(PCA_324[i]-average)+np.abs(PCA_1101[i]-average)).sum())/(3*average.sum())
        error.append(relative_error)
        print(relative_error)
    fig = plt.figure(1, figsize=(3, 1.5), dpi=300)
    x = np.linspace(0, 1, 7)
    plt.plot(x,error,c='blue',zorder=1)
    plt.scatter(x, error,marker='x', c='red',s=15,zorder=2)
    plt.xticks(x, data_num, fontproperties='Arial', size=7)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8], ['0%', '20%', "40%", "60%", "80%"], fontproperties='Arial', size=7)
    for i in range(len(data_num)):
        plt.text(data_num[i], error[i] + 0.02, '%s' % round(error[i], 3), ha='center',
                 fontproperties='Arial', fontsize=8)
    # plt.savefig('PPT_fig/paper/PCA_number_save.tiff', bbox_inches='tight',  dpi=300)
    # plt.savefig('PPT_fig/paper/PCA_number_save.pdf', bbox_inches='tight', dpi=300)
    plt.show()

def plot_radian(data,name):
    plt.style.use('ggplot')

    # data
    labels = np.array(['dog leg', 'dog eye', 'cat ear', 'cat nose', 'cat leg', 'cat eye', 'dog ear', 'dog nose'])
    # labels = np.array(['cat eye', 'cat leg', 'cat ear'])
    labels = np.concatenate((labels, [labels[0]]))  

    dataLenth = 8
    angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]]))  
    angles = np.concatenate((angles, [angles[0]])) 
    # plot
    plt.rcParams['font.size'] = 5
    plt.rcParams['font.family']='Arial'
    plt.rcParams['font.style']='italic'
    #plt.rcParams['font.weight'] = 'bold'
    fig = plt.figure(figsize=(4, 3),dpi=400)
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, data, '--',color='r', linewidth=1) 
    ax.scatter(angles,data,c='r',s=15)
    ax.fill(angles, data, facecolor='r', alpha=0.35)  
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="Arial", fontsize=8,fontstyle='normal')
    ax.set_rlim(0,1)
    # ax.set_title("matplotlib雷达图", va='bottom', fontproperties="SimHei",fontsize=22)
    ax.grid(True)
    plt.savefig(f'../PPT_fig/radian_{name}.tiff',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'../PPT_fig/radian_{name}.pdf',
                bbox_inches='tight', dpi=300)
    plt.show()

def plot_BubbleChart(data_new,name):
# def plot_BubbleChart(data_1,data_2,name):
    plt.figure(figsize=(6, 6), dpi=300)
    plt.rcParams['font.family'] = 'Arial'
    font1 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 18,
             }
    font2 = {'family': 'Arial',
             'weight': 'normal',
             # "style": 'italic',
             'size': 15,
             }

    size = 0.2
    data = [[60., 32.]]

    vals = np.array(data)

    # cat_values = data_1
    # dog_values = data_2
    cat_values = data_new[:4]
    dog_values = data_new[4:8]
    cat_sum = np.sum(cat_values)
    dog_sum = np.sum(dog_values)

    plt.pie(vals.sum(axis=1), radius=0.5, colors=['#EBF0F2'],  # #EFF4FF, #EBF0F2, #F2EFDF, #ECF2FF, #F2F2F2
            wedgeprops=dict(width=0.15, edgecolor='w'))

    radius_inner = 0.5
    center_inner = (0, 0)

    cat_radius = radius_inner - 0.1  
    dog_radius = radius_inner - 0.2  
    cat_center = (center_inner[0] - radius_inner + 0.1, center_inner[1])  
    dog_center = (center_inner[0] + radius_inner - 0.1, center_inner[1])  

    if dog_sum > cat_sum:
        plt.scatter(*cat_center, s=2500, c='#4C81BA', alpha=1)  # #3A80BA,#857F71,#8698B8,#7A9190
        plt.scatter(*dog_center, s=2500, c='#2E4E70', alpha=1)  # #3A80BA, #59554C,#3A7D8C,#44668C,#38474D
        # plt.scatter(*cat_center, s=2500, c='#9096A6', alpha=1)  # #3A80BA,#857F71,#8698B8,#7A9190
        # plt.scatter(*dog_center, s=2500, c='#B8A9C0', alpha=1)  # #3A80BA, #59554C,#3A7D8C,#44668C,#38474D

    else:
        plt.scatter(*cat_center, s=2500, c='#2E4E70', alpha=1)  # dodgerblue, cornflowerblue
        plt.scatter(*dog_center, s=2500, c='#4C81BA', alpha=1)
        # plt.scatter(*cat_center, s=2500, c='#B8A9C0', alpha=1)  # dodgerblue, cornflowerblue
        # plt.scatter(*dog_center, s=2500, c='#9096A6', alpha=1)

    plt.text(cat_center[0], cat_center[1], f'cat\n{cat_sum:.2f}', color='white', ha='center', va='center',
             fontdict=font2, weight='bold')  # color='white',
    plt.text(dog_center[0], dog_center[1], f'dog\n{dog_sum:.2f}', color='white', ha='center', va='center',
             fontdict=font2, weight='bold')  # color='white',

    plt.pie(vals.sum(axis=1), radius=1, autopct='', pctdistance=0.8, colors=['#EFF4FF'],  # #EFF4FF, #F7FCED, #FFF7F6
            wedgeprops=dict(width=0.2, edgecolor='w'))

    bubble_radius = 0.9  
    num_bubbles = 8 

    angle_interval = 360 / num_bubbles

    # left_bubble_angles = np.arange(110, 250, angle_interval)
    # right_bubble_angles = np.arange(290, 430, angle_interval)
    left_bubble_angles = np.linspace(110, 250, 4)
    right_bubble_angles = np.linspace(290, 430, 4)

    if dog_sum > cat_sum:

        left_bubble_colors = ['#7EB5D6', '#7EB5D6', '#7EB5D6', '#7EB5D6']
        right_bubble_colors = ['#678AA8', '#678AA8', '#678AA8', '#678AA8']

        # left_bubble_colors = ['#AEBFBE','#AEBFBE','#AEBFBE','#AEBFBE']  # #7EB5D6,#798C8C
        # right_bubble_colors = ['#798C8C','#798C8C','#798C8C','#798C8C']  # #678AA8

        # left_bubble_colors = ['#D4A19F','#D4A19F','#D4A19F','#D4A19F']
        # right_bubble_colors = ['#9C818A','#9C818A','#9C818A','#9C818A']

        # right_bubble_colors = ['#58A18C', '#58A18C','#58A18C','#58A18C']
        # left_bubble_colors = ['#A4D3D1','#A4D3D1','#A4D3D1','#A4D3D1']

        # right_bubble_colors = ['#A1A19C','#A1A19C','#A1A19C','#A1A19C']
        # left_bubble_colors = ['#D4D4CD','#D4D4CD','#D4D4CD','#D4D4CD']

        # left_bubble_colors = ['#C3D6E0', '#C3D6E0', '#C3D6E0', '#C3D6E0']
        # right_bubble_colors = ['#C7DBD2', '#C7DBD2', '#C7DBD2', '#C7DBD2']

    else:
        right_bubble_colors = ['#7EB5D6', '#7EB5D6', '#7EB5D6', '#7EB5D6']
        left_bubble_colors = ['#678AA8', '#678AA8', '#678AA8', '#678AA8']
        # right_bubble_colors = ['#C3D6E0', '#C3D6E0', '#C3D6E0', '#C3D6E0']
        # left_bubble_colors = ['#C7DBD2', '#C7DBD2', '#C7DBD2', '#C7DBD2']

    bubble_size = 2000

    cat_x = np.cos(np.radians(180)) * cat_radius
    cat_y = np.sin(np.radians(180)) * cat_radius

    dog_x = np.cos(np.radians(0)) * dog_radius
    dog_y = np.sin(np.radians(0)) * dog_radius


    for i, angle in enumerate(left_bubble_angles):
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        plt.scatter(x * bubble_radius, y * bubble_radius, s=bubble_size, c=left_bubble_colors[i], alpha=1)
        arrow_length = 0.1  
        arrow_begin_x = [x * bubble_radius + 0.05, x * bubble_radius + 0.13, x * bubble_radius + 0.13,
                         x * bubble_radius + 0.05]
        arrow_begin_y = [y * bubble_radius - 0.13, y * bubble_radius - 0.05, y * bubble_radius + 0.05,
                         y * bubble_radius + 0.12]
        arrow_end_x = [cat_x + x * arrow_length - 0.09, cat_x + x * arrow_length - 0.06,
                       cat_x + x * arrow_length - 0.06, cat_x + x * arrow_length - 0.08]
        arrow_end_y = [cat_y + y * arrow_length, cat_y + y * arrow_length, cat_y + y * arrow_length,
                       cat_y + y * arrow_length - 0.03]
        # arrow_end_x = cat_x + x * arrow_length
        # arrow_end_y = cat_y + y * arrow_length
        arrow = patches.FancyArrowPatch((arrow_begin_x[i], arrow_begin_y[i]), (arrow_end_x[i], arrow_end_y[i]),  # -0.07
                                        connectionstyle="arc3,rad=0.2", edgecolor=left_bubble_colors[0],
                                        facecolor=left_bubble_colors[0],
                                        arrowstyle='simple', mutation_scale=13, linewidth=0.5)
        plt.gca().add_patch(arrow)
        plt.text(x * bubble_radius, y * bubble_radius, f'{cat_values[i]:.2f}', color='white', ha='center', va='center',
                 fontdict=font2, weight='bold')  # , color='white'

    for i, angle in enumerate(right_bubble_angles):
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        plt.scatter(x * bubble_radius, y * bubble_radius, s=bubble_size, c=right_bubble_colors[i], alpha=1)
        arrow_length = 0.1 
        arrow_begin_x = [x * bubble_radius - 0.05, x * bubble_radius - 0.14, x * bubble_radius - 0.13,
                         x * bubble_radius - 0.05]
        arrow_begin_y = [y * bubble_radius + 0.13, y * bubble_radius + 0.03, y * bubble_radius - 0.05,
                         y * bubble_radius - 0.13]
        arrow_end_x = [dog_x + 0.24 + x * arrow_length - 0.05, dog_x + 0.21 + x * arrow_length - 0.05,
                       dog_x + 0.2 + x * arrow_length - 0.04, dog_x + 0.24 + x * arrow_length - 0.05]
        arrow_end_y = [dog_y + y * arrow_length, dog_y + y * arrow_length, dog_y + y * arrow_length,
                       dog_y + y * arrow_length + 0.025]
        # arrow_end_x = dog_x + x * arrow_length
        # arrow_end_y = dog_y + y * arrow_length
        arrow = patches.FancyArrowPatch((arrow_begin_x[i], arrow_begin_y[i]), (arrow_end_x[i], arrow_end_y[i]),
                                        connectionstyle="arc3,rad=0.2", edgecolor=right_bubble_colors[0],
                                        facecolor=right_bubble_colors[0],
                                        arrowstyle='simple', mutation_scale=13, linewidth=0.5)
        plt.gca().add_patch(arrow)
        plt.text(x * bubble_radius, y * bubble_radius, f'{dog_values[i]:.2f}', color='white', ha='center', va='center',
                 fontdict=font2, weight='bold')

    outer_radius = 1.13 
    outer_labels = ["eye", "leg", "ear", "nose", "nose", "ear", "leg", "eye"]
    outer_label_angles = np.linspace(110, 475, num_bubbles + 1)[:-1] 

    for angle, label in zip(outer_label_angles, outer_labels):
        x = np.cos(np.radians(angle))
        y = np.sin(np.radians(angle))
        plt.text(x * outer_radius, y * outer_radius, label, ha='center', va='center', fontdict=font2, weight='bold')

    plt.title('AS-XAI',fontsize=15,fontweight='bold')
    plt.axis('equal')
    plt.savefig(f'../lime_save_2kinds/XAI_{name}.tiff', bbox_inches='tight', dpi=300)
    plt.show()


def transform_pro(prediction):
    prediction=torch.flatten(prediction)
    prediction_cat=torch.exp(prediction[0])/(torch.exp(prediction[0])+torch.exp(prediction[1]))
    prediction_dog = torch.exp(prediction[1]) / (torch.exp(prediction[0]) + torch.exp(prediction[1]))
    return prediction_cat,prediction_dog



def print_word(conv_out,reshape_num):
    config.dataset_situation = 'val'
    # adverse_sample = torch.from_numpy(np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-fake-dog-PGD-0.05.npy').astype(np.float32))
    # true_sample = torch.from_numpy(np.load(r'D:\pycharm project\VAE_PDE\fake_picture\cat-true-dog-PGD-0.01.npy').astype(np.float32))
    for i, data in enumerate(test_loader):
        # if i<19:
        #     continue
        print('----------第%d次迭代----------' % (i))
        image, label = data
        #image=true_sample[i]
        transform = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        #fname = r'D:\pycharm project\VAE_PDE\big_cat_1.jpg'
        file_name='val'
        # image = Image.open(fname)
        # image = transform(image).reshape([1, 3, 224, 224])


        image = Variable(image.cuda(), requires_grad=True)
        label_true, _, x_feature = use_model(image)

        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.cuda().reshape(reshape_num)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')

        cdf_record=[]
        scale_factor = {
            ('dog', 'eye'): 0.4412,
            ('dog', 'leg'): 0.1102,
            ('dog', 'ear'): 0.1775,
            ('dog', 'nose'): 0.2717,
            ('cat', 'eye'): 0.3811,
            ('cat', 'leg'): 0.1407,
            ('cat', 'ear'): 0.2938,
            ('cat', 'nose'): 0.1787
        }
        for i in ['eye', 'leg', 'ear', 'nose']:
            for j in ['dog','cat']:
                config.position_space = i
                config.position_animal = j
                _, _, space_index, space_value = get_position_2(conv_out, 5, show_picture=False)
                space_index = np.array(space_index, dtype=int)
                space = np.load(f'../result_save_2kinds/{config.position_animal}_space_2000.npy')

                eye_space = space[:, space_index]
                sum = 0
                for k in range(eye_space.shape[1]):
                    sum += eye_space[:, k] * space_value[k]
                mean = sum / eye_space.shape[1]

                eye_space = feature_space[space_index].cpu().data.numpy()
                sum = 0
                for k in range(eye_space.shape[0]):
                    sum += eye_space[k] * space_value[k]
                mean_pic = sum / eye_space.shape[0]
                # cdf_record.append(get_cdf_ratio_2(mean_pic, mean))
                # print(f'{j},{i},{get_cdf_ratio_2(mean_pic, mean)}')
                cdf_record.append(get_cdf_ratio(mean_pic,mean, scale_factor[(config.position_animal, config.position_space)]))
                print(f'{j},{i},{get_cdf_ratio(mean_pic,mean, scale_factor[(config.position_animal, config.position_space)])}')
        dog_cdf=np.array(cdf_record)[[0,2,4,6]]
        cat_cdf=np.array(cdf_record)[[1,3,5,7]]
        delta=dog_cdf-cat_cdf
        word=''
        word_1=word_2=word_3=word_4=word_5=''

        word_list = ['eyes', 'legs', 'ears', 'noses']
        is_list=['are','is','are']
        dog_cdf_all = dog_cdf[0] + dog_cdf[1] + dog_cdf[2] + dog_cdf[3]
        cat_cdf_all = cat_cdf[0] + cat_cdf[1] + cat_cdf[2] + cat_cdf[3]

        # if label_true[0,1]>=label_true[0,0]:
        if dog_cdf_all>cat_cdf_all:  # label_true[0,1]>=label_true[0,0]:   # dog
            ## new
            if cat_cdf_all < 0.1 and dog_cdf_all < 0.1:
                word += "From this perspective, I’m not sure whether this is a dog."

            if np.max(delta)<0.2 and (dog_cdf[0] < 0.2 and dog_cdf[1] < 0.2 and dog_cdf[2] < 0.2) or (dog_cdf[0] < 0.2 and dog_cdf[1] < 0.2 and dog_cdf[3] < 0.2)\
                    or (dog_cdf[1] < 0.2 and dog_cdf[2] < 0.2 and dog_cdf[3] < 0.2) or (dog_cdf[0] < 0.2 and dog_cdf[2] < 0.2 and dog_cdf[3] < 0.2):
                # word+='It might be a dog, but I am not sure.'
                pass
            else:
                if np.max(delta) > 0.5 and (dog_cdf[0] > 0.5 or dog_cdf[1] > 0.5 or dog_cdf[2] > 0.5) or (dog_cdf[0] > 0.5 or dog_cdf[1] > 0.5 or dog_cdf[3] > 0.5) \
                        or (dog_cdf[1] > 0.5 or dog_cdf[2] > 0.5 or dog_cdf[3] > 0.5) or (dog_cdf[0] > 0.5 or dog_cdf[2] > 0.5 or dog_cdf[3] > 0.5):
                    word+='I am sure it is a dog mainly because '
                else:
                    word+='It is probably a dog mainly because'
                for m in range(len(word_list)):
                    if dog_cdf[m] > 0.5 and delta[m] > 0.5:
                        if word_1=='':
                            word_1+=word_list[m]
                        else:
                            word_1 +=' and '
                            word_1+=word_list[m]
                    if dog_cdf[m] > 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_2 == '':
                            word_2 += word_list[m]
                        else:
                            word_2+= ' and '
                            word_2 += word_list[m]
                    if dog_cdf[m] > 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_3 == '':
                            word_3 += word_list[m]
                        else:
                            word_3 += ' and '
                            word_3 += word_list[m]
                    if 0.2 < dog_cdf[m] < 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_4 == '':
                            word_4 += word_list[m]
                        else:
                            word_4 += ' and '
                            word_4 += word_list[m]
                    if 0.2 < dog_cdf[m] < 0.5 and 0.2 < delta[m] <= 0.35:
                        if word_5 == '':
                            word_5 += word_list[m]
                        else:
                            word_5 += ' and '
                            word_5 += word_list[m]
                print(word_1,word_2,word_3,word_4,word_5)
                #m=np.where(delta==np.max(delta))[0][0]
                flag=0
                if word_1!='':
                    is_are='are'
                    if word_1=='legs':
                        is_are='is'
                    word+=f"it has vivid {word_1}, which {is_are} dog's {word_1} obviously. "
                    flag=1
                if word_2 != '':
                    is_are = 'are'
                    if word_2 == 'legs':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has vivid {word_2}, which {is_are} something like dog's {word_2}. "
                    if flag==0:
                        word += f"it has vivid {word_2}, which {is_are} something like dog's {word_2}. "

                if word_3 != '':
                    is_are = 'are'
                    if word_3 == 'legs':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has vivid {word_3}, which {is_are} perhaps dog's {word_3}. "
                    if flag==0:
                        word += f"it has vivid {word_3}, which {is_are} perhaps dog's {word_3}. "

                if word_4 != '':
                    is_are = 'are'
                    if word_4 == 'legs':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has {word_4}, which {is_are} something like dog's {word_4}. "
                    if flag==0:
                        word += f"it has {word_4}, which {is_are} something like dog's {word_4}. "

                if word_5 != '':
                    is_are = 'are'
                    if word_5 == 'legs':
                        is_are = 'is'
                    if flag==1:
                        word+=f"Meanwhile, it has {word_5}, which {is_are} perhaps dog's {word_5}. "
                    if flag==0:
                        word += f"it has {word_5}, which {is_are} perhaps dog's {word_5}. "


                new_list=[]
                for n in range(4):
                    if dog_cdf[n]>0.5 and delta[n]<=0.2:
                        print(11)
                        new_list += [word_list[n]]
                        word += f"Although its {word_list[n]} {is_list[n]} a little confusing. "
                    if 0.2 < dog_cdf[n] < 0.5 and delta[n] <= 0.1:
                        print(111)
                        new_list += [word_list[n]]
                        word += f'In addition, it seems to have {word_list[n]}, which {is_list[n]} a little confusing.'

                # print(new_list)
                # if len(new_list) == 3:
                #     word += f"Although its {new_list[0]} and {new_list[1]} and {new_list[2]} are a little confusing. "
                # if len(new_list) == 2:
                #     word += f"Although its {new_list[0]} and {new_list[1]} are a little confusing. "
                # else:
                #     word += f"Although its {word_list[0]} is a little confusing. "

        # if label_true[0,0]<label_true[0,1]:
        if dog_cdf_all<cat_cdf_all: # label_true[0,0]<label_true[0,1]:
            delta=-delta
            if cat_cdf_all < 0.1 and dog_cdf_all < 0.1:
                word += "From this perspective, I’m not sure whether this is a cat."

            if np.max(delta) < 0.2 and (cat_cdf[0] < 0.2 and cat_cdf[1] < 0.2 and cat_cdf[2] < 0.2) or (cat_cdf[0] < 0.2 and cat_cdf[1] < 0.2 and cat_cdf[3] < 0.2)\
                    or (cat_cdf[1] < 0.2 and cat_cdf[2] < 0.2 and cat_cdf[3] < 0.2) or (cat_cdf[0] < 0.2 and cat_cdf[2] < 0.2 and cat_cdf[3] < 0.2):
                # word += 'It may be a cat, but I am not sure.'
                pass
            else:
                if np.max(delta) > 0.5 or (cat_cdf[0] > 0.5 or cat_cdf[1] > 0.5 or cat_cdf[2] > 0.5) or (cat_cdf[0] > 0.5 or cat_cdf[1] > 0.5 or cat_cdf[3] > 0.5)\
                    or (cat_cdf[1] > 0.5 or cat_cdf[2] > 0.5 or cat_cdf[3] > 0.5) or (cat_cdf[0] > 0.5 or cat_cdf[2] > 0.5 or cat_cdf[3] > 0.5):
                    word += 'I am sure it is a cat mainly because '
                else:
                    word += 'It is probably a cat mainly because '

                for m in range(len(word_list)):
                    if cat_cdf[m] > 0.5 and delta[m] > 0.5:
                        if word_1=='':
                            word_1+=word_list[m]
                        else:
                            word_1 +=' and '
                            word_1+=word_list[m]
                    if cat_cdf[m] > 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_2 == '':
                            word_2 += word_list[m]
                        else:
                            word_2+= ' and '
                            word_2 += word_list[m]
                    if cat_cdf[m] > 0.5 and 0.1 < delta[m] <= 0.35:
                        if word_3 == '':
                            word_3 += word_list[m]
                        else:
                            word_3 += ' and '
                            word_3 += word_list[m]
                    if 0.1 < cat_cdf[m] < 0.5 and 0.35 < delta[m] <= 0.5:
                        if word_4 == '':
                            word_4 += word_list[m]
                        else:
                            word_4 += ' and '
                            word_4 += word_list[m]
                    if 0.1 < cat_cdf[m] < 0.5 and 0.1 < delta[m] <= 0.35:
                        if word_5 == '':
                            word_5 += word_list[m]
                        else:
                            word_5 += ' and '
                            word_5 += word_list[m]
                print(word_1,word_2,word_3,word_4,word_5)

                flag = 0
                if word_1 != '':
                    is_are = 'are'
                    if word_1 == 'legs':
                        is_are = 'is'
                    word += f"it has vivid {word_1}, which {is_are} cat's {word_1} obviously. "
                    flag = 1
                if word_2 != '':
                    is_are = 'are'
                    if word_2 == 'legs':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has vivid {word_2}, which {is_are} something like cat's {word_2}. "
                    if flag == 0:
                        word += f"it has vivid {word_2}, which {is_are} something like cat's {word_2}. "

                if word_3 != '':
                    is_are = 'are'
                    if word_3 == 'legs':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has vivid {word_3}, which {is_are} perhaps cat's {word_3}. "
                    if flag == 0:
                        word += f"it has vivid {word_3}, which {is_are} perhaps cat's {word_3}. "

                if word_4 != '':
                    is_are = 'are'
                    if word_4 == 'legs':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has {word_4}, which {is_are} something like cat's {word_4}. "
                    if flag == 0:
                        word += f"it has {word_4}, which {is_are} something like cat's {word_4}. "

                if word_5 != '':
                    is_are = 'are'
                    if word_5 == 'legs':
                        is_are = 'is'
                    if flag == 1:
                        word += f"Meanwhile, it has {word_5}, which {is_are} perhaps cat's {word_5}. "
                    if flag == 0:
                        word += f"it has {word_5}, which {is_are} perhaps cat's {word_5}. "

            for n in range(4):
                if cat_cdf[n] > 0.5 and delta[n] <= 0.1:
                    word += f"Although its {word_list[n]} {is_list[n]} a little confusing. "
                    # if 0.2 < cat_cdf[n] < 0.5 and delta[n] <= 0.2:
                    #     word+=f'In addition, it seems to have {word_list[n]}, which {is_list[n]} a little confusing. '

        print(word)
        print(dog_cdf)
        print(cat_cdf)
        result=np.zeros([8])
        # result[0]=dog_cdf[1]
        # result[1]=dog_cdf[0]
        # result[2]=cat_cdf[2]
        # result[3]=cat_cdf[3]
        # result[4]=cat_cdf[1]
        # result[5]=cat_cdf[0]
        # result[6]=dog_cdf[2]
        # result[7]=dog_cdf[3]
        result[0] = cat_cdf[0]
        result[1] = cat_cdf[1]
        result[2] = cat_cdf[2]
        result[3] = cat_cdf[3]
        result[4] = dog_cdf[3]
        result[5] = dog_cdf[2]
        result[6] = dog_cdf[1]
        result[7] = dog_cdf[0]

        data_1 = cat_cdf
        data_2 = dog_cdf

        # plot_radian(result,file_name)
        # plot_BubbleChart(data_1, data_2, file_name)
        plot_BubbleChart(result, file_name)


def global_score(scores, multis=2):
    word = ''
    if multis == 5:
        Russian_Blue_cat_scores = scores[0]
        Siamese_cat_scores = scores[1]
        Bull_Terrier_dog_scores = scores[2]
        Weimaraner_dog_scores = scores[3]
        Pembroke_dog_scores = scores[4]

        # labels = ['cat1', 'cat2', 'dog1', 'dog2', 'dog3']
        labels = ['Russian Blue cat','Siamese cat', 'Bull Terrier dog', 'Weimaraner dog', 'Pembroke dog']
        # scores = Russian_Blue_cat_scores + Siamese_cat_scores + Bull_Terrier_dog_scores + Weimaraner_dog_scores + Pembroke_dog_scores

        
        bar_width = 0.3

        Russian_Blue_cat_color = '#8283b4'
        Siamese_cat_color = '#8283b4'
        Bull_Terrier_dog_color = '#8283b4'
        Weimaraner_dog_color = '#8283b4'
        Pembroke_dog_color = '#8283b4'
        # Russian_Blue_cat_color = '#EF6656'
        # Siamese_cat_color = '#62B975'
        # Bull_Terrier_dog_color = '#69A9CF'
        # Weimaraner_dog_color = '#FEA176'
        # Pembroke_dog_color = '#D0BDF0'

        plt.figure(figsize=(12, 18), dpi=300)  # H,W
        plt.style.use("seaborn-darkgrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.bar(labels, scores, width=bar_width, align='center', color=[Russian_Blue_cat_color, Siamese_cat_color, Bull_Terrier_dog_color, Weimaraner_dog_color, Pembroke_dog_color], label=('Russian_Blue_cat', 'Siamese_cat', 'Bull_Terrier_dog', 'Weimaraner_dog', 'Pembroke_dog'))
        plt.bar(labels[0], Russian_Blue_cat_scores, width=bar_width, align='center', color=Russian_Blue_cat_color) # label='Russian_Blue_cat'
        plt.bar(labels[1], Siamese_cat_scores, width=bar_width, align='center', color=Siamese_cat_color) # label='Siamese_cat'
        plt.bar(labels[2], Bull_Terrier_dog_scores, width=bar_width, align='center', color=Bull_Terrier_dog_color) # label='Bull_Terrier_dog'
        plt.bar(labels[3], Weimaraner_dog_scores, width=bar_width, align='center', color=Weimaraner_dog_color) # label='Weimaraner_dog'
        plt.bar(labels[4], Pembroke_dog_scores, width=bar_width, align='center', color=Pembroke_dog_color) # label='Pembroke_dog'

        plt.xlabel('Category', font2)
        plt.ylabel('Score', font2)
        plt.title('Global Scores of Cat and Dog Categories', font2, pad=30)
        for i in range(len(labels)):
            plt.text(labels[i], scores[i], f'{scores[i]:.2f}', ha='center', va='bottom',fontdict=font2)
        # plt.legend(('Russian_Blue_cat', 'Siamese_cat', 'Bull_Terrier_dog', 'Weimaraner_dog', 'Pembroke_dog'), loc='upper right')
        plt.legend(loc='upper right', fontsize=30)
        plt.xticks(range(len(labels)), labels, rotation=45, fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=30)
        plt.ylim(0, 5)
        plt.savefig(f'../lime_save_2kinds/global_scores.tiff', bbox_inches='tight', dpi=300)
        plt.show()

        index = np.argmax(scores)
        labels_1 = ['Russian_Blue_cat','Siamese_cat','Bull_Terrier_dog','Weimaraner_dog','Pembroke_dog']
        word += f"The {labels_1[index]} shows a higher semantic similarity score compared to all categories when considering the color and structural similarity of local semantic parts in this image."
        print(word)

    if multis == 2:
        cat_scores = scores[0]
        dog_scores = scores[1]


        labels = ['cat', 'dog']
        # scores = cat_scores + dog_scores

        bar_width = 0.3

        # cat_color = '#BBCEC8'
        # dog_color = '#8F94A6'
        cat_color = '#8283B4'
        dog_color = '#8283B4'

        plt.figure(figsize=(6, 10), dpi=300)
        plt.style.use("seaborn-darkgrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # plt.bar(labels, scores, width=bar_width, align='center', color=[Russian_Blue_cat_color, Siamese_cat_color, Bull_Terrier_dog_color, Weimaraner_dog_color, Pembroke_dog_color], label=('Russian_Blue_cat', 'Siamese_cat', 'Bull_Terrier_dog', 'Weimaraner_dog', 'Pembroke_dog'))
        plt.bar(labels[0], cat_scores, width=bar_width, align='center', color=cat_color, label='cat')
        plt.bar(labels[1], dog_scores, width=bar_width, align='center', color=dog_color, label='dog')

        plt.xlabel('Category', font2)
        plt.ylabel('Score', font2)
        plt.title('Global Scores of Categories', font2, pad=20)

        for i in range(len(labels)):
            plt.text(i, scores[i], f'{scores[i]:.2f}', ha='center', va='bottom',fontdict=font2)
        # for i in range(len(labels)):
        #     plt.text(labels[i], scores[i], f'{scores[i]:.2f}', ha='center', va='bottom')
        # plt.legend(loc='upper right', fontsize=8)
        plt.xticks(fontsize=20)
        plt.ylim(0, 5)
        plt.yticks(fontsize=20)
        plt.savefig(f'../lime_save_2kinds/global_scores.tiff', bbox_inches='tight', dpi=300)
        # plt.show()

        if cat_scores > dog_scores:
            word += f"The {labels[0]} shows a higher semantic similarity score compared to all categories when considering the color and structural similarity of local semantic parts in this image."
        else:
            word += f"The {labels[1]} shows a higher semantic similarity score compared to all categories when considering the color and structural similarity of local semantic parts in this image."
        print(word)


def identify_adverse(conv_out,reshape_num):
    config.dataset_situation = 'val'
    adverse_sample = torch.from_numpy(
        np.load('../result_save/cat_fake.npy').astype(np.float32))
    true_sample = torch.from_numpy(
        np.load('../result_save/cat_true.npy').astype(np.float32))
    attack_accuracy=0
    defence_accuray=0
    for i in range(7):
        print('----------第%d次迭代----------' % (i))
        image = adverse_sample[i]
        image_true=true_sample[i]
        image = Variable(image.to(device), requires_grad=True)
        label_true,_, x_feature = use_model(image)
        _, prediction = torch.max(label_true.data, 1)
        if prediction==1:
            attack_accuracy+=1
        print(transform_pro(label_true))
        act = conv_out.features
        x_feature = act.to(device).reshape(reshape_num)
        feature_space = x_feature
        pic = image.cpu().data
        img = transform_raw_picture(pic)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # if i==6:
        #         pic=image.cpu().data
        #         img=transform_raw_picture(pic)
        #         plt.imshow(img)
        #         break

        cdf_record = []
        for i in ['eye', 'leg', 'ear', 'nose']:
            for j in ['dog', 'cat']:
                config.position_space = i
                config.position_animal = j
                _, _, space_index, space_value = get_position_2(conv_out, 5, show_picture=False)
                space_index = np.array(space_index, dtype=int)
                space = np.load(f'../result_save/{config.position_animal}_space_2000.npy')

                eye_space = space[:, space_index]
                sum = 0
                for k in range(eye_space.shape[1]):
                    sum += eye_space[:, k] * space_value[k]
                mean = sum / eye_space.shape[1]

                eye_space = feature_space[space_index].cpu().data.numpy()
                sum = 0
                for k in range(eye_space.shape[0]):
                    sum += eye_space[k] * space_value[k]
                mean_pic = sum / eye_space.shape[0]
                cdf_record.append(get_cdf_ratio(mean_pic, mean,scale_factor=False))
                print(f'{j},{i},{get_cdf_ratio(mean_pic, mean,scale_factor=False)}')
        dog_cdf=np.array(cdf_record)[[0,2,4,6]]
        if (dog_cdf>0.99).any() and np.sum(dog_cdf>0.9)>=2:
            defence_accuray+=1

        print(attack_accuracy,defence_accuray)


ppnet = torch.load('../saved_models/cat_vs_dog_scq/vgg19/hsvtest/0nopush1.0000.pth')
# ppnet = torch.load('../saved_models/oxford_102_flower_dataset/vgg19/test/0nopush0.7931.pth') 
# ppnet = torch.load('../saved_models/cat_dog_scq/vgg19/_ortest/0nopush0.9927.pth') 
# ppnet = torch.load('../saved_models/cat_vs_dog_scq_2/resnet50/test/0nopush0.9845.pth')
# ppnet = torch.load('../saved_models/cat_vs_dog_scq_2/densenet121/test/0nopush1.0000.pth')

use_model = ppnet.cuda()
# use_model = use_model.features
print(use_model)
conv_out = LayerActivations(use_model.features[2], config.visual_layer) # vgg
# conv_out = LayerActivations(use_model.features[-1], config.visual_layer) # resnet
# conv_out = LayerActivations(use_model[-1], config.visual_layer)  # .features[0][9][3]


# generate_feature_matrix(use_model, conv_out, 324, lime=True)

# plot_lime_PCA(use_model, conv_out, N_select=512)

# plot=PLOT(conv_out,512)
# plot.plot_position() 
# plot.plot_one_sample()

# val_distribution(conv_out,512)  # vgg
# val_distribution(conv_out,1024)

# sort_index_origin,sort_index_position,space_index,space_value=get_position(conv_out,5,show_picture=False)
# _,_,space_index_auto,space_value_auto=get_position_2(conv_out,5,show_picture=False)
# plot_distribution(space_index,space_index_auto,space_value,space_value_auto,picture='small')

# get_inverse_position(conv_out, N_select = 512)


print_word(conv_out,512)

# global score 语句自动生成
# scores = [3.65, 2.94]
# scores = [3.27, 4.11]
# scores = [3.3686, 2.5546]
# global_score(scores, multis=2)
# scores = [3.90, 3.64, 2.95, 3.23, 3.22]  # 1
# scores = [2.73, 2.85, 3.31, 4.05, 3.45]  # 2
# global_score(scores, multis=5)

# identify_adverse(conv_out,reshape_num = 512)


# compare_different_num_inverse(use_model,conv_out, N_select = 512)
# plot_N()

# pca_scores(use_model)
