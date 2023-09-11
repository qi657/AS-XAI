import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import shutil

import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from util.helpers import makedir
import core.push, models.model, core.train_and_test as tnt
# import push, model, train_and_test_or as tnt
from util import save
from util.log import create_logger
from util.preprocess import mean, std, preprocess_input_function
import config.settings_cat_dog as settings_cat_dog

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', type=str, default='0')
parser.add_argument('-arch', type=str, default='vgg19')  # resnet50, vgg19, densenet121, resnet152

parser.add_argument('-dataset', type=str, default="cat_vs_dog_scq_2")  # CUB, standford dog, cat_vs_dog_scq_2
parser.add_argument('-times', type=str, default="test", help="experiment_run")
parser.add_argument('-name', type=str, default="619", help="loss type")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
print(os.environ['CUDA_VISIBLE_DEVICES'])

# setting parameter
experiment_run = args.times
base_architecture = args.arch
dataset_name = args.dataset
name = args.name

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
# model save dir
model_dir = './saved_models/' + dataset_name + '/' + base_architecture + name + '/' + args.times + '/'

if os.path.exists(model_dir) is True:
    shutil.rmtree(model_dir)
makedir(model_dir)

shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings_cat_dog.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'models', base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the hyper param
if dataset_name == "standford dog" or "cat_vs_dog_scq_2":
    # model param
    num_classes = settings_cat_dog.num_classes
    img_size = settings_cat_dog.img_size
    add_on_layers_type = settings_cat_dog.add_on_layers_type
    prototype_shape = settings_cat_dog.prototype_shape
    prototype_activation_function = settings_cat_dog.prototype_activation_function
    # datasets
    train_dir = settings_cat_dog.train_dir
    train_push_dir = settings_cat_dog.train_push_dir
    test_dir = settings_cat_dog.test_dir
    train_batch_size = settings_cat_dog.train_batch_size
    test_batch_size = settings_cat_dog.test_batch_size
    train_push_batch_size = settings_cat_dog.train_push_batch_size
    # optimzer
    joint_optimizer_lrs = settings_cat_dog.joint_optimizer_lrs
    joint_lr_step_size = settings_cat_dog.joint_lr_step_size
    warm_optimizer_lrs = settings_cat_dog.warm_optimizer_lrs

    last_layer_optimizer_lr = settings_cat_dog.last_layer_optimizer_lr
    # weighting of different training losses
    coefs = settings_cat_dog.coefs
    # number of training epochs, number of warm epochs, push start epoch, push epochs
    num_train_epochs = settings_cat_dog.num_train_epochs
    num_warm_epochs = settings_cat_dog.num_warm_epochs
    push_start = settings_cat_dog.push_start
    push_epochs = settings_cat_dog.push_epochs

else:
    raise Exception("there are no settings file of datasets {}".format(dataset_name))

log(train_dir)

normalize = transforms.Normalize(mean=mean, std=std)

# train data

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=0, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=False)


log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))

log('batch size: {0}'.format(train_batch_size))

log("backbone architecture:{}".format(base_architecture))
log("basis concept size:{}".format(prototype_shape))

# construct the model
ppnet = models.model.construct_TesNet(base_architecture=base_architecture,
                                 pretrained=True, img_size=img_size,
                                 prototype_shape=prototype_shape,
                                 num_classes=num_classes,
                                 prototype_activation_function=prototype_activation_function,
                                 add_on_layers_type=add_on_layers_type)
# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
print(ppnet)
# exit(0)

class_specific = True

# define optimizer
from config.settings_cat_dog import joint_optimizer_lrs, joint_lr_step_size

joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
     # bias are now also being regularized
     {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
     ]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from config.settings_cat_dog import warm_optimizer_lrs

warm_optimizer_specs = \
    [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
     {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
     ]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from config.settings_cat_dog import last_layer_optimizer_lr

last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# best acc
best_acc = 0
best_epoch = 0
best_time = 0

# train the model
log('start training')
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))
    # stage 1: Embedding space learning
    # train
    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _, train_results = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                     class_specific=class_specific, coefs=coefs, log=log)  # data_loader_image["train_augment_1"]
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _, train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                                     class_specific=class_specific, coefs=coefs, log=log)

    # test
    accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                  class_specific=class_specific, log=log)



    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)
    # stage2: Embedding space transparency
    if epoch >= push_start and epoch in push_epochs:
        core.push.push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi,  # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                      class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)
        # stage3: concept based classification
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(5):
                log('iteration: \t{0}'.format(i))
                _, train_results= tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                             class_specific=class_specific, coefs=coefs, log=log)

                accu, test_results = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                              class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)

logclose()

