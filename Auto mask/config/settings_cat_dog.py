img_size = 224
prototype_shape = (600, 64, 1, 1)  # (10): cub-2000/dog-1200
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

# cat_dog_scq
data_path = "D:/datasets/cat_vs_dog_scq/"  # 二分类
train_dir = data_path + 'train_augment/'
test_dir = data_path + 'counterfactual/'   # test, counterfactual
# test_dir = 'C:/Users/sunchangqi/Desktop/img/'
train_push_dir = data_path + 'train/'

# data_path = "D:/TesNet-1/test_img/vgg19/opl_test/3push0.9969.pth/"
# train_dir = data_path + 'train_augment/'
# test_dir = data_path + 'test/'   # test
# train_push_dir = data_path + 'train/'


# data_path = "D:/datasets/cat_dog_scq/"  # 五分类
# train_dir = data_path + 'train_cropped_augmented/'
# test_dir = data_path + 'test/'   # test
# train_push_dir = data_path + 'train/'

# stanford dogs
# data_path = "D:/datasets/stanford dogs/cropped/"  # 7分类
# train_dir = data_path + 'train_augment_small/'
# test_dir = data_path + 'test_small/'   # test
# train_push_dir = data_path + 'train_small/'



train_batch_size = 32
test_batch_size = 80
train_push_batch_size = 32

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
    'orth': 1e-4,
    'sub_sep': -1e-7,
    # 'opl': 1,
    # 'lpips': 1,
    # 'filter': 1,
    # 'diff':0.01
}

num_train_epochs = 10
num_warm_epochs = 1

push_start = 2
push_epochs = [i for i in range(num_train_epochs) if i % 1 == 0]

#print(push_epochs)



