import Augmentor
import os
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

# datasets_root_dir = 'D:/datasets/afhq/train/'  # 'D:/TesNet-1/CUB_200_2011/datasets/cub200_cropped/'
# dir = 'D:/datasets/stanford dogs/cropped/train/'  # datasets_root_dir + 'train_cropped/'
# target_dir = 'D:/datasets/stanford dogs/cropped/train_augment_1/' # datasets_root_dir + train_cropped_augmented_2/

# datasets_root_dir = 'D:/datasets/12_kind_cat/'  # 'D:/TesNet-1/CUB_200_2011/datasets/cub200_cropped/'
# dir =  datasets_root_dir + 'train/'  # datasets_root_dir + 'train_cropped/'
# target_dir = datasets_root_dir + '/train_augment/' # datasets_root_dir + train_cropped_augmented_2/

# datasets_root_dir = 'D:/datasets/tapir/'
# dir = datasets_root_dir + 'train'
# target_dir = datasets_root_dir + '/train_tapir_augment/'

datasets_root_dir = 'D:/datasets/Obfuscated dataset/'
dir = datasets_root_dir + 'train-zebra donkey'
target_dir = datasets_root_dir + 'train_zebra donkey_augment/'

# datasets_root_dir = 'D:/datasets/lion-tiger-dataset/'
# dir = datasets_root_dir + 'train'
# target_dir = datasets_root_dir + 'train_augment/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
print(folders)
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
# rotation 
p = Augmentor.Pipeline(source_directory=dir, output_directory=target_dir)
p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
p.flip_left_right(probability=0.5)
for i in range(4):
    p.process()
del p
# skew  
p = Augmentor.Pipeline(source_directory=dir, output_directory=target_dir)
p.skew(probability=1, magnitude=0.2)  # max 45 degrees
p.flip_left_right(probability=0.5)
for i in range(4):
    p.process()
del p
# shear 
p = Augmentor.Pipeline(source_directory=dir, output_directory=target_dir)
p.shear(probability=1, max_shear_left=10, max_shear_right=10)
p.flip_left_right(probability=0.5)
for i in range(4):
    p.process()
del p
#random_distortion
p = Augmentor.Pipeline(source_directory=dir, output_directory=target_dir)
p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
p.flip_left_right(probability=0.5)
for i in range(4):
   p.process()
del p
