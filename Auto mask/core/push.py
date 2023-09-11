import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from ..util.receptive_field import compute_rf_prototype
from ..util.helpers import makedir, find_high_activation_crop


def generate_irregular_mask(activation_map, threshold):
    # 创建二值掩码
    mask = np.zeros_like(activation_map)
    mask[activation_map >= threshold] = 255
    mask = np.uint8(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    irregular_mask = np.zeros_like(activation_map)

    cv2.drawContours(irregular_mask, contours, -1, 255, thickness=cv2.FILLED)

    return irregular_mask

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

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset   0：整个数据集中的图像索引
    1: height start index                  1：高度起始索引
    2: height end index                    2：高度结束指数
    3: width start index                   3：宽度起始索引
    4: width end index                     4：宽度结束索引
    5: (optional) class identity           5：（可选）类标识
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes #保存的地址
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch, _ = prototype_network_parallel.module.push_forward(search_batch)  # tesnet原始

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())#[batchsize,128,14,14]
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())#[batchsize,2000,14,14]

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0] #2000
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):#2000
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:] #
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:] #[batchsize,14,14]

        batch_min_proto_dist_j = np.amin(proto_dist_j) #
        if batch_min_proto_dist_j < global_min_proto_dist[j]: #
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))#
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0] #
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h #1
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w #+1

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch #
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item() #

            # find the highly activated region of the original image 找到原始图像的高度激活区域
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            proto_act_img_j = - proto_dist_img_j
            # if prototype_network_parallel.module.prototype_activation_function == 'log':
            #     proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            # elif prototype_network_parallel.module.prototype_activation_function == 'linear':
            #     proto_act_img_j = max_dist - proto_dist_img_j
            # else:
            #     proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)

            # new code
            # 假设你已经有了激活图 activation_map 和原始图像 original_image
            activation_map = upsampled_act_img_j  # 激活图
            original_image = original_img_j  # 原始图像

            percentile = 97  # 阈值
            threshold = np.percentile(activation_map, percentile)

            # 生成不规则掩码
            irregular_mask_1 = generate_irregular_mask(activation_map, threshold)
            irregular_mask_2 = generate_irregular_mask(activation_map, threshold)
            irregular_mask_3 = generate_irregular_mask(activation_map, threshold)
            # plt.imshow(irregular_mask)
            # plt.show()

            # 应用不规则掩码到原始图像
            masked_image_1 = np.copy(original_image)
            masked_image_1[irregular_mask_1 == 0] = 0  # 黑白的高激活区域（用来做mask）

            masked_image_2 = np.copy(original_image)
            masked_image_2[irregular_mask_2 == 0] = masked_image_2[irregular_mask_2 == 0] * 0.7  # 调整亮度值   可视化（贴图用）

            masked_image_3 = np.copy(original_image)
            masked_image_3[irregular_mask_3 == 255] = [255, 255, 0]

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

            masked_image_2 = cv2.cvtColor(masked_image_2, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(dir_for_saving_prototypes,
                        prototype_img_filename_prefix + '-self_act' + str(j) + '.png'), masked_image_2 * 255)
            cv2.imwrite(os.path.join(dir_for_saving_prototypes, prototype_img_filename_prefix + '-new_mask' + str(j) + '.png'),
                        masked_image_3[:, :, [2, 1, 0]] * 255)


            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image  裁剪出具有高激活度的图像块作为原型图像
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)  保存原型边界（高度激活区域的矩形边界）
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png  将包含原型的整个图像保存为png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result  在原始图像上覆盖（上采样）自激活并保存结果
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png   如果与原始（整个）图像不同，请将原型感受野保存为png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)  保存原型图像（整个图像的高度激活区域）
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)


                    imsave_with_bbox(fname=os.path.join(dir_for_saving_prototypes,
                                                        prototype_img_filename_prefix + '_mask'+str(j) + '.png'),
                                     img_rgb=original_img_j,
                                     bbox_height_start=proto_bound_j[0],
                                     bbox_height_end=proto_bound_j[1],
                                     bbox_width_start=proto_bound_j[2],
                                     bbox_width_end=proto_bound_j[3], color=(0, 255, 255))

                    # new code
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-new_act' + str(j) + '.png'),
                               masked_image_1,
                               vmin=0.0,
                               vmax=1.0)

                    # plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                         prototype_img_filename_prefix + '-new_mask' + str(j) + '.png'),
                    #            masked_image_3)
                
    if class_specific:
        del class_to_img_index_dict
