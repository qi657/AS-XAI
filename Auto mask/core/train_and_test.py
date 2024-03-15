import time
import torch
import torch.nn.functional as F

from ..util.helpers import list_of_distances
from ..util import ortho_conv

from ..core.loss import OrthogonalProjectionLoss

import cv2


def color_space_transform(img):
    img_np = img.cpu().detach().numpy()  
    img_np = img_np.transpose((1, 2, 0)) 
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)  
    img_np = img_np.transpose((2, 0, 1)) 
    img = torch.from_numpy(img_np) 
    return img


def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0

    total_cross_entropy = 0
    total_cluster_cost = 0
    total_orth_cost = 0
    total_subspace_sep_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_opl_cost = 0
    total_lpips_cost = 0
    total_filter_cost =0
    total_diff_cost = 0

    device = 'cuda'

    # oroth regular
    if True:  # vgg
        l_imp = {}
        for conv_ind in [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]:
            l_imp.update({conv_ind: model.module.features[0][conv_ind].bias.shape[0] ** (1 / 2)})
        normalizer = 0
        for key, val in l_imp.items():
            normalizer += val
        for key, val in l_imp.items():
            l_imp[key] = val / normalizer

        l_imp_1 = {}
        for conv_ind_1 in [0, 2]:
            l_imp_1.update({conv_ind_1: model.module.add_on_layers[conv_ind_1].bias.shape[0] ** (1 / 2)})
        normalizer = 0
        for key, val in l_imp_1.items():
            normalizer += val
        for key, val in l_imp_1.items():
            l_imp_1[key] = val / normalizer

    if False:  # resnet
        l_imp = {}
        for conv_ind in [0]:
            l_imp.update({conv_ind: model.module.features[conv_ind].out_channels ** (1 / 2)})
        for layer_ind in [4, 5, 6, 7]:
            for conv_ind in [0,1,2]:
                l_imp.update({conv_ind: model.module.features[layer_ind][conv_ind].conv1.out_channels ** (1 / 2)})
                l_imp.update({conv_ind: model.module.features[layer_ind][conv_ind].conv2.out_channels ** (1 / 2)})
                l_imp.update({conv_ind: model.module.features[layer_ind][conv_ind].conv3.out_channels ** (1 / 2)})
        normalizer = 0
        for key, val in l_imp.items():
            normalizer += val
        for key, val in l_imp.items():
            l_imp[key] = val / normalizer

        l_imp_1 = {}
        for conv_ind_1 in [0, 2]:
            l_imp_1.update({conv_ind_1: model.module.add_on_layers[conv_ind_1].bias.shape[0] ** (1 / 2)})
        normalizer = 0
        for key, val in l_imp_1.items():
            normalizer += val
        for key, val in l_imp_1.items():
            l_imp_1[key] = val / normalizer

    predicted_label = []
    target_label = []

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()
        target_label.append(target.data.cpu().numpy())

        # rgb_2_hsv

        # input = torch.stack([color_space_transform(img) for img in input])
        # cv2.imshow('HSV', input_data_color_transformed[1,:,:,:].numpy().transpose((1, 2, 0)))
        #
        # H, S, V = cv2.split(input_data_color_transformed[1,:,:,:].numpy().transpose((1, 2, 0)))
        # cv2.imshow('H', H)
        # cv2.imshow('S', S)
        # cv2.imshow('V', V)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print(H)
        # print(S)
        # print(V)
        # exit(0)


        #with autograd.detect_anomaly():
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, features = model(input)

            if False:
                ## ortho_conv loss
                ## resnet50
                # diff = ortho_conv.orth_dist(model.module.features[4][0].downsample[0].weight) + ortho_conv.orth_dist(model.module.features[5][0].downsample[0].weight) + ortho_conv.orth_dist(model.module.features[6][0].downsample[0].weight) + ortho_conv.orth_dist(model.module.features[7][0].downsample[0].weight)
                # diff += ortho_conv.deconv_orth_dist(model.module.features[4][0].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[4][1].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[4][2].conv2.weight, stride=1)
                # diff += ortho_conv.deconv_orth_dist(model.module.features[5][0].conv2.weight, stride=2) + ortho_conv.deconv_orth_dist(model.module.features[5][1].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[5][2].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[5][3].conv2.weight, stride=1)
                # diff += ortho_conv.deconv_orth_dist(model.module.features[6][0].conv2.weight, stride=2) + ortho_conv.deconv_orth_dist(model.module.features[6][1].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[6][2].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[6][3].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[6][4].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[6][5].conv2.weight, stride=1)
                # diff += ortho_conv.deconv_orth_dist(model.module.features[7][0].conv2.weight, stride=2) + ortho_conv.deconv_orth_dist(model.module.features[7][1].conv2.weight, stride=1) + ortho_conv.deconv_orth_dist(model.module.features[7][2].conv2.weight, stride=1)
                #
                # diff += ortho_conv.orth_dist(model.module.features[4][0].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[4][1].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[4][2].conv3.weight,stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[5][0].conv3.weight, stride=2) + ortho_conv.orth_dist(model.module.features[5][1].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[5][2].conv3.weight,stride=1) + ortho_conv.orth_dist(model.module.features[5][3].conv3.weight, stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[6][0].conv3.weight, stride=2) + ortho_conv.orth_dist(model.module.features[6][1].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[6][2].conv3.weight,stride=1) + ortho_conv.orth_dist(model.module.features[6][3].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[6][4].conv3.weight,stride=1) + ortho_conv.orth_dist(model.module.features[6][5].conv3.weight, stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[7][0].conv3.weight, stride=2) + ortho_conv.orth_dist(model.module.features[7][1].conv3.weight, stride=1) + ortho_conv.orth_dist(model.module.features[7][2].conv3.weight,stride=1)
                #
                # diff += ortho_conv.orth_dist(model.module.features[4][0].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[4][1].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[4][2].conv1.weight,stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[5][0].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[5][1].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[5][2].conv1.weight,stride=1) + ortho_conv.orth_dist(model.module.features[5][3].conv1.weight, stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[6][0].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[6][1].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[6][2].conv1.weight,stride=1) + ortho_conv.orth_dist(model.module.features[6][3].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[6][4].conv1.weight,stride=1) + ortho_conv.orth_dist(model.module.features[6][5].conv1.weight, stride=1)
                # diff += ortho_conv.orth_dist(model.module.features[7][0].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[7][1].conv1.weight, stride=1) + ortho_conv.orth_dist(model.module.features[7][2].conv1.weight,stride=1)

                ## vgg19
                diff = ortho_conv.orth_dist(model.module.features[0][2].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][5].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][7].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][10].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][12].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][14].weight, stride=1) + + ortho_conv.orth_dist(model.module.features[0][16].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][19].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][21].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][23].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][25].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][28].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][30].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][32].weight, stride=1) + ortho_conv.orth_dist(model.module.features[0][34].weight, stride=1)

                diff += ortho_conv.orth_dist(model.module.add_on_layers[0].weight, stride=1) + ortho_conv.orth_dist(model.module.add_on_layers[2].weight, stride=1)
                #####
                diff_loss = 0.01 * diff

                del diff


            del input
            # compute loss
            # cross entorpy loss  分类交叉熵损失
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            # functional call
            # lpips_loss = lpips(output, target, net_type='vgg', version='0.1')

            # opl
            op_loss = OrthogonalProjectionLoss(gamma=0.5)
            loss_op = op_loss(features, target)
            op_ce = cross_entropy + loss_op

            del op_loss
            del loss_op

            # filter ortho with conv
            L_angle = 0

            if True: # vgg
                ### Conv_ind == 0 ###
                w_mat = model.module.features[0][0].weight
                w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
                b_mat = model.module.features[0][0].bias
                b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
                params = torch.cat((w_mat1, b_mat1), dim=1)
                angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).cuda()
                L_angle += (l_imp[0]) * (angle_mat).norm(1)

                ### Conv_ind != 0 ###
                for conv_ind in [2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]:
                    w_mat = model.module.features[0][conv_ind].weight
                    w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
                    b_mat = model.module.features[0][conv_ind].bias
                    b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
                    params = torch.cat((w_mat1, b_mat1), dim=1)
                    angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(w_mat.shape[0]).cuda()
                    L_angle += (l_imp[conv_ind]) * (angle_mat).norm(1)
            else:  # resnet50
                ### Conv_ind == 0 ###
                w_mat = model.module.features[0].weight
                w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
                b_mat = model.module.features[0].bias
                b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
                params = torch.cat((w_mat1, b_mat1), dim=1)
                angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).cuda()
                L_angle += (l_imp[0]) * (angle_mat).norm(1)

                ### Conv_ind != 0 ###
                for conv_ind in [2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]:
                    w_mat = model.module.features[0][conv_ind].weight
                    w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
                    b_mat = model.module.features[0][conv_ind].bias
                    b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
                    params = torch.cat((w_mat1, b_mat1), dim=1)
                    angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(w_mat.shape[0]).cuda()
                    L_angle += (l_imp[conv_ind]) * (angle_mat).norm(1)

            filter_loss_0 = 1e-2 * L_angle

            del L_angle
            del w_mat
            del w_mat1
            del b_mat
            del b_mat1
            del params
            del angle_mat
            #
            # # filter ortho with add_on_layer
            # L_angle_1 = 0
            #
            # ### Conv_ind == 0 ###
            # w_mat = model.module.add_on_layers[0].weight
            # w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
            # b_mat = model.module.add_on_layers[0].bias
            # b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
            # params = torch.cat((w_mat1, b_mat1), dim=1)
            # angle_mat = torch.matmul(torch.t(params), params) - torch.eye(params.shape[1]).cuda()
            # L_angle_1 += (l_imp[0]) * (angle_mat).norm(1)
            #
            # ### Conv_ind != 0 ###
            # w_mat = model.module.add_on_layers[2].weight
            # w_mat1 = (w_mat.reshape(w_mat.shape[0], -1))
            # b_mat = model.module.add_on_layers[2].bias
            # b_mat1 = (b_mat.reshape(b_mat.shape[0], -1))
            # params = torch.cat((w_mat1, b_mat1), dim=1)
            # angle_mat = torch.matmul(params, torch.t(params)) - torch.eye(w_mat.shape[0]).cuda()
            # L_angle_1 += (l_imp[conv_ind]) * (angle_mat).norm(1)
            #
            # filter_loss_1 = 1e-2 * L_angle_1
            #
            # del L_angle_1
            # del w_mat
            # del w_mat1
            # del b_mat
            # del b_mat1
            # del params
            # del angle_mat
            #
            # # filter_loss = filter_loss_0 + filter_loss_1
            filter_loss = filter_loss_0
            # # filter_loss = filter_loss_1

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])


                subspace_max_dist = (model.module.prototype_shape[0]* model.module.prototype_shape[2]* model.module.prototype_shape[3]) #2000
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost  
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()  # [80, 2000] 
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)  
                cluster_cost = torch.mean(max_dist - inverted_distances)  

                # calculate separation cost  
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)



                # optimize orthogonality of prototype_vector 
                
                cur_basis_matrix = torch.squeeze(model.module.prototype_vectors) #[2000,64]
                subspace_basis_matrix = cur_basis_matrix.reshape(model.module.num_classes,model.module.num_prototypes_per_class,model.module.prototype_shape[1])#[200,10,64]
                subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2) #[200,10,64]->[200,64,10]
                orth_operator = torch.matmul(subspace_basis_matrix,subspace_basis_matrix_T)  # [200,10,64] [200,64,10] -> [200,10,10]
                I_operator = torch.eye(subspace_basis_matrix.size(1),subspace_basis_matrix.size(1)).cuda() #[10,10]  
                difference_value = orth_operator - I_operator #[200,10,10]-[10,10]->[200,10,10]
                orth_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0)) #[200]->[1]

                del cur_basis_matrix
                del orth_operator
                del I_operator
                del difference_value



                #subspace sep 格拉斯曼流形损失-类间分离
                projection_operator = torch.matmul(subspace_basis_matrix_T,subspace_basis_matrix)#[200,64,10] [200,10,64] -> [200,64,64]
                del subspace_basis_matrix
                del subspace_basis_matrix_T

                projection_operator_1 = torch.unsqueeze(projection_operator,dim=1)#[200,1,128,128]  B(c1)trans*B(c1)
                projection_operator_2 = torch.unsqueeze(projection_operator, dim=0)#[1,200,128,128]  B(c2)trans*B(c2)
                pairwise_distance = torch.norm(projection_operator_1-projection_operator_2+1e-10,p='fro',dim=[2,3]) #[200,200,128,128]->[200,200]
                subspace_sep = 0.5 * torch.norm(pairwise_distance,p=1,dim=[0,1],dtype=torch.double) / torch.sqrt(torch.tensor(2,dtype=torch.double)).cuda()
                del projection_operator_1
                del projection_operator_2
                del pairwise_distance


                ## row_ortho_loss
                # row_projection_operator = torch.matmul((subspace_basis_matrix-subspace_basis_matrix.mean(0, keepdim=True)),
                #                                    torch.transpose((subspace_basis_matrix-subspace_basis_matrix.mean(0, keepdim=True)),1,2))  # [200,64,10] [200,10,64] -> [200,64,64]
                # del subspace_basis_matrix
                #
                # row_projection_operator_1 = torch.unsqueeze(row_projection_operator, dim=1)  # [200,1,128,128]  B(c1)
                # row_projection_operator_2 = torch.unsqueeze(row_projection_operator, dim=0)  # [1,200,128,128]  B(c2)
                # torch.clamp(torch.sum(sigma ** 2) - self.margin, min=0)
                # pairwise_distance = torch.norm(row_projection_operator_1 - row_projection_operator_2 + 1e-10, p='fro', dim=[2, 3])  # [200,200,128,128]->[200,200]
                # subspace_sep = 0.5 * torch.norm(pairwise_distance, p=1, dim=[0, 1], dtype=torch.double) / torch.sqrt(
                #     torch.tensor(2, dtype=torch.double)).cuda()
                # del projection_operator_1
                # del projection_operator_2
                # del pairwise_distance



                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                    # weight 200,2000   prototype_class_identity [2000,200]
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)
            else:
                cluster_cost = torch.Tensor([0])
                separation_cost = torch.Tensor([0])
                l1 = torch.Tensor([0])
                orth_cost = torch.Tensor([0])
                subspace_sep = torch.Tensor([0])
                avg_separation_cost = torch.Tensor([0])
                filter_loss = torch.tensor(([0]))


            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            # predicted_label.append(predicted.data.cpu().numpy())
            #
            # print(predicted_label)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_orth_cost += orth_cost.item()
            total_subspace_sep_cost += subspace_sep.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            # total_opl_cost += op_ce.item()
            # total_lpips_cost += lpips_loss.item()
            # total_filter_cost += filter_loss.item()
            # total_diff_cost += diff_loss.item()

        # df = pd.DataFrame({'labels': predicted_label})
        # df.to_csv('./result_save/data_with_labels.csv', index=False)

        # df = pd.DataFrame({'labels': target_label})
        # df.to_csv('./result_save/data_with_target_labels.csv', index=False)


        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + coefs['orth'] * orth_cost
                          + coefs['sub_sep'] * subspace_sep
                          # + coefs['opl'] * op_ce
                          # + coefs['lpips'] * lpips_loss
                          # + coefs['filter'] * filter_loss
                          # + coefs['diff'] * diff_loss
                            )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1 + 1 * orth_cost - 1e-7 * subspace_sep
            # print(
            #     "{}/{} loss:{} cre:{} clst:{} sep:{} l1:{} orth:{} sub_sep:{}".format(i, len(dataloader), loss,cross_entropy, cluster_cost,separation_cost, l1,orth_cost, subspace_sep))

            # print("{}/{} loss:{} cre:{} clst:{} sep:{} l1:{} orth:{} sub_sep:{}".format(i,len(dataloader),loss,opl_ce,cluster_cost,separation_cost,l1,orth_cost,subspace_sep))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #nomalize basis vectors
            model.module.prototype_vectors.data = F.normalize(model.module.prototype_vectors, p=2, dim=1).data



        #del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\torth: \t{0}'.format(total_orth_cost / n_batches))
    log('\tsubspace_sep: \t{0}'.format(total_subspace_sep_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))

    # log('\topl: \t{0}'.format(total_opl_cost / n_batches))
    # log('\tlpips: \t{0}'.format(total_lpips_cost / n_batches))
    # log('\tfilter: \t{0}'.format(total_filter_cost / n_batches))
    # log('\tdiff: \t{0}'.format(total_diff_cost / n_batches))

    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    results_loss = {'cross_entropy': total_cross_entropy / n_batches,
                    'cluster_loss': total_cluster_cost / n_batches,
                    'orth_loss': total_orth_cost / n_batches,
                    'subspace_sep_loss': total_subspace_sep_cost / n_batches,
                    'separation_loss': total_separation_cost / n_batches,
                    'avg_separation': total_avg_separation_cost / n_batches,
                    'l1':model.module.last_layer.weight.norm(p=1).item(),
                    'p_avg_pair_dist':p_avg_pair_dist,
                    'accu' : n_correct/n_examples,
                    # 'opl': total_opl_cost / n_batches,
                    # 'lpips': total_lpips_cost / n_batches,
                    # 'filter': total_filter_cost / n_batches,
                    # 'diff': total_diff_cost / n_batches
                    }
    return n_correct / n_examples, results_loss


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')

    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = False
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
