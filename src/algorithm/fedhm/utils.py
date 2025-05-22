import copy
import torch
import numpy as np

from collections import OrderedDict

def spectral_init(weight, rank):
    try:
        U, S, V = torch.svd(weight)
    except:
        print("====== UNSTABLE SPECTRAL DECOMPOSITION ======")
        U, S, V = torch.linalg.svd(weight.to('cuda'), driver='gesvd')

    sqrtS = torch.diag(torch.sqrt(S[:rank]))

    u_weight_sliced, v_weight_sliced = torch.matmul(U[:, :rank], sqrtS), torch.matmul(
        V[:, :rank], sqrtS).T

    return u_weight_sliced, v_weight_sliced

def decide_upper_bound(rank_factor, freeze_bn, model_name):
    scaler = 1 if freeze_bn else 2

    if model_name == 'ResNet50':
        upper_bound = 33 * scaler
        skip_name = 'downsample'
        decompose_name = 'conv'
    elif model_name == 'ResNet18':
        upper_bound = 8 * scaler
        skip_name = 'downsample'
        decompose_name = 'conv'
    elif model_name == 'ResNet10':
        upper_bound = 8 * scaler
        skip_name = 'downsample'
        decompose_name = 'conv'
    elif model_name == 'ResNet34':
        upper_bound = 48 * scaler
        skip_name = 'downsample'
        decompose_name = 'conv'
    else:
        upper_bound = 1
        skip_name = 'transition'
        decompose_name = 'conv1'

    if rank_factor > 64:
        upper_bound = 1 * scaler

    return decompose_name, skip_name, upper_bound

def get_conv_index(fullrank_model, lowrank_model, rank_factor, freeze_bn, model_name):
    index_map = OrderedDict()

    if rank_factor <= 1:
        return index_map
    
    decompose_name, skip_name, upper_bound = decide_upper_bound(rank_factor, freeze_bn, model_name)
    master_conv_index, master_other_indx = [], []

    for index, (param_name, param) in enumerate(fullrank_model.state_dict().items()):
        if 'conv' in param_name and skip_name not in param_name \
                and index not in range(0, upper_bound):
            master_conv_index.append(index)
        else:
            master_other_indx.append(index)

    local_conv_indx, local_other_indx = [], []
    for index, (param_name, param) in enumerate(lowrank_model.state_dict().items()):
        if 'bn1_u' in param_name or 'bn2_u' in param_name:
            continue
        if 'conv' in param_name and index not in range(0, upper_bound) and skip_name not in param_name:
            local_conv_indx.append(index)
        else:
            local_other_indx.append(index)

    local_index = 0
    for index in master_conv_index:
        index_map[index] = local_conv_indx[local_index]
        local_index += 2

    local_index = 0
    for index in master_other_indx:
        index_map[index] = local_other_indx[local_index]
        local_index += 1
    return index_map

def split_conv(fullrank_model, lowrank_model, rank_factor, freeze_bn, model_name):
    fullrank_state = fullrank_model.state_dict()
    decompose_name, skip_name, upper_bound = decide_upper_bound(rank_factor, freeze_bn, model_name)
    
    reconstructed = list(lowrank_model.state_dict().values())
    index_map = get_conv_index(fullrank_model, lowrank_model, rank_factor, freeze_bn, model_name)

    for idx, (param_name, param) in enumerate(fullrank_state.items()):
        if 'conv' in param_name and idx not in range(0, upper_bound) \
                and skip_name not in param_name:
            out_channels, in_channels, ks1, ks2 = param.shape
            dim1, dim2 = out_channels * ks1, in_channels * ks2
            param_reshaped = param.permute(0, 2, 1, 3).reshape(dim1, dim2)

            if decompose_name not in param_name:
                sliced_rank = out_channels
            else:
                sliced_rank = max(int(round(out_channels / rank_factor)), 1)

            u_weight_sliced, v_weight_sliced = spectral_init(param_reshaped, sliced_rank)

            reconstructed[index_map[idx]] = v_weight_sliced
            reconstructed[index_map[idx] + 1] = u_weight_sliced
        else:
            reconstructed[index_map[idx]] = param
    

    reload_state_dict = {}

    for item_index, (param_name, param) in enumerate(lowrank_model.state_dict().items()):
        reload_state_dict[param_name] = param
        if 'bn1_u' in param_name or 'bn2_u' in param_name:
            continue
        reload_state_dict[param_name] = reconstructed[item_index]

    return reload_state_dict