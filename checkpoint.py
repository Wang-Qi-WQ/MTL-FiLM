import os
from typing import Dict, Union

import torch
import dataclasses
import datetime
import collections

from constant_macro import *
from utils import func, optim_custom


@dataclasses.dataclass
class ModelInfo:
    model_dict: Union[Dict, collections.OrderedDict]
    optim_dict: dict
    model_name: str
    dataset_name: str
    train_time: datetime
    train_epoch: int
    train_batch_idx: int
    train_batch_tp_fp_fn: dict
    train_batch_metrics: dict


def resume(model_path, specify_name: str = ''):
    if specify_name in os.listdir(model_path):
        return torch.load(os.path.join(model_path, specify_name))
    raise ValueError(f'Unknown model checkpoint: {specify_name}')


def reload_dict(new_dict, pretrained_dict, excluded=None):
    if type(pretrained_dict[list(pretrained_dict)[0]]) == torch.Tensor:
        popped_keys = set()
        for key, value in list(pretrained_dict.items()):
            if excluded is not None:
                for excluded_item in excluded:
                    if excluded_item in key and key not in popped_keys:
                        pretrained_dict.pop(key)
                        popped_keys.add(key)
            elif (key in new_dict) and (new_dict[key].shape == pretrained_dict[key].shape):
                pass
            else:
                print(f'Unexpected model parameter from loaded dict & shape: '
                      f'{key}. [{pretrained_dict[key].shape}]')
                pretrained_dict.pop(key)
    else:
        pretrained_dict = {key: value for key, value in pretrained_dict.items()
                           if key in new_dict}
    new_dict.update(pretrained_dict)
    return new_dict


def save(model_path, model_info: ModelInfo, batch_idx: int = -1):
    train_time = model_info.train_time.strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
    if model_info.dataset_name == 'maestro-v1.0.0':
        ds_name = 'MV1'
    elif model_info.dataset_name == 'maestro-v3.0.0':
        ds_name = 'MV3'
    elif model_info.dataset_name == 'maestro-v2.0.0':
        ds_name = 'MV2'
    else:
        raise ValueError(f'Unexpected dataset: {model_info.dataset_name}')
    if batch_idx != -1:
        torch.save(model_info, f'{model_path}{train_time}_{ds_name}_{model_info.model_name}_ep{model_info.train_epoch}'
                               f'_bt{batch_idx + 1}.pt')
    else:
        torch.save(model_info, f'{model_path}{train_time}_{ds_name}_{model_info.model_name}_ep{model_info.train_epoch}'
                               f'.pt')


def fine_tuning(model, optimizer, load_len, bm_info, sub_modules):
    pt_info = resume(CKP_PATH, CHECKPOINT)
    pt_opt_info = pt_info
    if pt_info is None:
        raise RuntimeError('Target Checkpoint Does Not Exist')
    assert type(pt_info) is ModelInfo

    if REFRESH_EPOCH_NUMBER:
        epoch_start = 1
    else:
        epoch_start = pt_info.train_epoch + 1
        if epoch_start > EPOCH_RANGE:
            raise ValueError(f'fine-tuning start epoch: {epoch_start} is out of epoch-range: {EPOCH_RANGE}')

    if hasattr(pt_info, 'dataset_name'):
        pt_info.dataset_name = DATASET_NAME

    pt_tt = pt_info.train_time.strftime('%y_%m_%d_%H_%M_%S').replace(':', '_')
    print(f'Fine-tuning From: {pt_tt}_{pt_info.model_name}_ep{pt_info.train_epoch}')
    model.load_state_dict(reload_dict(model.state_dict(), pt_info.model_dict, excluded=FT_EXCLUDED))

    # model name
    model_name = pt_info.model_name
    if NEW_MODEL_NAME:
        model_name = type(model).__name__ + MODEL_NAME_APPENDIX
        if type(model).__name__ == 'DataParallel':
            model_name = type(model.module).__name__ + MODEL_NAME_APPENDIX

    lr = LEARNING_RATE
    reload_step = 0
    if not NEW_OPTIMIZER:
        optimizer.load_state_dict(reload_dict(optimizer.state_dict(), pt_opt_info.optim_dict))
        lr_list = optim_custom.get_learning_rate(optimizer)
        lr_list = [LEARNING_RATE for x in lr_list]
        lr = lr_list[0]
        optim_custom.adjust_learning_rate(optimizer, lr_list)
        reload_step = pt_info.train_epoch * load_len
    print(f'New Optimizer: {NEW_OPTIMIZER}')
    print(f'Optimizer Registered Modules: {sub_modules if sub_modules is not None else "All Modules"}')

    lr_scheduler = func.StairDownLR(optimizer=optimizer, lr=lr, decay=0.02, stair_length=STEP_STAIR_LENGTH,
                                    reload_step=reload_step)

    lr = lr_scheduler.lr()
    print(f'Learning Rate: {lr:.10f}')

    train_time = None
    if not NEW_LOG_TIME:
        train_time = pt_info.train_time
    if hasattr(pt_info, 'train_batch_idx') and hasattr(pt_info, 'train_batch_metrics'):
        if not NEW_DATA_EPOCH:
            bm_info['frm'].reload_batch_train_info(pt_info.train_batch_idx,
                                                   pt_info.train_batch_tp_fp_fn['frame'],
                                                   pt_info.train_batch_metrics['frame'])
            bm_info['ons'].reload_batch_train_info(pt_info.train_batch_idx,
                                                   pt_info.train_batch_tp_fp_fn['onset'],
                                                   pt_info.train_batch_metrics['onset'])
            bm_info['off'].reload_batch_train_info(pt_info.train_batch_idx,
                                                   pt_info.train_batch_tp_fp_fn['offset'],
                                                   pt_info.train_batch_metrics['offset'])
            bm_info['vel'].reload_batch_train_info(pt_info.train_batch_idx,
                                                   pt_info.train_batch_tp_fp_fn['velocity'],
                                                   pt_info.train_batch_metrics['velocity'])
    tf_obj = {'pt_info': pt_info,
              'model': model,
              'model_name': model_name,
              'lr_scheduler': lr_scheduler,
              'epoch_start': epoch_start,
              'train_time': train_time}
    return tf_obj
