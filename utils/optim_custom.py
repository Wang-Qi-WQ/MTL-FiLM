from typing import Union

import torch.optim as optim


def register_adam_optimizer(model, lr, sub_modules: list = None, sub_lr: dict = None):
    params = list(model.named_parameters())

    if sub_modules is None:
        if sub_lr is None:
            sub_modules = []
            for name, m in model.named_modules():
                if name.find('.') != -1 or len(name) == 0:
                    continue
                sub_modules.append(name)
            sub_lr = {sub_name: lr for sub_name in sub_modules}
        else:
            sub_modules = [*sub_lr.keys()]
    if sub_lr is None:
        assert type(sub_modules) == list
        sub_lr = {sub_name: lr for sub_name in sub_modules}

    params_group = []
    params_group_name = []
    for sub_name in sub_modules:
        g = []
        g_name = []
        for n, p in params:
            # if sub_keywords is None:
            if sub_name in n:
                g.append(p)
                g_name.append(n)
        params_group.append({'params': g, 'lr': sub_lr[sub_name]})
        # params_group_name is only for debugging
        params_group_name.append({'params': g_name})

    optimizer = optim.Adam(params_group, lr=lr)
    return optimizer


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


def adjust_learning_rate(optimizer, lr: Union[float, list]):
    if type(lr) == float:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif type(lr) == list:
        if len(lr) == len(optimizer.param_groups):
            for idx, lr_idx in enumerate(lr):
                optimizer.param_groups[idx]['lr'] = lr_idx
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr[0]
    else:
        raise ValueError(f'unexpect lr type: {type(lr)}')
